import json

import httpx
import pytest
from pydantic import BaseModel

from elara.core.errors import ProviderError, ProviderUnavailable
from elara.database.audit import AuditLog
from elara.providers import (
    ChatMessage,
    CompletionRequest,
    LLMService,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from elara.providers.anthropic import AnthropicProvider
from elara.providers.openai import OpenAIProvider
from tests.fakes import ScriptedProvider, text


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


REQ = CompletionRequest(
    system="sys", model="m1", max_tokens=100,
    messages=[
        ChatMessage(role="user", text="hi"),
        ChatMessage(role="assistant", tool_calls=[ToolCall(id="t1", name="calc", arguments={"x": 1})]),
        ChatMessage(role="user", tool_results=[ToolResult(tool_call_id="t1", content="2")]),
    ],
    tools=[ToolSpec(name="calc", description="d", input_schema={"type": "object"})],
)


class TestAnthropic:
    async def test_request_shape_and_response_parsing(self):
        seen = {}

        def handler(req: httpx.Request):
            seen["headers"] = req.headers
            seen["body"] = json.loads(req.content)
            return httpx.Response(200, json={
                "model": "m1", "stop_reason": "tool_use",
                "content": [{"type": "thinking", "thinking": "", "signature": "s"},
                            {"type": "text", "text": "Checking."},
                            {"type": "tool_use", "id": "t2", "name": "calc", "input": {"x": 2}}],
                "usage": {"input_tokens": 12, "output_tokens": 7}})

        p = AnthropicProvider(_client(handler), api_key="sk-ant-test", base_url="https://x")
        resp = await p.complete(REQ)
        body = seen["body"]
        assert seen["headers"]["x-api-key"] == "sk-ant-test"
        assert seen["headers"]["anthropic-version"]
        assert body["system"] == "sys" and "temperature" not in body
        assert body["messages"][1]["content"][0]["type"] == "tool_use"
        assert body["messages"][2]["content"][0] == {"type": "tool_result", "tool_use_id": "t1",
                                                     "content": "2", "is_error": False}
        assert body["tools"][0]["input_schema"] == {"type": "object"}
        assert resp.text == "Checking."
        assert resp.tool_calls[0].name == "calc" and resp.tool_calls[0].arguments == {"x": 2}
        assert resp.usage.input_tokens == 12
        # Raw content (incl. thinking block) is replayed verbatim on the next turn.
        replay = p._message(resp.as_message("anthropic"))
        assert replay["content"][0]["type"] == "thinking"

    async def test_retries_on_529_then_succeeds(self):
        calls = {"n": 0}

        def handler(req):
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(529, json={"error": {"message": "overloaded"}})
            return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}],
                                             "usage": {}})

        p = AnthropicProvider(_client(handler), api_key="k", base_url="https://x",
                              max_retries=3, backoff_base_s=0)
        assert (await p.complete(REQ)).text == "ok"
        assert calls["n"] == 3

    async def test_auth_error_not_retried(self):
        calls = {"n": 0}

        def handler(req):
            calls["n"] += 1
            return httpx.Response(401, json={"error": {"message": "invalid x-api-key"}})

        p = AnthropicProvider(_client(handler), api_key="k", base_url="https://x",
                              max_retries=3, backoff_base_s=0)
        with pytest.raises(ProviderError, match="invalid API key"):
            await p.complete(REQ)
        assert calls["n"] == 1

    async def test_network_error_retried_then_raised(self):
        def handler(req):
            raise httpx.ConnectError("down")

        p = AnthropicProvider(_client(handler), api_key="k", base_url="https://x",
                              max_retries=2, backoff_base_s=0)
        with pytest.raises(ProviderError, match="network error"):
            await p.complete(REQ)

    async def test_streaming(self):
        sse = (
            'event: message_start\ndata: {"type":"message_start"}\n\n'
            'event: content_block_delta\ndata: {"delta":{"type":"text_delta","text":"Sa"}}\n\n'
            'event: content_block_delta\ndata: {"delta":{"type":"text_delta","text":"lam"}}\n\n'
            'event: message_stop\ndata: {"type":"message_stop"}\n\n')
        p = AnthropicProvider(_client(lambda r: httpx.Response(200, text=sse)), api_key="k",
                              base_url="https://x")
        assert "".join([c async for c in p.stream(REQ)]) == "Salam"


class TestOpenAI:
    async def test_request_shape_and_parsing(self):
        seen = {}

        def handler(req):
            seen["body"] = json.loads(req.content)
            seen["auth"] = req.headers["authorization"]
            return httpx.Response(200, json={
                "model": "m1",
                "choices": [{"finish_reason": "tool_calls", "message": {
                    "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
                        "name": "calc", "arguments": "{\"x\": 3}"}}]}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2}})

        p = OpenAIProvider(_client(handler), api_key="sk-test", base_url="https://o/v1")
        resp = await p.complete(REQ)
        msgs = seen["body"]["messages"]
        assert seen["auth"] == "Bearer sk-test"
        assert msgs[0] == {"role": "system", "content": "sys"}
        assert msgs[2]["tool_calls"][0]["function"]["name"] == "calc"
        assert msgs[3] == {"role": "tool", "tool_call_id": "t1", "content": "2"}
        assert resp.tool_calls[0].arguments == {"x": 3}
        assert resp.usage.output_tokens == 2

    async def test_streaming(self):
        sse = ('data: {"choices":[{"delta":{"content":"He"}}]}\n\n'
               'data: {"choices":[{"delta":{"content":"llo"}}]}\n\ndata: [DONE]\n\n')
        p = OpenAIProvider(_client(lambda r: httpx.Response(200, text=sse)), api_key="k",
                           base_url="https://o")
        assert "".join([c async for c in p.stream(REQ)]) == "Hello"


class Answer(BaseModel):
    answer: str
    confidence: float


class TestService:
    async def test_unavailable_without_provider(self, settings):
        svc = LLMService(None, settings)
        assert not svc.available
        with pytest.raises(ProviderUnavailable):
            await svc.complete(CompletionRequest(messages=[ChatMessage(role="user", text="x")]))

    async def test_model_selection_and_usage_recorded(self, settings, db):
        settings = settings.model_copy(update={"model_fast": "fast-m", "model_strong": "strong-m"})
        prov = ScriptedProvider([text("a"), text("b")])
        svc = LLMService(prov, settings, AuditLog(db))
        req = CompletionRequest(messages=[ChatMessage(role="user", text="x")])
        await svc.complete(req, tier="fast")
        await svc.complete(req, tier="strong")
        assert [r.model for r in prov.requests] == ["fast-m", "strong-m"]
        assert AuditLog(db).usage_totals()["calls"] == 2

    async def test_structured_output_with_repair(self, settings):
        settings = settings.model_copy(update={"model_fast": "m"})
        prov = ScriptedProvider([text("sure! {\"answer\": 1}"),
                                 text("```json\n{\"answer\": \"yes\", \"confidence\": 0.9}\n```")])
        svc = LLMService(prov, settings)
        out = await svc.structured(
            CompletionRequest(messages=[ChatMessage(role="user", text="q")]), Answer)
        assert out == Answer(answer="yes", confidence=0.9)
        assert "JSON schema" in prov.requests[0].system

    async def test_structured_output_gives_up(self, settings):
        settings = settings.model_copy(update={"model_fast": "m"})
        svc = LLMService(ScriptedProvider([text("no"), text("still no")]), settings)
        with pytest.raises(ProviderError):
            await svc.structured(
                CompletionRequest(messages=[ChatMessage(role="user", text="q")]), Answer)
