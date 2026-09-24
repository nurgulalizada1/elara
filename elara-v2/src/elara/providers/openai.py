"""OpenAI Chat Completions provider (raw HTTP over httpx).

Also works with OpenAI-compatible local servers by pointing ELARA_OPENAI_BASE_URL at them.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from elara.core.errors import ProviderError
from elara.providers.base import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
    ToolCall,
    Usage,
    http_error,
    iter_sse,
)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, client: httpx.AsyncClient, *, api_key: str, base_url: str,
                 timeout_s: float = 90.0, **kw):
        super().__init__(client, **kw)
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self._api_key}", "content-type": "application/json"}

    def _messages(self, req: CompletionRequest) -> list[dict]:
        out: list[dict] = []
        if req.system:
            out.append({"role": "system", "content": req.system})
        for m in req.messages:
            out.extend(self._message(m))
        return out

    @staticmethod
    def _message(m: ChatMessage) -> list[dict]:
        msgs: list[dict] = [{"role": "tool", "tool_call_id": r.tool_call_id,
                             "content": ("ERROR: " if r.is_error else "") + r.content}
                            for r in m.tool_results]
        if m.role == "assistant":
            msg: dict = {"role": "assistant", "content": m.text or None}
            if m.tool_calls:
                msg["tool_calls"] = [{"id": c.id, "type": "function",
                                      "function": {"name": c.name,
                                                   "arguments": json.dumps(c.arguments)}}
                                     for c in m.tool_calls]
            msgs.append(msg)
        elif m.text:
            msgs.append({"role": "user", "content": m.text})
        return msgs

    def _payload(self, req: CompletionRequest, stream: bool = False) -> dict:
        payload: dict = {"model": req.model, "messages": self._messages(req),
                         "max_completion_tokens": req.max_tokens}
        if req.tools:
            payload["tools"] = [{"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.input_schema}}
                for t in req.tools]
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if stream:
            payload["stream"] = True
        return payload

    async def _complete_once(self, req: CompletionRequest) -> CompletionResponse:
        data = await self._post_json(f"{self.base_url}/chat/completions", self._payload(req),
                                     self._headers(), self.timeout_s)
        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError) as e:
            raise ProviderError("openai: malformed response") from e
        calls = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_invalid_json": tc["function"].get("arguments", "")}
            calls.append(ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=args))
        usage = data.get("usage") or {}
        return CompletionResponse(
            text=msg.get("content") or "", tool_calls=calls,
            stop_reason=choice.get("finish_reason"),
            usage=Usage(input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0)),
            model=data.get("model", req.model),
        )

    async def stream(self, req: CompletionRequest) -> AsyncIterator[str]:
        try:
            async with self.client.stream("POST", f"{self.base_url}/chat/completions",
                                          json=self._payload(req, stream=True),
                                          headers=self._headers(), timeout=self.timeout_s) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise http_error(self.name, resp)
                async for _event, data in iter_sse(resp):
                    if data.strip() == "[DONE]":
                        break
                    choices = json.loads(data).get("choices") or []
                    if choices and (delta := choices[0].get("delta", {}).get("content")):
                        yield delta
        except httpx.TransportError as e:
            raise ProviderError(f"openai: network error: {type(e).__name__}",
                                retryable=True) from e

    async def check(self) -> str:
        try:
            resp = await self.client.get(f"{self.base_url}/models", headers=self._headers(),
                                         timeout=10)
        except httpx.TransportError as e:
            raise ProviderError(f"openai: unreachable ({type(e).__name__})") from e
        if resp.status_code >= 400:
            raise http_error(self.name, resp)
        return "reachable, key accepted"
