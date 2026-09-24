"""Anthropic Messages API provider (raw HTTP over httpx)."""

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

API_VERSION = "2023-06-01"


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, client: httpx.AsyncClient, *, api_key: str, base_url: str,
                 timeout_s: float = 90.0, **kw):
        super().__init__(client, **kw)
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key, "anthropic-version": API_VERSION,
                "content-type": "application/json"}

    def _payload(self, req: CompletionRequest, stream: bool = False) -> dict:
        payload: dict = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "messages": [self._message(m) for m in req.messages],
        }
        if req.system:
            payload["system"] = req.system
        if req.tools:
            payload["tools"] = [{"name": t.name, "description": t.description,
                                 "input_schema": t.input_schema} for t in req.tools]
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if stream:
            payload["stream"] = True
        return payload

    def _message(self, m: ChatMessage) -> dict:
        if m.role == "assistant" and m.raw is not None and m.raw_provider == self.name:
            return {"role": "assistant", "content": m.raw}  # faithful replay (thinking blocks)
        blocks: list[dict] = []
        for r in m.tool_results:
            blocks.append({"type": "tool_result", "tool_use_id": r.tool_call_id,
                           "content": r.content, "is_error": r.is_error})
        if m.text:
            blocks.append({"type": "text", "text": m.text})
        for c in m.tool_calls:
            blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments})
        return {"role": m.role, "content": blocks or [{"type": "text", "text": "."}]}

    async def _complete_once(self, req: CompletionRequest) -> CompletionResponse:
        data = await self._post_json(f"{self.base_url}/v1/messages", self._payload(req),
                                     self._headers(), self.timeout_s)
        content = data.get("content") or []
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        calls = [ToolCall(id=b["id"], name=b["name"], arguments=b.get("input") or {})
                 for b in content if b.get("type") == "tool_use"]
        usage = data.get("usage") or {}
        return CompletionResponse(
            text=text, tool_calls=calls, stop_reason=data.get("stop_reason"),
            usage=Usage(input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0)),
            model=data.get("model", req.model), raw=content,
        )

    async def stream(self, req: CompletionRequest) -> AsyncIterator[str]:
        url = f"{self.base_url}/v1/messages"
        try:
            async with self.client.stream("POST", url, json=self._payload(req, stream=True),
                                          headers=self._headers(), timeout=self.timeout_s) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise http_error(self.name, resp)
                async for event, data in iter_sse(resp):
                    if event == "error":
                        raise ProviderError(f"anthropic stream error: {data[:200]}",
                                            retryable=True)
                    if event != "content_block_delta":
                        continue
                    delta = json.loads(data).get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta.get("text", "")
        except httpx.TransportError as e:
            raise ProviderError(f"anthropic: network error: {type(e).__name__}",
                                retryable=True) from e

    async def check(self) -> str:
        try:
            resp = await self.client.get(f"{self.base_url}/v1/models", headers=self._headers(),
                                         timeout=10)
        except httpx.TransportError as e:
            raise ProviderError(f"anthropic: unreachable ({type(e).__name__})") from e
        if resp.status_code >= 400:
            raise http_error(self.name, resp)
        return "reachable, key accepted"
