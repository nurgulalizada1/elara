"""Test doubles. ScriptedProvider replays canned responses and records requests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

import httpx

from elara.core.errors import ProviderError
from elara.providers.base import CompletionRequest, CompletionResponse, LLMProvider, ToolCall, Usage

Responder = Callable[[CompletionRequest], CompletionResponse]


def text(t: str) -> CompletionResponse:
    return CompletionResponse(text=t, stop_reason="end_turn", usage=Usage(input_tokens=10,
                                                                          output_tokens=5))


def tool(name: str, args: dict, call_id: str = "call_1", say: str = "") -> CompletionResponse:
    return CompletionResponse(text=say, tool_calls=[ToolCall(id=call_id, name=name, arguments=args)],
                              stop_reason="tool_use", usage=Usage(input_tokens=10, output_tokens=5))


class ScriptedProvider(LLMProvider):
    name = "scripted"

    def __init__(self, script: list[CompletionResponse | Responder | Exception] | None = None):
        super().__init__(httpx.AsyncClient(), max_retries=0)
        self.script = list(script or [])
        self.requests: list[CompletionRequest] = []

    def push(self, *items) -> None:
        self.script.extend(items)

    async def _complete_once(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if not self.script:
            raise ProviderError("scripted provider: script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(request)
        return item

    async def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        resp = await self._complete_once(request)
        for word in resp.text.split(" "):
            yield word + " "

    async def check(self) -> str:
        return "scripted"
