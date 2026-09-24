"""Provider-neutral LLM types and the ``LLMProvider`` interface.

Providers translate these neutral types to/from their wire format. Everything above
this layer (agent, research synthesis) only sees these types.
"""

from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal, TypeVar

import httpx
from pydantic import BaseModel, Field

from elara.core.errors import ProviderError
from elara.core.logging import get_logger

log = get_logger(__name__)
T = TypeVar("T")


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_call_id: str
    content: str
    is_error: bool = False


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    # Provider-native content for faithful replay within a tool loop (e.g. thinking blocks).
    raw: Any = None
    raw_provider: str | None = None


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class CompletionRequest(BaseModel):
    messages: list[ChatMessage]
    system: str = ""
    tools: list[ToolSpec] = Field(default_factory=list)
    model: str = ""
    max_tokens: int = 4096
    temperature: float | None = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class CompletionResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    raw: Any = None

    def as_message(self, provider: str) -> ChatMessage:
        return ChatMessage(role="assistant", text=self.text, tool_calls=self.tool_calls,
                           raw=self.raw, raw_provider=provider)


RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class LLMProvider(ABC):
    """Base class. Subclasses implement the wire protocol; retry lives here."""

    name: str = "base"
    supports_tools: bool = True
    supports_streaming: bool = True

    def __init__(self, client: httpx.AsyncClient, *, max_retries: int = 3,
                 backoff_base_s: float = 0.5):
        self.client = client
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s

    @abstractmethod
    async def _complete_once(self, request: CompletionRequest) -> CompletionResponse: ...

    @abstractmethod
    def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        """Yield text deltas. Tool calls are not streamed."""

    @abstractmethod
    async def check(self) -> str:
        """Cheap connectivity/auth check (no token spend). Returns a short status."""

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return await self._retry(lambda: self._complete_once(request))

    async def _retry(self, fn: Callable[[], Awaitable[T]]) -> T:
        attempt = 0
        while True:
            try:
                return await fn()
            except ProviderError as e:
                if not e.retryable or attempt >= self.max_retries:
                    raise
                delay = getattr(e, "retry_after", None) or (
                    self.backoff_base_s * (2 ** attempt) + random.uniform(0, 0.25))
                log.warning("provider.retry", extra={"provider": self.name, "attempt": attempt + 1,
                                                     "status": e.status, "delay_s": round(delay, 2)})
                await asyncio.sleep(min(delay, 30))
                attempt += 1

    # Shared HTTP helpers --------------------------------------------------------------
    async def _post_json(self, url: str, payload: dict, headers: dict, timeout: float) -> dict:
        try:
            resp = await self.client.post(url, json=payload, headers=headers, timeout=timeout)
        except httpx.TimeoutException as e:
            raise ProviderError(f"{self.name}: request timed out", retryable=True) from e
        except httpx.TransportError as e:
            raise ProviderError(f"{self.name}: network error: {type(e).__name__}",
                                retryable=True) from e
        if resp.status_code >= 400:
            raise http_error(self.name, resp)
        try:
            return resp.json()
        except ValueError as e:
            raise ProviderError(f"{self.name}: invalid JSON response", retryable=True) from e


def http_error(provider: str, resp: httpx.Response) -> ProviderError:
    detail = ""
    try:
        body = resp.json()
        err = body.get("error", body)
        detail = err.get("message", "") if isinstance(err, dict) else str(err)
    except ValueError:
        detail = resp.text[:200]
    status = resp.status_code
    hint = {401: "invalid API key", 403: "permission denied", 404: "model or endpoint not found",
            429: "rate limited"}.get(status, "")
    msg = f"{provider}: HTTP {status}" + (f" ({hint})" if hint else "") + (
        f": {detail[:300]}" if detail else "")
    err = ProviderError(msg, retryable=status in RETRYABLE_STATUS, status=status)
    ra = resp.headers.get("retry-after")
    if ra:
        try:
            err.retry_after = float(ra)  # type: ignore[attr-defined]
        except ValueError:
            pass
    return err


async def iter_sse(resp: httpx.Response) -> AsyncIterator[tuple[str | None, str]]:
    """Parse a Server-Sent Events stream into (event, data) pairs."""
    event: str | None = None
    data: list[str] = []
    async for line in resp.aiter_lines():
        if line == "":
            if data:
                yield event, "\n".join(data)
            event, data = None, []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        yield event, "\n".join(data)
