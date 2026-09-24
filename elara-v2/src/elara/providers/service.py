"""LLMService: the single entry point the rest of ELARA uses to talk to models.

Responsibilities: map a routing tier to a concrete model, record latency/token usage,
give a clear ProviderUnavailable when nothing is configured, and offer structured
(JSON-schema validated) output on top of any provider.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from typing import Literal, TypeVar

import httpx
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from elara.config.settings import Settings
from elara.core.errors import ProviderError, ProviderUnavailable
from elara.core.logging import get_logger
from elara.database.audit import AuditLog
from elara.providers.anthropic import AnthropicProvider
from elara.providers.base import ChatMessage, CompletionRequest, CompletionResponse, LLMProvider
from elara.providers.openai import OpenAIProvider

log = get_logger(__name__)
Tier = Literal["fast", "strong"]
M = TypeVar("M", bound=BaseModel)


def build_provider(settings: Settings, client: httpx.AsyncClient) -> LLMProvider | None:
    name = settings.resolved_provider()
    common = {"timeout_s": settings.llm_timeout_s, "max_retries": settings.llm_max_retries}
    if name == "anthropic":
        if not settings.anthropic_api_key:
            return None
        return AnthropicProvider(client, api_key=settings.anthropic_api_key.get_secret_value(),
                                 base_url=settings.anthropic_base_url, **common)
    if name == "openai":
        if not settings.openai_api_key:
            return None
        return OpenAIProvider(client, api_key=settings.openai_api_key.get_secret_value(),
                              base_url=settings.openai_base_url, **common)
    return None


class LLMService:
    def __init__(self, provider: LLMProvider | None, settings: Settings,
                 audit: AuditLog | None = None):
        self.provider = provider
        self.settings = settings
        self.audit = audit

    @property
    def available(self) -> bool:
        return self.provider is not None

    def _require(self) -> LLMProvider:
        if self.provider is None:
            raise ProviderUnavailable(
                "No LLM provider configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY "
                "(and optionally ELARA_PROVIDER).")
        return self.provider

    def _prepare(self, req: CompletionRequest, tier: Tier) -> CompletionRequest:
        model = req.model or self.settings.model_for(tier)
        if not model:
            raise ProviderUnavailable(f"No model configured for tier '{tier}'.")
        return req.model_copy(update={
            "model": model,
            "temperature": req.temperature if req.temperature is not None
            else self.settings.temperature,
        })

    async def complete(self, req: CompletionRequest, *, tier: Tier = "fast",
                       purpose: str = "chat") -> CompletionResponse:
        provider = self._require()
        req = self._prepare(req, tier)
        t0 = time.perf_counter()
        try:
            resp = await provider.complete(req)
        except ProviderError as e:
            self._record(provider, req.model, tier, purpose, None, t0, "error", str(e))
            raise
        self._record(provider, req.model, tier, purpose, resp, t0, "ok")
        return resp

    async def stream(self, req: CompletionRequest, *, tier: Tier = "fast",
                     purpose: str = "chat") -> AsyncIterator[str]:
        provider = self._require()
        req = self._prepare(req, tier)
        t0 = time.perf_counter()
        status, err = "ok", None
        try:
            async for chunk in provider.stream(req):
                yield chunk
        except ProviderError as e:
            status, err = "error", str(e)
            raise
        finally:
            self._record(provider, req.model, tier, purpose, None, t0, status, err)

    async def structured(self, req: CompletionRequest, schema: type[M], *, tier: Tier = "fast",
                         purpose: str = "structured") -> M:
        """Ask for JSON matching ``schema``; validate; one repair round on failure."""
        instruction = (
            "Respond with ONLY a JSON object (no prose, no code fences) that validates against "
            f"this JSON schema:\n{json.dumps(schema.model_json_schema(), ensure_ascii=False)}")
        req = req.model_copy(update={"system": (req.system + "\n\n" + instruction).strip(),
                                     "tools": []})
        resp = await self.complete(req, tier=tier, purpose=purpose)
        try:
            return schema.model_validate(extract_json(resp.text))
        except (ValueError, PydanticValidationError) as first_error:
            repair = req.model_copy(update={"messages": [
                *req.messages,
                ChatMessage(role="assistant", text=resp.text),
                ChatMessage(role="user", text=f"That was not valid: {str(first_error)[:500]}. "
                                              "Reply with the corrected JSON object only."),
            ]})
            resp2 = await self.complete(repair, tier=tier, purpose=purpose + ":repair")
            try:
                return schema.model_validate(extract_json(resp2.text))
            except (ValueError, PydanticValidationError) as e:
                raise ProviderError(f"model did not return valid structured output: {e}") from e

    def _record(self, provider: LLMProvider, model: str, tier: str, purpose: str,
                resp: CompletionResponse | None, t0: float, status: str,
                error: str | None = None) -> None:
        latency = int((time.perf_counter() - t0) * 1000)
        usage = resp.usage if resp else None
        log.info("model.call", extra={
            "provider": provider.name, "model": model, "tier": tier, "purpose": purpose,
            "latency_ms": latency, "status": status,
            "input_tokens": usage.input_tokens if usage else None,
            "output_tokens": usage.output_tokens if usage else None})
        if self.audit:
            try:
                self.audit.model_call(provider=provider.name, model=model, tier=tier,
                                      purpose=purpose,
                                      input_tokens=usage.input_tokens if usage else None,
                                      output_tokens=usage.output_tokens if usage else None,
                                      latency_ms=latency, status=status, error=error)
            except Exception:  # auditing must never break a user request
                log.exception("audit.model_call_failed")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict:
    """Extract the first JSON object from model text (tolerates code fences / prose)."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in response")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("JSON is not an object")
    return obj
