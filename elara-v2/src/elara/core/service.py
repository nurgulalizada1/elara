"""ElaraCore: the application-independent entry point for every frontend.

    CLI ─────┐
    API ─────┤
    Voice ───┤──→ ElaraCore.process(CoreRequest) -> CoreResult
    Desktop ─┤
    Mobile ──┘

It is a thin facade over the existing composition root (`build_container`) and
`Assistant`; routing, memory, tools, research and security are unchanged. Frontends
should depend only on this module (plus `container` for management views such as
memory listing), never on `Assistant` internals.

    async with ElaraCore.open() as core:
        result = await core.process(CoreRequest(text="What's 17 * 42?", channel="cli"))
        print(result.text, result.resolver, result.used_llm)
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from elara.agent.assistant import AssistantReply
from elara.config.settings import Settings, get_settings
from elara.core.buildinfo import build_info
from elara.core.container import Container, build_container
from elara.core.logging import get_logger

log = get_logger(__name__)


class CoreRequest(BaseModel):
    """A normalized user request, independent of where it came from."""

    text: str
    conversation_id: str | None = None
    request_id: str | None = None
    channel: str = Field(default="unknown", max_length=32,
                         description="Originating frontend: cli, api, voice, desktop, mobile…")


class CoreResult(AssistantReply):
    """AssistantReply plus execution/build metadata.

    Inherited fields: text, conversation_id, request_id, language, intent, tier, used_llm,
    tool_calls, memory_events, pending_action, sources, resolver.
    """

    channel: str = "unknown"
    duration_ms: int = 0
    build: dict[str, Any] = Field(default_factory=dict)


class ElaraCore:
    def __init__(self, container: Container, *, owns_container: bool = True):
        self.container = container
        self._owns = owns_container

    @classmethod
    def open(cls, settings: Settings | None = None, **container_kwargs: Any) -> ElaraCore:
        """Build all services (DB migrations, providers, tools, research) for this process."""
        return cls(build_container(settings or get_settings(), **container_kwargs))

    @property
    def settings(self) -> Settings:
        return self.container.settings

    async def process(self, request: CoreRequest) -> CoreResult:
        t0 = time.perf_counter()
        reply = await self.container.assistant.handle(request.text, request.conversation_id,
                                                      request.request_id)
        return self._result(reply, request.channel, t0)

    async def confirm(self, action_id: str, approve: bool, *, conversation_id: str | None = None,
                      channel: str = "unknown") -> CoreResult:
        """Approve or reject a pending action (the out-of-band alternative to 'yes'/'no')."""
        t0 = time.perf_counter()
        reply = await self.container.assistant.confirm(action_id, approve, conversation_id)
        return self._result(reply, channel, t0)

    @staticmethod
    def _result(reply: AssistantReply, channel: str, t0: float) -> CoreResult:
        result = CoreResult(**reply.model_dump(), channel=channel,
                            duration_ms=int((time.perf_counter() - t0) * 1000),
                            build=build_info())
        log.info("core.processed", extra={"channel": channel, "intent": result.intent,
                                          "resolver": result.resolver, "tier": result.tier,
                                          "used_llm": result.used_llm,
                                          "duration_ms": result.duration_ms})
        return result

    async def close(self) -> None:
        if self._owns:
            await self.container.aclose()

    async def __aenter__(self) -> ElaraCore:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
