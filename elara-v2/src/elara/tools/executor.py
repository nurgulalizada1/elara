"""ToolExecutor: the only path by which any tool runs.

Pipeline: lookup -> validate input -> permission decision -> (confirmation) ->
run with timeout -> validate output -> wrap untrusted output -> audit.
It never reports success for something that did not succeed.
"""

from __future__ import annotations

import asyncio
import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError

from elara.core.errors import ElaraError, PermissionDenied
from elara.core.logging import get_logger
from elara.database.audit import AuditLog
from elara.security.injection import InjectionDetector
from elara.security.untrusted import Trust, wrap_untrusted
from elara.tools.base import Tool, ToolContext
from elara.tools.confirmations import ConfirmationStore, PendingAction
from elara.tools.permissions import Decision, PermissionPolicy
from elara.tools.registry import ToolRegistry

log = get_logger(__name__)


class Status(StrEnum):
    OK = "ok"
    ERROR = "error"
    INVALID = "invalid"
    DENIED = "denied"
    NEEDS_CONFIRMATION = "needs_confirmation"


class ToolOutcome(BaseModel):
    tool: str
    status: Status
    output: dict[str, Any] | None = None
    text: str = ""                       # rendering for model/user (enveloped if untrusted)
    error: str | None = None
    trust: Trust = Trust.SYSTEM
    pending: PendingAction | None = None
    injection_categories: list[str] = Field(default_factory=list)
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.status == Status.OK

    def for_model(self) -> tuple[str, bool]:
        """(content, is_error) to hand back to an LLM as a tool result."""
        if self.status == Status.OK:
            return self.text, False
        if self.status == Status.NEEDS_CONFIRMATION:
            return ("NOT EXECUTED: this action requires the user's explicit confirmation. "
                    "The user has been asked; do not claim it was done."), True
        return f"FAILED ({self.status.value}): {self.error}", True


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, policy: PermissionPolicy,
                 confirmations: ConfirmationStore, audit: AuditLog | None = None,
                 detector: InjectionDetector | None = None):
        self.registry = registry
        self.policy = policy
        self.confirmations = confirmations
        self.audit = audit
        self.detector = detector or InjectionDetector()

    async def execute(self, name: str, raw_args: dict[str, Any] | None,
                      ctx: ToolContext) -> ToolOutcome:
        raw_args = raw_args or {}
        tool = self.registry.get(name)
        if tool is None:
            return self._finish(None, name, raw_args, ctx, "-", ToolOutcome(
                tool=name, status=Status.INVALID, error=f"unknown tool '{name}'"))
        try:
            args = tool.Input.model_validate(raw_args)
        except PydanticValidationError as e:
            msg = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}"
                            for err in e.errors()[:5])
            return self._finish(tool, name, raw_args, ctx, tool.permission.value, ToolOutcome(
                tool=name, status=Status.INVALID, error=f"invalid arguments: {msg}"))

        perm = tool.permission_for(args)
        decision = self.policy.decide(tool, perm, ctx)
        if decision.decision == Decision.DENY:
            return self._finish(tool, name, raw_args, ctx, perm.value, ToolOutcome(
                tool=name, status=Status.DENIED, error=decision.reason))
        if decision.decision == Decision.CONFIRM:
            pending = self.confirmations.create(
                ctx.conversation_id, name, args.model_dump(mode="json"), decision.reason,
                tool.describe_action(args))
            return self._finish(tool, name, raw_args, ctx, perm.value, ToolOutcome(
                tool=name, status=Status.NEEDS_CONFIRMATION, pending=pending,
                error=decision.reason))

        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(tool.run(args, ctx), timeout=tool.timeout_s)
            if not isinstance(result, tool.Output):
                result = tool.Output.model_validate(result)
        except TimeoutError:
            outcome = ToolOutcome(tool=name, status=Status.ERROR,
                                  error=f"timed out after {tool.timeout_s:.0f}s")
        except PermissionDenied as e:
            outcome = ToolOutcome(tool=name, status=Status.DENIED, error=str(e))
        except (ElaraError, OSError, ValueError, PydanticValidationError) as e:
            outcome = ToolOutcome(tool=name, status=Status.ERROR, error=str(e) or type(e).__name__)
        except Exception as e:  # unexpected bug: report it honestly, keep ELARA alive
            log.exception("tool.crash", extra={"tool": name})
            outcome = ToolOutcome(tool=name, status=Status.ERROR,
                                  error=f"internal error: {type(e).__name__}")
        else:
            outcome = self._success(tool, args, result)
        outcome.duration_ms = int((time.perf_counter() - t0) * 1000)
        return self._finish(tool, name, raw_args, ctx, perm.value, outcome)

    async def execute_pending(self, pending: PendingAction, ctx: ToolContext) -> ToolOutcome:
        """Run a previously confirmed action exactly as it was described to the user."""
        if not self.confirmations.resolve(pending.id, "confirmed"):
            return ToolOutcome(tool=pending.tool_name, status=Status.DENIED,
                               error="this action was already resolved")
        ctx.confirmed = True
        return await self.execute(pending.tool_name, pending.arguments, ctx)

    def _success(self, tool: Tool, args: BaseModel, result: BaseModel) -> ToolOutcome:
        rendered = tool.render(result)
        outcome = ToolOutcome(tool=tool.name, status=Status.OK,
                              output=result.model_dump(mode="json"), trust=tool.output_trust)
        if tool.output_trust == Trust.UNTRUSTED:
            wrapped = wrap_untrusted(rendered, tool.source_label(args), detector=self.detector)
            outcome.text = wrapped.text
            if wrapped.report.suspicious:
                outcome.injection_categories = sorted(set(wrapped.report.categories))
                if self.audit:
                    self.audit.security_event(
                        "prompt_injection_suspected", wrapped.report.severity,
                        f"{tool.source_label(args)}: {wrapped.report.summary()} :: "
                        f"{'; '.join(wrapped.report.evidence[:3])}")
        else:
            outcome.text = rendered
        return outcome

    def _finish(self, tool: Tool | None, name: str, raw_args: dict, ctx: ToolContext,
                permission: str, outcome: ToolOutcome) -> ToolOutcome:
        log.info("tool.call", extra={
            "tool": name, "origin": ctx.origin.value, "status": outcome.status.value,
            "permission": permission, "tainted": ctx.tainted, "confirmed": ctx.confirmed,
            "duration_ms": outcome.duration_ms, "error": outcome.error})
        if self.audit:
            try:
                self.audit.tool_call(
                    tool_name=name, arguments=raw_args, origin=ctx.origin.value,
                    permission=permission, status=outcome.status.value, confirmed=ctx.confirmed,
                    result_summary=outcome.text[:300] if outcome.ok else None,
                    error=outcome.error, duration_ms=outcome.duration_ms)
            except Exception:
                log.exception("audit.tool_call_failed")
        return outcome
