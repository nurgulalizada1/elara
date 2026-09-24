"""Permission policy: decides ALLOW / CONFIRM / DENY for each tool invocation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from elara.tools.base import Origin, Permission, Tool, ToolContext


class Decision(StrEnum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass
class PolicyResult:
    decision: Decision
    reason: str


class PermissionPolicy:
    def __init__(self, disabled_tools: list[str] | None = None,
                 enabled_flags: dict[str, bool] | None = None):
        self.disabled = set(disabled_tools or [])
        self.enabled_flags = enabled_flags or {}

    def decide(self, tool: Tool, perm: Permission, ctx: ToolContext) -> PolicyResult:
        if tool.name in self.disabled or self.enabled_flags.get(tool.name) is False:
            return PolicyResult(Decision.DENY, f"tool '{tool.name}' is disabled in configuration")
        if ctx.origin == Origin.LLM and not tool.exposed_to_llm:
            return PolicyResult(Decision.DENY, f"tool '{tool.name}' cannot be invoked by the model")
        if perm == Permission.READ_ONLY:
            return PolicyResult(Decision.ALLOW, "read-only")
        if ctx.confirmed:
            return PolicyResult(Decision.ALLOW, "confirmed by user")
        if perm == Permission.LOW_RISK_WRITE:
            if ctx.origin == Origin.LLM and ctx.tainted:
                return PolicyResult(Decision.CONFIRM,
                                    "the request follows untrusted external content")
            return PolicyResult(Decision.ALLOW, "low-risk write")
        if perm == Permission.HIGH_RISK_WRITE:
            return PolicyResult(Decision.CONFIRM, "this changes or deletes existing data")
        return PolicyResult(Decision.CONFIRM, "this runs a program on your computer")
