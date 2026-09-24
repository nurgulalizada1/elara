"""Tool contract. Each tool is a small class with typed input/output models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel

from elara.providers.base import ToolSpec
from elara.security.untrusted import Trust


class Permission(StrEnum):
    READ_ONLY = "read_only"
    LOW_RISK_WRITE = "low_risk_write"
    HIGH_RISK_WRITE = "high_risk_write"
    SYSTEM = "system"


class Origin(StrEnum):
    USER = "user"  # deterministic handling of a user's explicit command
    LLM = "llm"    # chosen by a model
    API = "api"    # direct call through the HTTP API / CLI by the user


@dataclass
class ToolContext:
    origin: Origin
    conversation_id: str | None = None
    tainted: bool = False     # untrusted content has entered this turn's context
    confirmed: bool = False   # the user explicitly confirmed this exact action
    language: str = "az"
    origin_message_id: int | None = None


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    Input: ClassVar[type[BaseModel]]
    Output: ClassVar[type[BaseModel]]
    permission: ClassVar[Permission] = Permission.READ_ONLY
    timeout_s: ClassVar[float] = 15.0
    output_trust: ClassVar[Trust] = Trust.SYSTEM
    category: ClassVar[str] = "general"
    exposed_to_llm: ClassVar[bool] = True
    safety_notes: ClassVar[str] = ""

    def permission_for(self, args: BaseModel) -> Permission:
        """Override when risk depends on arguments (e.g. overwrite vs create)."""
        return self.permission

    def describe_action(self, args: BaseModel) -> str:
        shown = ", ".join(f"{k}={_short(v)}" for k, v in args.model_dump().items())
        return f"{self.name}({shown})"

    def source_label(self, args: BaseModel) -> str:
        return f"tool:{self.name}"

    @abstractmethod
    async def run(self, args: Any, ctx: ToolContext) -> BaseModel: ...

    def render(self, output: BaseModel) -> str:
        """Text given to the model / user. Default: compact JSON."""
        return output.model_dump_json(exclude_none=True)

    def spec(self) -> ToolSpec:
        schema = self.Input.model_json_schema()
        schema.pop("title", None)
        return ToolSpec(name=self.name, description=self.description, input_schema=schema)

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name, "description": self.description, "category": self.category,
            "permission": self.permission.value, "timeout_s": self.timeout_s,
            "output_trust": self.output_trust.value, "exposed_to_llm": self.exposed_to_llm,
            "input_schema": self.spec().input_schema,
            "output_schema": self.Output.model_json_schema(),
            "safety_notes": self.safety_notes,
        }


def _short(v: Any, n: int = 80) -> str:
    s = repr(v)
    return s if len(s) <= n else s[: n - 3] + "..."
