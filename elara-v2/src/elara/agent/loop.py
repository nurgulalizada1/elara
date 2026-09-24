"""The LLM tool-use loop (Tiers 1 and 2).

The model proposes tool calls; every call goes through ToolExecutor (validation,
permissions, confirmation, timeouts, audit). Untrusted tool output taints the rest
of the turn, which escalates any write to "needs confirmation".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from elara.conversation.i18n import t
from elara.core.logging import get_logger
from elara.providers.base import ChatMessage, CompletionRequest, ToolResult
from elara.providers.service import LLMService
from elara.security.untrusted import Trust
from elara.tools.base import Origin, ToolContext
from elara.tools.confirmations import PendingAction
from elara.tools.executor import Status, ToolExecutor, ToolOutcome
from elara.tools.registry import ToolRegistry

log = get_logger(__name__)


@dataclass
class LoopResult:
    text: str
    outcomes: list[ToolOutcome] = field(default_factory=list)
    pending: PendingAction | None = None
    steps: int = 0
    tainted: bool = False


class AgentLoop:
    def __init__(self, llm: LLMService, registry: ToolRegistry, executor: ToolExecutor,
                 max_steps: int = 6, max_tokens: int = 4096):
        self.llm = llm
        self.registry = registry
        self.executor = executor
        self.max_steps = max_steps
        self.max_tokens = max_tokens

    async def run(self, *, system: str, history: list[ChatMessage], user_text: str,
                  language: str, conversation_id: str, model_tier: str,
                  tainted: bool = False) -> LoopResult:
        messages = [*history, ChatMessage(role="user", text=user_text)]
        tools = self.registry.llm_specs()
        result = LoopResult(text="", tainted=tainted)
        provider_name = self.llm.provider.name if self.llm.provider else "none"
        for step in range(1, self.max_steps + 1):
            result.steps = step
            resp = await self.llm.complete(
                CompletionRequest(system=system, messages=messages, tools=tools,
                                  max_tokens=self.max_tokens),
                tier=model_tier, purpose="chat")  # type: ignore[arg-type]
            if resp.stop_reason == "refusal" and not resp.tool_calls:
                result.text = resp.text.strip() or t("refusal", language)
                return result
            if not resp.tool_calls:
                result.text = resp.text.strip()
                if resp.stop_reason in ("max_tokens", "length"):
                    result.text += " …"
                return result
            messages.append(resp.as_message(provider_name))
            tool_results: list[ToolResult] = []
            for call in resp.tool_calls:
                ctx = ToolContext(origin=Origin.LLM, conversation_id=conversation_id,
                                  tainted=result.tainted, language=language)
                outcome = await self.executor.execute(call.name, call.arguments, ctx)
                result.outcomes.append(outcome)
                if outcome.status == Status.NEEDS_CONFIRMATION and outcome.pending:
                    result.pending = outcome.pending
                if outcome.ok and outcome.trust == Trust.UNTRUSTED:
                    result.tainted = True
                content, is_error = outcome.for_model()
                tool_results.append(ToolResult(tool_call_id=call.id, content=content,
                                               is_error=is_error))
            messages.append(ChatMessage(role="user", tool_results=tool_results))
            if result.pending:
                # Stop here: the user must decide before anything else happens.
                result.text = resp.text.strip()
                return result
        log.warning("agent.step_limit", extra={"steps": self.max_steps})
        result.text = t("step_limit", language)
        return result
