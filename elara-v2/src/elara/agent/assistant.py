"""Assistant: the single conversational entry point used by the CLI and the API.

handle(text) -> validate -> language -> pending confirmation? -> intent -> route ->
{deterministic handler | research workflow | LLM tool loop} -> implicit memory ->
persist -> reply.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from elara.agent.intent import Intent, IntentClassifier, IntentResult
from elara.agent.loop import AgentLoop
from elara.agent.prompts import build_system_prompt
from elara.agent.replies import confirmation_prompt, outcome_text, parse_yes_no
from elara.agent.router import Route, Router, Tier
from elara.config.settings import Settings
from elara.conversation.i18n import t
from elara.conversation.store import ConversationStore
from elara.core.context import conversation_id_var, request_scope
from elara.core.errors import DatabaseError, ProviderError
from elara.core.language import detect_language
from elara.core.logging import get_logger
from elara.memory import MemoryService, MemorySource
from elara.providers.base import ChatMessage
from elara.providers.service import LLMService
from elara.research.engine import ResearchEngine
from elara.research.models import Record
from elara.research.synthesis import Synthesizer
from elara.security.input import clean_user_text
from elara.security.untrusted import Trust, wrap_untrusted
from elara.tools.base import Origin, ToolContext
from elara.tools.confirmations import ConfirmationStore, PendingAction
from elara.tools.builtin.calculator import evaluate, format_number, normalize_expression
from elara.tools.executor import Status, ToolExecutor, ToolOutcome

log = get_logger(__name__)
_PATH_IN_TEXT = re.compile(r"(~?/[^\s,;\"'“”]+)")


class ToolCallInfo(BaseModel):
    tool: str
    status: str
    error: str | None = None


class PendingInfo(BaseModel):
    id: str
    description: str
    reason: str
    expires_at: str


class AssistantReply(BaseModel):
    text: str
    conversation_id: str
    request_id: str
    language: str
    intent: str
    tier: int
    used_llm: bool = False
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)
    memory_events: list[str] = Field(default_factory=list)
    pending_action: PendingInfo | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)


class _Turn:
    """Mutable per-turn state."""

    def __init__(self, cid: str, rid: str, lang: str, state: dict):
        self.cid, self.rid, self.lang, self.state = cid, rid, lang, state
        self.intent = Intent.GENERAL
        self.tier = Tier.DETERMINISTIC
        self.used_llm = False
        self.tool_calls: list[ToolCallInfo] = []
        self.memory_events: list[str] = []
        self.pending: PendingAction | None = None
        self.sources: list[dict] = []
        self.user_message_id: int | None = None

    def record(self, outcome: ToolOutcome) -> None:
        self.tool_calls.append(ToolCallInfo(tool=outcome.tool, status=outcome.status.value,
                                            error=outcome.error))


class Assistant:
    def __init__(self, *, settings: Settings, conversations: ConversationStore,
                 memory: MemoryService, llm: LLMService, executor: ToolExecutor,
                 confirmations: ConfirmationStore, research: ResearchEngine,
                 synthesizer: Synthesizer, classifier: IntentClassifier, router: Router,
                 loop: AgentLoop):
        self.settings = settings
        self.conversations = conversations
        self.memory = memory
        self.llm = llm
        self.executor = executor
        self.confirmations = confirmations
        self.research = research
        self.synthesizer = synthesizer
        self.classifier = classifier
        self.router = router
        self.loop = loop

    # ------------------------------------------------------------------ entry point --
    async def handle(self, text: str, conversation_id: str | None = None,
                     request_id: str | None = None) -> AssistantReply:
        with request_scope(request_id, conversation_id) as rid:
            try:
                cid = self.conversations.ensure(conversation_id)
                state = self.conversations.get_state(cid)
            except DatabaseError as e:
                lang = detect_language(text, self.settings.default_language)
                return AssistantReply(text=t("db_error", lang, error=str(e)),
                                      conversation_id=conversation_id or "", request_id=rid,
                                      language=lang, intent="error", tier=0)
            conversation_id_var.set(cid)
            text = clean_user_text(text)
            lang = detect_language(text, state.get("language") or self.settings.default_language)
            turn = _Turn(cid, rid, lang, state)
            if not text:
                return self._reply(turn, t("empty_input", lang), persist=False)
            if len(text) > self.settings.max_message_chars:
                return self._reply(turn, t("input_too_long", lang, n=len(text),
                                             max=self.settings.max_message_chars), persist=False)
            try:
                turn.user_message_id = self.conversations.add_message(cid, "user", text,
                                                                      language=lang)
                reply_text = await self._dispatch(turn, text)
            except DatabaseError as e:
                log.exception("assistant.db_error")
                return self._reply(turn, t("db_error", lang, error=str(e)), persist=False)
            state["language"] = lang
            return self._reply(turn, reply_text)

    async def _dispatch(self, turn: _Turn, text: str) -> str:
        pending = self.confirmations.pending_for(turn.cid)
        if pending:
            answer = parse_yes_no(text)
            if answer is not None:
                return await self._resolve_pending(turn, pending, answer)
            self.confirmations.resolve(pending.id, "cancelled")  # user moved on

        result = self.classifier.classify(text, has_references=bool(turn.state.get("references")))
        route = self.router.route(result.intent, text)
        turn.intent, turn.tier = result.intent, route.tier
        log.info("assistant.route", extra={"intent": result.intent.value, "tier": int(route.tier),
                                           "reason": route.reason, "language": turn.lang})
        if route.tier == Tier.DETERMINISTIC:
            reply = await self._deterministic(turn, result, text)
            if reply is not None:
                return reply
            turn.intent, route = Intent.GENERAL, self.router.route(Intent.GENERAL, text)
            turn.tier = route.tier
        if route.tier == Tier.RESEARCH:
            return await self._research(turn, result.slots["query"])
        if result.intent == Intent.REFERENCE:
            return await self._reference(turn, text, result.slots["index"], route)
        self._implicit_memory(turn, text)
        return await self._llm(turn, text, route)

    # --------------------------------------------------------------- confirmations --
    async def _resolve_pending(self, turn: _Turn, pending: PendingAction, yes: bool) -> str:
        turn.intent = Intent.GENERAL
        if not yes:
            self.confirmations.resolve(pending.id, "cancelled")
            return t("confirm_cancelled", turn.lang)
        if self.confirmations.is_expired(pending):
            self.confirmations.resolve(pending.id, "expired")
            return t("confirm_expired", turn.lang)
        outcome = await self.executor.execute_pending(pending, ToolContext(
            origin=Origin.USER, conversation_id=turn.cid, language=turn.lang,
            origin_message_id=turn.user_message_id))
        turn.record(outcome)
        return outcome_text(outcome, turn.lang)

    async def confirm(self, action_id: str, approve: bool,
                      conversation_id: str | None = None) -> AssistantReply:
        """Confirm/cancel a pending action by id (used by the API)."""
        with request_scope(None, conversation_id) as rid:
            pending = self.confirmations.get(action_id)
            lang = self.settings.default_language
            cid = conversation_id or (pending.conversation_id if pending else None) or ""
            turn = _Turn(cid, rid, lang, self.conversations.get_state(cid) if cid else {})
            if not pending or pending.status != "pending":
                return self._reply(turn, t("confirm_expired", lang), persist=False)
            text = await self._resolve_pending(turn, pending, approve)
            return self._reply(turn, text, persist=bool(cid))

    # --------------------------------------------------------------- deterministic --
    async def _deterministic(self, turn: _Turn, r: IntentResult, text: str) -> str | None:
        lang = turn.lang
        match r.intent:
            case Intent.GREETING:
                name = self.memory.user_name()
                reply = t("greeting", lang)
                return reply.replace("!", f", {name}!", 1) if name else reply
            case Intent.HOW_ARE_YOU:
                return t("how_are_you", lang)
            case Intent.THANKS:
                return t("thanks", lang)
            case Intent.GOODBYE:
                return t("goodbye", lang)
            case Intent.CALCULATION:
                expr = r.slots["expression"]
                return t("calc_result", lang, expr=normalize_expression(expr).replace("**", "^"),
                         value=format_number(evaluate(expr)))
            case Intent.TIME:
                now = datetime.now().astimezone()
                return t("time_now", lang, time=now.strftime("%H:%M"),
                         date=now.strftime("%Y-%m-%d"))
            case Intent.MEMORY_STORE:
                return self._remember(turn, r.slots["text"], MemorySource.USER_EXPLICIT)
            case Intent.MEMORY_FORGET:
                return self._forget(turn, r.slots["text"])
            case Intent.MEMORY_LIST:
                items = self.memory.store.list(limit=50)
                if not items:
                    return t("memory_empty", lang)
                return t("memory_list_header", lang) + "\n" + "\n".join(
                    f"  #{m.id} [{m.kind}] {m.content}" for m in items)
            case Intent.MEMORY_QUERY:
                hits = self.memory.relevant(r.slots["query"], limit=3)
                if hits:
                    return t("memory_found", lang, content=hits[0].content)
                if self.llm.available:
                    return None  # let the model answer (it may be general knowledge)
                return t("memory_not_found", lang)
            case Intent.FILE_LIST:
                return await self._run_tool(turn, "list_files", {"path": r.slots["path"]})
            case Intent.OPEN_PATH:
                target = self._resolve_named_path(r.slots["target"])
                if target is None:
                    if self.llm.available:
                        return None
                    return t("open_unknown", lang, target=r.slots["target"])
                return await self._run_tool(turn, "open_path", {"path": target})
        return None

    async def _run_tool(self, turn: _Turn, name: str, args: dict) -> str:
        outcome = await self.executor.execute(name, args, ToolContext(
            origin=Origin.USER, conversation_id=turn.cid, language=turn.lang,
            origin_message_id=turn.user_message_id))
        turn.record(outcome)
        if outcome.status == Status.NEEDS_CONFIRMATION and outcome.pending:
            turn.pending = outcome.pending
            return confirmation_prompt(outcome.pending, turn.lang)
        return outcome_text(outcome, turn.lang)

    def _resolve_named_path(self, target: str) -> str | None:
        key = target.lower().strip()
        for suffix in (" folder", " directory", " qovluğu", " qovluğum", " klasörü", " klasörüm"):
            key = key.removesuffix(suffix)
        if key in self.settings.named_paths:
            return str(self.settings.named_paths[key])
        if target.startswith(("/", "~", ".")):
            return target
        for k in (f"{key} folder", f"{key} directory", key):
            m = self.memory.store.by_key(k)
            if m and (p := _PATH_IN_TEXT.search(m.content)):
                return p.group(1)
        for m in self.memory.relevant(f"{key} folder", limit=3):
            if (p := _PATH_IN_TEXT.search(m.content)) and key.split()[0] in m.content.lower():
                return p.group(1)
        return None

    # --------------------------------------------------------------------- memory ----
    def _remember(self, turn: _Turn, content: str, source: MemorySource) -> str:
        out = self.memory.remember(content, source=source, trust=Trust.USER,
                                   conversation_id=turn.cid,
                                   origin_message_id=turn.user_message_id)
        if not out.saved or out.result is None:
            return t("memory_rejected", turn.lang, reason=out.reason)
        res = out.result
        turn.memory_events.append(f"{res.action}:#{res.memory.id}")
        if res.action == "duplicate":
            return t("memory_duplicate", turn.lang, content=res.memory.content)
        if res.action == "updated" and res.previous:
            return t("memory_updated", turn.lang, content=res.memory.content,
                     old=res.previous.content)
        return t("memory_saved", turn.lang, content=res.memory.content)

    def _forget(self, turn: _Turn, what: str) -> str:
        out = self.memory.forget(what, conversation_id=turn.cid)
        if out.deleted:
            turn.memory_events.extend(f"deleted:#{m.id}" for m in out.deleted)
            return t("memory_deleted", turn.lang, content="; ".join(m.content
                                                                     for m in out.deleted))
        if out.candidates:
            items = "\n".join(f"  #{m.id} {m.content}" for m in out.candidates)
            return t("memory_ambiguous", turn.lang, items=items)
        return t("memory_not_found", turn.lang)

    def _implicit_memory(self, turn: _Turn, text: str) -> None:
        cand = self.memory.policy.evaluate_statement(text)
        if cand is None:
            return
        out = self.memory.remember(text, source=MemorySource.USER_STATEMENT, trust=Trust.USER,
                                   conversation_id=turn.cid,
                                   origin_message_id=turn.user_message_id)
        if out.saved and out.result and out.result.action != "duplicate":
            turn.memory_events.append(f"{out.result.action}:#{out.result.memory.id}")

    # ------------------------------------------------------------------- research ----
    async def _research(self, turn: _Turn, query: str) -> str:
        result = await self.research.research(query, language=turn.lang)
        synthesis = await self.synthesizer.synthesize(query, result, turn.lang)
        turn.used_llm = synthesis.used_llm
        turn.sources = [_ref(r) for r in result.records]
        if result.records:
            turn.state["references"] = turn.sources
            turn.state["focus"] = 1
        return synthesis.text

    async def _reference(self, turn: _Turn, text: str, index: int, route: Route) -> str:
        refs: list[dict] = turn.state.get("references") or []
        if index == 0:
            index = turn.state.get("focus", 1)
        if index == -1:
            index = len(refs)
        if not (1 <= index <= len(refs)):
            return t("reference_unresolved", turn.lang)
        ref = refs[index - 1]
        turn.state["focus"] = index
        detail = (f"[{index}] {ref['citation']}\nAbstract/summary: "
                  f"{ref.get('abstract') or 'not available'}")
        if not self.llm.available:
            return detail
        wrapped = wrap_untrusted(detail, f"earlier result #{index}")
        prompt = (f"{text}\n\n(The user is referring to earlier result #{index}; its stored "
                  f"metadata follows.)\n{wrapped.text}")
        return await self._llm(turn, text, route, override_user_text=prompt, tainted=True)

    # ------------------------------------------------------------------------ LLM ----
    async def _llm(self, turn: _Turn, text: str, route: Route, *,
                   override_user_text: str | None = None, tainted: bool = False) -> str:
        if not self.llm.available:
            return t("llm_unavailable", turn.lang, error="no provider configured")
        system = build_system_prompt(
            language=turn.lang, profile=self.memory.profile(),
            relevant=self.memory.relevant(text, limit=5), user_name=self.memory.user_name())
        history = self._history(turn)
        try:
            result = await self.loop.run(system=system, history=history,
                                         user_text=override_user_text or text,
                                         language=turn.lang, conversation_id=turn.cid,
                                         model_tier=route.model_tier, tainted=tainted)
        except ProviderError as e:
            return t("llm_unavailable", turn.lang, error=str(e))
        turn.used_llm = True
        for o in result.outcomes:
            turn.record(o)
        if result.pending:
            turn.pending = result.pending
            prefix = (result.text + "\n\n") if result.text else ""
            return prefix + confirmation_prompt(result.pending, turn.lang)
        return result.text or "…"

    def _history(self, turn: _Turn) -> list[ChatMessage]:
        msgs = self.conversations.recent_messages(turn.cid,
                                                  limit=self.settings.context_max_messages + 1)
        msgs = [m for m in msgs if m.id != turn.user_message_id]
        while msgs and msgs[0].role != "user":
            msgs.pop(0)
        return [ChatMessage(role=m.role, text=m.content) for m in msgs]  # type: ignore[arg-type]

    # ---------------------------------------------------------------------- reply ----
    def _reply(self, turn: _Turn, text: str, persist: bool = True) -> AssistantReply:
        if persist and turn.cid:
            self.conversations.add_message(
                turn.cid, "assistant", text, language=turn.lang, intent=turn.intent.value,
                tier=int(turn.tier), metadata={"tools": [c.model_dump() for c in turn.tool_calls],
                                               "memory": turn.memory_events})
            self.conversations.set_state(turn.cid, turn.state)
        pending = None
        if turn.pending:
            pending = PendingInfo(id=turn.pending.id, description=turn.pending.description,
                                  reason=turn.pending.reason, expires_at=turn.pending.expires_at)
        return AssistantReply(text=text, conversation_id=turn.cid, request_id=turn.rid,
                              language=turn.lang, intent=turn.intent.value, tier=int(turn.tier),
                              used_llm=turn.used_llm, tool_calls=turn.tool_calls,
                              memory_events=turn.memory_events, pending_action=pending,
                              sources=turn.sources)


def _ref(r: Record) -> dict[str, Any]:
    return {"title": r.title, "source": r.source, "year": r.year, "doi": r.doi, "pmid": r.pmid,
            "url": r.best_url(), "evidence_type": r.evidence_type.value,
            "citation": r.citation(), "abstract": (r.abstract or "")[:1500] or None}
