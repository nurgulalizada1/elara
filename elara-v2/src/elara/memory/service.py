"""MemoryService: the API the rest of ELARA uses for memory."""

from __future__ import annotations

from dataclasses import dataclass, field

from elara.core.logging import get_logger
from elara.database.audit import AuditLog
from elara.memory.models import Memory, MemoryKind, MemorySource, SaveResult
from elara.memory.policy import MemoryPolicy
from elara.memory.store import MemoryStore
from elara.security.untrusted import Trust

log = get_logger(__name__)
_DEICTIC = {"this", "that", "it", "bunu", "onu", "şunu", "bunları", "that one", "sonuncunu",
            "sonuncuyu", "the last one"}


@dataclass
class RememberOutcome:
    saved: bool
    result: SaveResult | None = None
    reason: str = ""


@dataclass
class ForgetOutcome:
    deleted: list[Memory] = field(default_factory=list)
    candidates: list[Memory] = field(default_factory=list)


class MemoryService:
    def __init__(self, store: MemoryStore, policy: MemoryPolicy, audit: AuditLog | None = None):
        self.store = store
        self.policy = policy
        self.audit = audit

    def remember(self, content: str, *, source: MemorySource, trust: Trust,
                 conversation_id: str | None = None, origin_message_id: int | None = None,
                 kind: MemoryKind | None = None, key: str | None = None) -> RememberOutcome:
        cand = self.policy.classify(content, source)
        if kind:
            cand.kind = kind
        if key:
            cand.key = key
        ok, reason = self.policy.validate(cand, trust)
        if not ok:
            log.info("memory.rejected", extra={"reason": reason, "source": source.value})
            if self.audit and trust != Trust.USER:
                self.audit.security_event("memory_write_blocked", "medium",
                                          f"{reason}: {content[:200]}")
            return RememberOutcome(saved=False, reason=reason)
        result = self.store.add(cand, conversation_id=conversation_id,
                                origin_message_id=origin_message_id)
        log.info("memory.saved", extra={"memory_id": result.memory.id, "action": result.action,
                                        "kind": cand.kind.value, "source": source.value})
        return RememberOutcome(saved=True, result=result)

    def forget(self, what: str, conversation_id: str | None = None) -> ForgetOutcome:
        what_l = what.strip().lower()
        if what_l.isdigit():
            m = self.store.get(int(what_l))
            if m and self.store.delete(m.id):
                return ForgetOutcome(deleted=[m])
            return ForgetOutcome()
        if what_l in _DEICTIC:
            recent = [m for m in self.store.list(limit=20)
                      if conversation_id is None or m.conversation_id == conversation_id]
            if recent:
                self.store.delete(recent[0].id)
                return ForgetOutcome(deleted=[recent[0]])
            return ForgetOutcome()
        matches = self.store.search(what, limit=5)
        if not matches:
            return ForgetOutcome()
        if len(matches) == 1:
            self.store.delete(matches[0].id)
            return ForgetOutcome(deleted=matches)
        return ForgetOutcome(candidates=matches)

    def relevant(self, query: str, limit: int = 5) -> list[Memory]:
        return self.store.search(query, limit=limit)

    def profile(self, limit: int = 20) -> list[Memory]:
        """Always-in-context memories: preferences and keyed facts (name etc.)."""
        prefs = self.store.list(MemoryKind.PREFERENCE, limit=limit)
        facts = [m for m in self.store.list(MemoryKind.FACT, limit=50) if m.key][:limit]
        episodic = self.store.list(MemoryKind.EPISODIC, limit=5)
        return prefs + facts + episodic

    def user_name(self) -> str | None:
        m = self.store.by_key("name")
        if not m:
            return None
        return self.policy.extract_name(m.content) or m.content
