"""Memory decision mechanism: what (if anything) from a user message becomes memory.

Deterministic, multilingual (az/en/tr). Three outcomes:
- explicit command ("remember that ...", "yadda saxla ...", "hatırla ...") -> save
- self-disclosure (name, preferences, dated personal events) -> save with lower confidence
- everything else (questions, chit-chat, requests) -> not saved

Validation refuses: non-user origins, suspected injections, secrets, questions, junk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum

from elara.core.timeutil import iso, utcnow
from elara.memory.models import MemoryCandidate, MemoryKind, MemorySource
from elara.security.injection import InjectionDetector
from elara.security.untrusted import Trust

F = re.I | re.S


class CommandType(StrEnum):
    STORE = "store"
    FORGET = "forget"
    LIST = "list"
    QUERY = "query"


@dataclass
class MemoryCommand:
    type: CommandType
    text: str = ""  # content to store / query to search / what to forget


_STORE = [
    re.compile(r"^(?:please\s+)?(?:save|store|add)(?:\s+(?:this|that))?\s+(?:to|in|into)\s+"
               r"(?:your\s+|long[- ]term\s+)?memory\s*[:,]?\s+(?P<c>.+)$", F),
    re.compile(r"^(?:please\s+)?(?:save|store) (?:this|that)\s*:\s*(?P<c>.+)$", F),
    re.compile(r"^(?:please\s+|pls\s+)?(?:remember|don'?t forget|do not forget|keep in mind|"
               r"note)(?:\s+that)?\s*[:,]?\s+(?P<c>.+)$", F),
    re.compile(r"^(?:xahiş edirəm\s+)?(?:yadda saxla|yadında saxla|unutma|qeyd et)"
               r"(?:\s+ki)?\s*[:,]?\s+(?P<c>.+)$", F),
    re.compile(r"^(?:lütfen\s+)?(?:şunu\s+)?(?:hatırla|unutma|aklında tut|not et)"
               r"(?:\s+ki)?\s*[:,]?\s+(?P<c>.+)$", F),
    re.compile(r"^(?P<c>.+?)[\s,.;:—-]*(?:bunu\s+|şunu\s+)?(?:yadda saxla|yadında saxla|"
               r"unutma|hatırla|aklında tut)[.!\s]*$", F),
]
_FORGET = [
    re.compile(r"^(?:please\s+)?forget(?:\s+(?:that|about))?\s+(?P<c>.+?)[.!]*$", F),
    re.compile(r"^(?:unut|yaddaşdan sil|hafızandan sil)\s*[:,]?\s+(?P<c>.+?)[.!]*$", F),
    re.compile(r"^(?P<c>.+?)\s+(?:haqqında\w*\s+|hakkında\w*\s+)?(?:unut|yaddaşdan sil|"
               r"hafızandan sil)[.!\s]*$", F),
]
_LIST = [
    re.compile(r"\bwhat (?:do|did) you (?:know|remember) about me\b", F),
    re.compile(r"\b(?:show|list)(?: me)?(?: all)?(?: my| your)? memor(?:y|ies)\b", F),
    re.compile(r"\bwhat have you (?:remembered|saved|stored)\b", F),
    re.compile(r"\b(?:mənim haqqımda|haqqımda)\s+nə\s+(?:bilirsən|xatırlayırsan|yadda saxlamısan)",
               F),
    re.compile(r"\byaddaşında nə var\b", F),
    re.compile(r"\bhakkımda ne (?:biliyorsun|hatırlıyorsun)\b", F),
]
_QUERY = [
    re.compile(r"^what(?:'s| is| was)\s+my\s+(?P<c>.+?)\s*\??$", F),
    re.compile(r"^do you (?:remember|know)\s+(?:what\s+)?my\s+(?P<c>.+?)\s*(?:is)?\??$", F),
    re.compile(r"^mənim\s+(?P<c>.+?)\s+(?:nədir|nə idi|kimdir|hansıdır)\s*\??$", F),
    re.compile(r"^benim\s+(?P<c>.+?)\s+(?:ne|nedir|neydi|kim)\s*\??$", F),
]

_NAME = [
    re.compile(r"^(?:hi,?\s+)?(?:my name is|call me|i'?m called)\s+(?P<v>[\w\-' ]{2,40}?)[.!]?$",
               F),
    re.compile(r"^(?:salam,?\s+)?(?:mənim\s+)?adım\s+(?P<v>[\w\-' ]{2,40}?)(?:dır|dir|dur|dür)?[.!]?$",
               F),
    re.compile(r"^(?:merhaba,?\s+)?(?:benim\s+)?adım\s+(?P<v>[\w\-' ]{2,40}?)[.!]?$", F),
]
_MY_X_IS_Y = re.compile(r"^my\s+(?P<k>[\w\- ]{2,50}?)\s+(?:is|are)\s+(?P<v>.{1,200}?)[.!]?$", F)
_PREFERENCE = re.compile(
    r"\b(i (?:really )?(?:prefer|like|love|hate|dislike|don'?t like|enjoy|want you to)|"
    r"my fav(?:ou)?rite|my (?:\w+\s+){0,3}preferences?|"
    r"üstünlük verirəm|xoşlayıram|xoşuma gəlir|sevirəm|sevmirəm|"
    r"sevimli|istəyirəm ki|tercih ederim|severim|seviyorum|sevmem|favori|en sevdiğim)\b", F)
_TEMPORAL = {
    "tomorrow": 2, "today": 1, "tonight": 1, "this week": 8, "next week": 15, "this weekend": 5,
    "on monday": 8, "on tuesday": 8, "on wednesday": 8, "on thursday": 8, "on friday": 8,
    "on saturday": 8, "on sunday": 8, "next month": 40,
    "sabah": 2, "bu gün": 1, "bugün": 1, "bu axşam": 1, "bu həftə": 8, "gələn həftə": 15,
    "gələn ay": 40, "yarın": 2, "bu akşam": 1, "bu hafta": 8, "haftaya": 15, "gelecek hafta": 15,
    "gelecek ay": 40,
}
_SELF_EVENT = re.compile(
    r"\b(my|i have|i'?ve got|i am|i'?m|mənim|\w+(?:ım|im|um|üm|m)\s+var|benim|\w+(?:ım|im|um|üm)\s+var)\b",
    F)
_QUESTION_START = re.compile(
    r"^(what|who|why|how|when|where|which|is|are|can|could|do|does|nə|kim|niyə|necə|nə vaxt|"
    r"harada|hansı|ne|neden|nasıl|nerede|hangi)\b", F)
_SECRET = [
    re.compile(r"\b(password|passwd|passcode|parol|parolum|şifrə|şifrəm|şifre|şifrem|pin)\b"
               r"\s*(?:is|:|=|-|dir|dur)?\s*\S+", F),
    re.compile(r"\b(sk-[A-Za-z0-9_\-]{12,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|"
               r"xox[baprs]-[A-Za-z0-9-]{10,})\b"),
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),  # card-like number
    re.compile(r"\b(api[ _-]?key|secret key|private key|seed phrase|recovery phrase)\b", F),
]


def _expiry_days(text: str) -> int | None:
    lowered = text.lower()
    hits = [days for phrase, days in _TEMPORAL.items()
            if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", lowered)]
    return max(hits) if hits else None


def _clean(content: str) -> str:
    return content.strip().strip("\"'“”«»").rstrip(" ,;").strip()


class MemoryPolicy:
    def __init__(self, detector: InjectionDetector | None = None):
        self.detector = detector or InjectionDetector()

    # --- commands ---------------------------------------------------------------------
    def detect_command(self, text: str) -> MemoryCommand | None:
        t = text.strip()
        for pat in _LIST:
            if pat.search(t):
                return MemoryCommand(CommandType.LIST)
        for pat in _STORE:
            m = pat.match(t)
            if m and len(_clean(m.group("c"))) >= 3:
                return MemoryCommand(CommandType.STORE, _clean(m.group("c")))
        for pat in _FORGET:
            m = pat.match(t)
            if m and len(_clean(m.group("c"))) >= 2:
                return MemoryCommand(CommandType.FORGET, _clean(m.group("c")))
        return None

    def detect_query(self, text: str) -> str | None:
        """'What's my X?' style lookups. Returns X, or None."""
        t = text.strip()
        for pat in _QUERY:
            m = pat.match(t)
            if m and 2 <= len(m.group("c")) <= 60:
                return _clean(m.group("c"))
        return None

    # --- classification ---------------------------------------------------------------
    def classify(self, content: str, source: MemorySource) -> MemoryCandidate:
        """Turn content the user wants remembered into a typed candidate."""
        confidence = 1.0 if source in (MemorySource.USER_EXPLICIT, MemorySource.API) else 0.8
        for pat in _NAME:
            m = pat.match(content)
            if m:
                return MemoryCandidate(kind=MemoryKind.FACT, key="name", content=content,
                                       source=source, confidence=confidence, reason="name")
        days = _expiry_days(content)
        if days is not None:
            return MemoryCandidate(kind=MemoryKind.EPISODIC, content=content, source=source,
                                   confidence=confidence,
                                   expires_at=iso(utcnow() + timedelta(days=days)),
                                   reason="time-bound event")
        key = None
        m = _MY_X_IS_Y.match(content)
        if m:
            key = m.group("k")
        if _PREFERENCE.search(content) or (key and key.lower().startswith(("favorite",
                                                                            "favourite"))):
            return MemoryCandidate(kind=MemoryKind.PREFERENCE, key=key, content=content,
                                   source=source, confidence=confidence, reason="preference")
        return MemoryCandidate(kind=MemoryKind.FACT, key=key, content=content, source=source,
                               confidence=confidence, reason="fact")

    @staticmethod
    def extract_name(content: str) -> str | None:
        for pat in _NAME:
            m = pat.match(content.strip())
            if m:
                return m.group("v").strip().title()
        return None

    def evaluate_statement(self, text: str) -> MemoryCandidate | None:
        """Implicit self-disclosure worth remembering, from a user message. Conservative."""
        t = text.strip()
        if not t or len(t) > 300 or self._is_question(t):
            return None
        for pat in _NAME:
            if pat.match(t):
                return self.classify(t, MemorySource.USER_STATEMENT)
        if _PREFERENCE.search(t) and re.search(r"\b(i|my|mən|mənim|ben|benim)\b|\w+(ıram|irəm|"
                                               r"uram|ürəm|erim|arım|iyorum|ıyorum)\b", t, F):
            return self.classify(t, MemorySource.USER_STATEMENT)
        if _MY_X_IS_Y.match(t):
            return self.classify(t, MemorySource.USER_STATEMENT)
        if _expiry_days(t) is not None and _SELF_EVENT.search(t):
            return self.classify(t, MemorySource.USER_STATEMENT)
        return None

    @staticmethod
    def _is_question(t: str) -> bool:
        return t.endswith("?") or bool(_QUESTION_START.match(t))

    # --- validation -------------------------------------------------------------------
    def validate(self, cand: MemoryCandidate, trust: Trust) -> tuple[bool, str]:
        if trust != Trust.USER:
            return False, "only information that comes from you can become memory"
        content = cand.content.strip()
        if len(content) < 3:
            return False, "too short"
        if len(content) > 500:
            return False, "too long (max 500 characters)"
        if content.endswith("?"):
            return False, "that is a question, not information"
        for pat in _SECRET:
            if pat.search(content):
                return False, "it looks like a password, key or card number; I never store those"
        report = self.detector.scan(content)
        if report.suspicious:
            return False, f"it contains instruction-like text ({report.summary()})"
        return True, "ok"
