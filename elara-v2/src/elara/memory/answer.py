"""Local (no-LLM) answering of questions about the user.

Order of use in the assistant: long-term memory first, then statements the user made
earlier in the *current* conversation (short-term context, never persisted). Answers
are only given when a stored statement clearly covers the question; otherwise the
request falls through to the next resolver (ultimately the LLM).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from elara.memory.models import Memory
from elara.memory.store import STOPWORDS

F = re.I
_WORD = re.compile(r"\w+", re.UNICODE)

_PERSONAL = [
    # English
    re.compile(r"\bwhat(?:'s| is| was| are| were)\s+my\b", F),
    re.compile(r"^(?:what|which|who|where|when)\b.*\bmy\b", F),
    re.compile(r"\bwho am i\b", F),
    re.compile(r"\b(?:what|which)\b[^?]{0,80}\b(?:do|did) i\s+(?:prefer|like|love|use|want|"
               r"study|enjoy|speak|live|work|need)\b", F),
    re.compile(r"^(?:do|did|am) i\s+(?:prefer|like|love|have|own|use|want|speak|study)\b", F),
    re.compile(r"\bwhere do i (?:live|work|study)\b", F),
    # Azerbaijani
    re.compile(r"\bmənim\b.*\b(?:nə|nədir|nə idi|hansı\w*|kim\w*|harada\w*)\b", F),
    re.compile(r"\b(?:nə|nəyi|hansı\w*|harada|kim)\b.*\w+(?:ram|rəm|yam|yəm|ıq|ik|uq|ük)\s*\??$",
               F),
    re.compile(r"\badım nədir\b|\bmən kiməm\b", F),
    # Turkish
    re.compile(r"\bbenim\b.*\b(?:ne|neydi|nedir|hangi\w*|kim\w*|nerede\w*)\b", F),
    re.compile(r"\b(?:ne|neyi|hangi\w*|nerede|kim)\b.*\w+(?:yorum|ıyorum|iyorum|uyorum|üyorum|"
               r"rım|rim|rum|rüm)\s*\??$", F),
    re.compile(r"\badım ne\b|\bben kimim\b", F),
]
_NAME_QUESTION = re.compile(r"\b(what(?:'s| is) my name|who am i|adım nədir|mən kiməm|mənim adım|"
                            r"adım ne|ben kimim|benim adım)\b", F)
_REQUEST_START = re.compile(r"^(can|could|would|will|please|how|help|tell me how|show me how)\b",
                            F)

QUESTION_WORDS = {
    "what", "which", "who", "where", "when", "why", "how", "do", "did", "does", "am", "is", "are",
    "was", "were", "i", "my", "me", "for", "with", "about", "to", "of", "the", "a", "an", "you",
    "nə", "nəyi", "nədir", "hansı", "hansıdır", "harada", "kim", "kimdir", "mən", "mənim", "ilə",
    "üçün", "idi", "ne", "neyi", "nedir", "hangi", "nerede", "ben", "benim", "için", "ile", "mi",
    "mı", "mu", "mü",
}


def is_personal_question(text: str) -> bool:
    t = text.strip()
    if _REQUEST_START.match(t):
        return False
    return any(p.search(t) for p in _PERSONAL)


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower())
            if len(w) >= 3 and w not in STOPWORDS and w not in QUESTION_WORDS]


def _same_stem(a: str, b: str) -> bool:
    if a == b:
        return True
    k = min(len(a), len(b), 5)
    return k >= 4 and a[:k] == b[:k]


def coverage(question: str, statement: str) -> float:
    """Fraction of the question's content words that the statement covers."""
    q = _tokens(question)
    if not q:
        return 0.0
    s = _tokens(statement)
    return sum(any(_same_stem(w, x) for x in s) for w in q) / len(q)


@dataclass
class LocalAnswer:
    content: str
    source: str            # "memory" | "conversation"
    memory_id: int | None = None
    score: float = 0.0


class LocalAnswerer:
    def __init__(self, min_coverage: float = 0.5):
        self.min_coverage = min_coverage

    def from_memories(self, question: str, candidates: Iterable[Memory],
                      name_memory: Memory | None = None) -> LocalAnswer | None:
        if _NAME_QUESTION.search(question) and name_memory is not None:
            return LocalAnswer(name_memory.content, "memory", name_memory.id, 1.0)
        best: LocalAnswer | None = None
        for m in candidates:
            score = coverage(question, (m.key or "") + " " + m.content)
            if score >= self.min_coverage and (best is None or score > best.score):
                best = LocalAnswer(m.content, "memory", m.id, score)
        return best

    def from_statements(self, question: str, statements: Iterable[str]) -> LocalAnswer | None:
        best: LocalAnswer | None = None
        for s in statements:
            score = coverage(question, s)
            if _NAME_QUESTION.search(question) and re.search(
                    r"\b(my name|call me|adım|ismim)\b", s, F):
                score = max(score, 1.0)
            if score >= self.min_coverage and (best is None or score >= best.score):
                best = LocalAnswer(s, "conversation", None, score)  # later statements win ties
        return best
