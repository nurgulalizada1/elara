"""Deterministic intent classification (Tier 0). Anything unclear -> GENERAL (LLM)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from elara.core.errors import ToolError
from elara.memory.policy import CommandType, MemoryPolicy
from elara.research.planner import RSID, VARIANT_ID
from elara.tools.builtin.calculator import evaluate

F = re.I


class Intent(StrEnum):
    GREETING = "greeting"
    HOW_ARE_YOU = "how_are_you"
    THANKS = "thanks"
    GOODBYE = "goodbye"
    CALCULATION = "calculation"
    TIME = "time"
    MEMORY_STORE = "memory_store"
    MEMORY_FORGET = "memory_forget"
    MEMORY_LIST = "memory_list"
    MEMORY_QUERY = "memory_query"
    RESEARCH = "research"
    FILE_LIST = "file_list"
    OPEN_PATH = "open_path"
    REFERENCE = "reference"   # follow-up about an earlier result ("the second paper")
    GENERAL = "general"


@dataclass
class IntentResult:
    intent: Intent
    slots: dict[str, Any] = field(default_factory=dict)


_NAME = r"(?:[\s,!]+(?:elara|ELARA|Elara))?"
_GREETING = re.compile(rf"^(salam|salamlar|hi|hello|hey|hey there|merhaba|selam|sabahın xeyir|"
                       rf"axşamın xeyir|good morning|good evening|günaydın|iyi akşamlar)"
                       rf"{_NAME}[\s!.]*$", F)
_HOW = re.compile(r"\b(necəsən|necəsiz|nə var nə yox|how are you|how's it going|nasılsın|"
                  r"nasılsınız|naber)\b", F)
_GREET_PREFIX = re.compile(rf"^(salam|hi|hello|hey|merhaba|selam){_NAME}[\s,!.]*", F)
_THANKS = re.compile(r"^(çox\s+)?(sağ ?ol(un)?|təşəkkür(lər)?( edirəm)?|thanks?( you)?( so much)?|"
                     r"thx|teşekkür(ler| ederim)?|sağol)[\s!.]*(elara)?[\s!.]*$", F)
_BYE = re.compile(r"^(bye|goodbye|see you|görüşərik|hələlik|sağ ol,? görüşərik|görüşürüz|"
                  r"hoşça kal|güle güle)[\s!.]*$", F)
_TIME = re.compile(r"\b(what time is it|what's the time|current time|what(?:'s| is) the date|"
                   r"what day is (it|today)|today'?s date|saat neçədir|saat neçə|bu gün ayın "
                   r"neçəsidir|bu gün nə gündür|bugün ayın kaçı|saat kaç|bugün günlerden ne|"
                   r"bugün hangi gün)\b", F)
_CALC_TRIGGERS = re.compile(
    r"^(what(?:'s| is)|calculate|compute|evaluate|how much is|hesabla|hesapla|neçədir|"
    r"neçə edir|kaç eder|kaçtır|nə qədərdir|=)\s*|\s*(=|neçədir|neçə edir|kaç eder|kaçtır|"
    r"nə edir|equals|is)?\s*\??$", F)
_MATH = re.compile(r"^[\d\s.+\-*/×÷^()%,√]*(sqrt|log|ln|sin|cos|tan|exp|factorial|pi|abs)?"
                   r"[\d\s.+\-*/×÷^()%,√a-z]*$", F)
_RESEARCH_WORDS = re.compile(
    r"\b(research|papers?|studies|study|literature|publications?|articles?|preprints?|"
    r"reviews? on|pubmed|europe ?pmc|semantic scholar|crossref|clinvar|gnomad|ensembl|"
    r"tədqiqat\w*|məqalə\w*|araşdırma\w*|elmi|makale\w*|yayın\w*|çalışma\w*)\b", F)
_SEARCH_VERBS = re.compile(r"\b(find|search|look up|lookup|show|get|list|latest|recent|what does "
                           r"the literature|tap|axtar|göstər|son|bul|ara|getir|listele)\w*", F)
_FILE_LIST = [
    re.compile(r"^(?:list|show)(?: me)?(?: the)? files(?: in| inside| under)?\s*(?P<p>.*?)[?.!]*$",
               F),
    re.compile(r"^(?P<p>\S+?)\s+(?:qovluğundakı|qovluğunda olan|klasöründeki)\s+"
               r"(?:faylları|dosyaları)\s+(?:göstər|sadala|listele|göster)[.!]*$", F),
    re.compile(r"^(?:faylları|dosyaları)\s+(?:göstər|listele|göster)[.!]*$", F),
]
_OPEN = [
    re.compile(r"^open\s+(?:up\s+)?(?:my\s+|the\s+)?(?P<t>.+?)(?:\s+folder|\s+directory)?[.!]*$",
               F),
    re.compile(r"^(?:mənim\s+)?(?P<t>.+?)\s+(?:qovluğunu|faylını)\s+aç[.!]*$", F),
    re.compile(r"^(?:benim\s+)?(?P<t>.+?)\s+(?:klasörünü|dosyasını)\s+aç[.!]*$", F),
]
_ORDINALS = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "fifth": 5,
    "birinci": 1, "ilk": 1, "ikinci": 2, "üçüncü": 3, "dördüncü": 4, "beşinci": 5,
    "last": -1, "sonuncu": -1, "sonuncusu": -1, "son": -1,
}
_REF_NOUN = r"(one|paper|article|study|result|record|variant|gene|məqalə\w*|nəticə\w*|tədqiqat\w*|" \
            r"makale\w*|sonuç\w*|çalışma\w*|biri|si|sı|su|sü)"
_ORDINAL_REF = re.compile(rf"\b(?P<o>{'|'.join(_ORDINALS)})\w*\s*{_REF_NOUN}?\b", F)
_NUMBER_REF = re.compile(r"(?:#|\bnumber\s+|\bnömrə\s+|\bnumara\s+|\[)(?P<n>\d{1,2})\]?", F)
_DEICTIC_REF = re.compile(r"\b(that|this|the previous|the last) (one|paper|article|study|result)\b|"
                          r"\b(o|bu|həmin|əvvəlki) (məqalə|tədqiqat|nəticə)\w*|"
                          r"\b(o|bu|önceki) (makale|çalışma|sonuç)\w*", F)


def _strip_greeting(text: str) -> str:
    return _GREET_PREFIX.sub("", text).strip()


def extract_math(text: str) -> str | None:
    t = text.strip()
    candidate = _CALC_TRIGGERS.sub("", t).strip().rstrip("?").strip()
    if not candidate or not re.search(r"\d", candidate):
        return None
    if not re.search(r"[+\-*/×÷^%]|sqrt|log|sin|cos|tan|factorial|√", candidate):
        return None
    if not _MATH.match(candidate):
        return None
    try:
        evaluate(candidate)
    except ToolError:
        return None
    return candidate


class IntentClassifier:
    def __init__(self, memory_policy: MemoryPolicy):
        self.memory_policy = memory_policy

    def classify(self, text: str, *, has_references: bool = False) -> IntentResult:
        t = text.strip()
        if _GREETING.match(t):
            return IntentResult(Intent.GREETING)
        if _HOW.search(t) and len(t.split()) <= 6:
            return IntentResult(Intent.HOW_ARE_YOU)
        if _THANKS.match(t):
            return IntentResult(Intent.THANKS)
        if _BYE.match(t):
            return IntentResult(Intent.GOODBYE)
        body = _strip_greeting(t)
        if expr := extract_math(body):
            return IntentResult(Intent.CALCULATION, {"expression": expr})
        if _TIME.search(body) and len(body.split()) <= 8:
            return IntentResult(Intent.TIME)
        cmd = self.memory_policy.detect_command(body)
        if cmd:
            mapping = {CommandType.STORE: Intent.MEMORY_STORE,
                       CommandType.FORGET: Intent.MEMORY_FORGET,
                       CommandType.LIST: Intent.MEMORY_LIST}
            return IntentResult(mapping[cmd.type], {"text": cmd.text})
        if has_references and (ref := self._reference(body)) is not None:
            return IntentResult(Intent.REFERENCE, {"index": ref})
        if self._is_research(body):
            return IntentResult(Intent.RESEARCH, {"query": body})
        for pat in _FILE_LIST:
            if m := pat.match(body):
                return IntentResult(Intent.FILE_LIST, {"path": (m.groupdict().get("p") or ".")
                                                       .strip() or "."})
        for pat in _OPEN:
            if m := pat.match(body):
                return IntentResult(Intent.OPEN_PATH, {"target": m.group("t").strip()})
        if q := self.memory_policy.detect_query(body):
            return IntentResult(Intent.MEMORY_QUERY, {"query": q})
        return IntentResult(Intent.GENERAL)

    @staticmethod
    def _is_research(t: str) -> bool:
        if RSID.search(t) or VARIANT_ID.search(t):
            return True
        return bool(_RESEARCH_WORDS.search(t) and _SEARCH_VERBS.search(t))

    @staticmethod
    def _reference(t: str) -> int | None:
        if m := _NUMBER_REF.search(t):
            return int(m.group("n"))
        if m := _ORDINAL_REF.search(t):
            word = m.group("o").lower()
            if word == "son" and not re.search(rf"\bson\w*\s+{_REF_NOUN}", t, F):
                return None
            return _ORDINALS[word]
        if _DEICTIC_REF.search(t):
            return 0  # "that paper": the most recently discussed item
        return None
