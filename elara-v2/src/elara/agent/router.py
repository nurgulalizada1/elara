"""Tiered routing: pick the cheapest capable path for a request.

Tier 0: deterministic/local (no model)       Tier 2: strong model (complex reasoning)
Tier 1: fast/cheap model (ordinary chat)     Tier 3: research workflow (+ strong synthesis)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum

from elara.agent.intent import Intent

DETERMINISTIC = {Intent.GREETING, Intent.HOW_ARE_YOU, Intent.THANKS, Intent.GOODBYE,
                 Intent.CALCULATION, Intent.TIME, Intent.MEMORY_STORE, Intent.MEMORY_FORGET,
                 Intent.MEMORY_LIST, Intent.MEMORY_QUERY, Intent.FILE_LIST, Intent.FILE_READ,
                 Intent.OPEN_PATH, Intent.HELP}
_COMPLEX = re.compile(
    r"\b(explain in detail|in depth|step[- ]by[- ]step|analy[sz]e|analysis|compare|contrast|"
    r"trade-?offs?|design|architecture|plan|strategy|prove|derive|debug|refactor|write (a |the )?"
    r"(code|program|script|function|essay|report)|implement|evaluate|critique|pros and cons|"
    r"ətraflı|təhlil|müqayisə|analiz|planla|kod yaz|izah et|karşılaştır|ayrıntılı|detaylı|"
    r"tasarla|açıkla)\b", re.I)


class Tier(IntEnum):
    DETERMINISTIC = 0
    FAST = 1
    STRONG = 2
    RESEARCH = 3


@dataclass
class Route:
    tier: Tier
    reason: str

    @property
    def model_tier(self) -> str:
        return "strong" if self.tier in (Tier.STRONG, Tier.RESEARCH) else "fast"


class Router:
    def __init__(self, long_message_chars: int = 400):
        self.long_message_chars = long_message_chars

    def route(self, intent: Intent, text: str) -> Route:
        if intent in DETERMINISTIC:
            return Route(Tier.DETERMINISTIC, f"deterministic handler for {intent.value}")
        if intent == Intent.RESEARCH:
            return Route(Tier.RESEARCH, "research workflow")
        if len(text) > self.long_message_chars:
            return Route(Tier.STRONG, "long request")
        if _COMPLEX.search(text):
            return Route(Tier.STRONG, "complex reasoning requested")
        if text.count("?") >= 3 or len(re.findall(r"\b(and then|then|sonra|ardından)\b", text,
                                                   re.I)) >= 2:
            return Route(Tier.STRONG, "multi-step request")
        return Route(Tier.FAST, "ordinary conversation")
