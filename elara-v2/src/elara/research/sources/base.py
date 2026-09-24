from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from elara.research.http import ResearchHttp
from elara.research.models import Record, ResearchIntent, ResearchPlan


class ResearchSource(ABC):
    """One external scientific resource. Implementations only map API <-> Record."""

    name: ClassVar[str]
    description: ClassVar[str]
    intents: ClassVar[set[ResearchIntent]]
    health_url: ClassVar[str | None] = None

    def __init__(self, http: ResearchHttp):
        self.http = http

    @abstractmethod
    async def search(self, plan: ResearchPlan, limit: int) -> list[Record]: ...

    def applicable(self, plan: ResearchPlan) -> bool:
        return plan.intent in self.intents


def clean_text(s: str | None) -> str | None:
    if not s:
        return None
    import re
    s = re.sub(r"<[^>]+>", "", s)  # strip JATS/HTML tags
    return re.sub(r"\s+", " ", s).strip() or None
