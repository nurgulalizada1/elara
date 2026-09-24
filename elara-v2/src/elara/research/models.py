from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EvidenceType(StrEnum):
    PRIMARY_RESEARCH = "primary_research"
    REVIEW = "review"                  # reviews, systematic reviews, meta-analyses
    PREPRINT = "preprint"              # not peer reviewed
    DATABASE_RECORD = "database_record"
    SECONDARY = "secondary"            # editorials, comments, news, books
    UNKNOWN = "unknown"


class ResearchIntent(StrEnum):
    LITERATURE = "literature"
    VARIANT = "variant"
    GENE = "gene"


class Record(BaseModel):
    source: str
    source_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    url: str | None = None
    abstract: str | None = None
    evidence_type: EvidenceType = EvidenceType.UNKNOWN
    citation_count: int | None = None
    also_in: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    def dedupe_key(self) -> str:
        if self.doi:
            return "doi:" + self.doi.lower()
        if self.pmid:
            return "pmid:" + self.pmid
        if self.evidence_type == EvidenceType.DATABASE_RECORD:
            return f"{self.source}:{self.source_id}"
        return "title:" + "".join(ch for ch in self.title.lower() if ch.isalnum())[:120]

    def best_url(self) -> str | None:
        if self.url:
            return self.url
        if self.doi:
            return f"https://doi.org/{self.doi}"
        if self.pmid:
            return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"
        return None

    def citation(self) -> str:
        authors = ", ".join(self.authors[:3]) + (" et al." if len(self.authors) > 3 else "")
        parts = [self.title.rstrip(".") + "."]
        if authors:
            parts.append(authors + ".")
        venue_year = ", ".join(str(x) for x in (self.venue, self.year) if x)
        if venue_year:
            parts.append(venue_year + ".")
        ids = []
        if self.pmid:
            ids.append(f"PMID:{self.pmid}")
        if self.doi:
            ids.append(f"DOI:{self.doi}")
        label = self.evidence_type.value.replace("_", " ")
        line = " ".join(parts) + f" [{label}; {self.source}]"
        if ids:
            line += " " + " · ".join(ids)
        url = self.best_url()
        if url and not self.doi and not self.pmid:
            line += f" {url}"
        return line


class SourceOutcome(BaseModel):
    source: str
    ok: bool
    count: int = 0
    error: str | None = None
    duration_ms: int = 0


class ResearchPlan(BaseModel):
    intent: ResearchIntent
    query: str                      # search string sent to sources
    original: str
    sources: list[str]
    recent: bool = False
    min_year: int | None = None
    identifiers: dict[str, str] = Field(default_factory=dict)  # rsid, variant_id, gene


class ResearchResult(BaseModel):
    plan: ResearchPlan
    records: list[Record]
    outcomes: list[SourceOutcome]
    research_query_id: int | None = None

    @property
    def failed_sources(self) -> list[str]:
        return [o.source for o in self.outcomes if not o.ok]
