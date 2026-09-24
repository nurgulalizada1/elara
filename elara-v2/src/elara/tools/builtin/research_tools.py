"""Research tools. Results are external data and are always untrusted."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from elara.research.engine import ResearchEngine
from elara.research.models import ResearchPlan
from elara.research.planner import classify_intent, extract_identifiers
from elara.research.sources.base import ResearchSource
from elara.security.untrusted import Trust
from elara.tools.base import Tool, ToolContext


class ResearchInput(BaseModel):
    query: str = Field(min_length=2, max_length=500, description="Research question or keywords "
                                                                  "(English works best)")
    sources: list[str] | None = Field(default=None, description="Optional subset of sources")
    recent: bool = False


class RecordOut(BaseModel):
    n: int
    title: str
    source: str
    evidence_type: str
    year: int | None = None
    venue: str | None = None
    authors: list[str] = Field(default_factory=list)
    doi: str | None = None
    pmid: str | None = None
    url: str | None = None
    abstract: str | None = None


class ResearchOutput(BaseModel):
    query: str
    intent: str
    results: list[RecordOut]
    failed_sources: dict[str, str]


def _records(result) -> list[RecordOut]:
    return [RecordOut(n=i, title=r.title, source=r.source, evidence_type=r.evidence_type.value,
                      year=r.year, venue=r.venue, authors=r.authors[:5], doi=r.doi, pmid=r.pmid,
                      url=r.best_url(), abstract=(r.abstract or "")[:1200] or None)
            for i, r in enumerate(result.records, 1)]


class ResearchSearchTool(Tool):
    name: ClassVar[str] = "research_search"
    description: ClassVar[str] = (
        "Search scientific sources (PubMed, Europe PMC, Semantic Scholar, Crossref, ClinVar, "
        "NCBI Gene, Ensembl, gnomAD). Intent (literature / variant / gene) is detected "
        "automatically. Only cite what this tool returns.")
    Input = ResearchInput
    Output = ResearchOutput
    output_trust = Trust.UNTRUSTED
    category = "research"
    timeout_s = 60.0

    def __init__(self, engine: ResearchEngine):
        self.engine = engine

    def source_label(self, args: ResearchInput) -> str:
        return f"research:{args.query[:80]}"

    async def run(self, args: ResearchInput, ctx: ToolContext) -> ResearchOutput:
        plan = await self.engine.planner.plan(args.query, "en", args.sources)
        if args.recent and not plan.recent:
            plan = plan.model_copy(update={"recent": True})
        result = await self.engine.run(plan)
        return ResearchOutput(query=plan.query, intent=plan.intent.value,
                              results=_records(result),
                              failed_sources={o.source: o.error or "" for o in result.outcomes
                                              if not o.ok})


class SourceSearchTool(Tool):
    """One tool per source (e.g. pubmed_search, clinvar_search) for direct API/CLI use."""

    Input = ResearchInput
    Output = ResearchOutput
    output_trust = Trust.UNTRUSTED
    category = "research"
    timeout_s = 40.0
    exposed_to_llm = False  # keeps the model's tool list (and token cost) small

    def __init__(self, engine: ResearchEngine, source: ResearchSource):
        self.engine = engine
        self.source = source
        self.name = f"{source.name}_search"  # type: ignore[misc]
        self.description = f"Search {source.description}."  # type: ignore[misc]

    async def run(self, args: ResearchInput, ctx: ToolContext) -> ResearchOutput:
        ids = extract_identifiers(args.query)
        intent = classify_intent(args.query, ids)
        if intent not in self.source.intents:
            intent = next(iter(self.source.intents))
        plan = ResearchPlan(intent=intent, query=args.query, original=args.query,
                            sources=[self.source.name], recent=args.recent, identifiers=ids)
        result = await self.engine.run(plan)
        return ResearchOutput(query=args.query, intent=intent.value, results=_records(result),
                              failed_sources={o.source: o.error or "" for o in result.outcomes
                                              if not o.ok})
