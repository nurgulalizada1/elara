"""ResearchEngine: plan -> parallel source search -> validate -> dedupe -> rank -> persist."""

from __future__ import annotations

import asyncio
import json
import time

from elara.core.context import conversation_id_var, request_id_var
from elara.core.errors import SourceError
from elara.core.logging import get_logger
from elara.core.timeutil import iso
from elara.database.db import Database
from elara.research.http import cache_flags, track_cache
from elara.research.models import (
    EvidenceType,
    Record,
    ResearchIntent,
    ResearchPlan,
    ResearchResult,
    SourceOutcome,
)
from elara.research.planner import QueryPlanner
from elara.research.sources.base import ResearchSource

log = get_logger(__name__)
_EVIDENCE_RANK = {EvidenceType.DATABASE_RECORD: 0, EvidenceType.REVIEW: 1,
                  EvidenceType.PRIMARY_RESEARCH: 2, EvidenceType.PREPRINT: 3,
                  EvidenceType.UNKNOWN: 4, EvidenceType.SECONDARY: 5}


class ResearchEngine:
    def __init__(self, sources: list[ResearchSource], planner: QueryPlanner,
                 db: Database | None = None, *, per_source_limit: int = 5,
                 max_results: int = 8, source_timeout_s: float = 25.0,
                 min_primary_results: int = 3):
        self.sources = {s.name: s for s in sources}
        self.planner = planner
        self.db = db
        self.per_source_limit = per_source_limit
        self.max_results = max_results
        self.source_timeout_s = source_timeout_s
        self.min_primary_results = min_primary_results

    async def research(self, text: str, *, language: str = "en",
                       sources: list[str] | None = None) -> ResearchResult:
        plan = await self.planner.plan(text, language, sources)
        return await self.run(plan)

    async def run(self, plan: ResearchPlan) -> ResearchResult:
        t0 = time.perf_counter()
        # Explicitly requested sources run even if the heuristic says "not applicable":
        # the user asked for them, and an honest "no results" beats silent substitution.
        chosen = [self.sources[n] for n in plan.sources if n in self.sources and (
            plan.explicit_sources or self.sources[n].applicable(plan))]
        results = await asyncio.gather(*(self._one(s, plan) for s in chosen))
        found = [r for _, recs in results for r in recs]
        extra = [self.sources[n] for n in plan.supplementary
                 if n in self.sources and self.sources[n].applicable(plan)]
        if extra and len(self._merge([r.model_copy(deep=True) for r in found])) < \
                self.min_primary_results:
            results += await asyncio.gather(*(self._one(s, plan) for s in extra))
            found = [r for _, recs in results for r in recs]
        outcomes = [o for o, _ in results]
        records = self._merge(found)
        limit = min(plan.limit or self.max_results, 50)
        if plan.explicit_sources and len(chosen) == 1:
            records = records[:limit]  # keep the source's own ranking
        else:
            records = self._rank(records, plan)[:limit]
        result = ResearchResult(plan=plan, records=records, outcomes=outcomes)
        duration = int((time.perf_counter() - t0) * 1000)
        log.info("research.done", extra={"intent": plan.intent.value, "query": plan.query,
                                         "results": len(records), "duration_ms": duration,
                                         "failed": result.failed_sources})
        if self.db:
            result.research_query_id = self._persist(result, duration)
        return result

    async def _one(self, source: ResearchSource, plan: ResearchPlan
                   ) -> tuple[SourceOutcome, list[Record]]:
        t0 = time.perf_counter()
        if plan.limit and plan.explicit_sources:
            limit = min(plan.limit, 50)
        elif plan.intent == ResearchIntent.LITERATURE or source.name != "pubmed":
            limit = self.per_source_limit
        else:
            limit = 3
        track_cache()  # per-task context: gather() runs each source in its own task
        try:
            recs = await asyncio.wait_for(source.search(plan, limit), self.source_timeout_s)
            valid = [r for r in recs if self._valid(r)]
            flags = cache_flags()
            return SourceOutcome(source=source.name, ok=True, count=len(valid),
                                 cached=bool(flags) and all(flags),
                                 duration_ms=int((time.perf_counter() - t0) * 1000)), valid
        except TimeoutError:
            err, kind = f"{source.name}: timed out", "timeout"
        except SourceError as e:
            err, kind = str(e), e.kind
        except Exception as e:  # a buggy/changed API must not break the whole search
            log.exception("research.source_crash", extra={"source": source.name})
            err, kind = f"{source.name}: unexpected response ({type(e).__name__})", \
                "invalid_response"
        return SourceOutcome(source=source.name, ok=False, error=err, error_kind=kind,
                             duration_ms=int((time.perf_counter() - t0) * 1000)), []

    @staticmethod
    def _valid(r: Record) -> bool:
        if not r.title or len(r.title) < 3:
            return False
        if not (r.best_url() or r.doi or r.pmid):
            return False
        return not (r.year is not None and not (1800 <= r.year <= 2100))

    @staticmethod
    def _merge(records: list[Record]) -> list[Record]:
        merged: dict[str, Record] = {}
        by_title: dict[str, str] = {}
        for r in records:
            tkey = "".join(ch for ch in r.title.lower() if ch.isalnum())[:120]
            key = r.dedupe_key()
            if key not in merged and tkey in by_title and r.evidence_type != \
                    EvidenceType.DATABASE_RECORD:
                key = by_title[tkey]
            if key not in merged:
                merged[key] = r
                by_title.setdefault(tkey, key)
                continue
            base = merged[key]
            if r.source not in base.also_in and r.source != base.source:
                base.also_in.append(r.source)
            for f in ("doi", "pmid", "pmcid", "abstract", "venue", "year", "citation_count"):
                if getattr(base, f) in (None, "") and getattr(r, f) not in (None, ""):
                    setattr(base, f, getattr(r, f))
            if base.evidence_type == EvidenceType.UNKNOWN:
                base.evidence_type = r.evidence_type
            if not base.authors:
                base.authors = r.authors
        return list(merged.values())

    @staticmethod
    def _rank(records: list[Record], plan: ResearchPlan) -> list[Record]:
        def score(r: Record):
            corroboration = -len(r.also_in)
            if plan.intent != ResearchIntent.LITERATURE:
                return (_EVIDENCE_RANK[r.evidence_type], corroboration, -(r.year or 0))
            if plan.recent:
                return (-(r.year or 0), corroboration, _EVIDENCE_RANK[r.evidence_type])
            return (corroboration, _EVIDENCE_RANK[r.evidence_type], -(r.citation_count or 0))
        return sorted(records, key=score)

    def _persist(self, result: ResearchResult, duration_ms: int) -> int | None:
        try:
            with self.db.transaction() as c:
                cur = c.execute(
                    "INSERT INTO research_queries(request_id, conversation_id, query, intent,"
                    " sources_json, result_count, duration_ms, created_at) VALUES (?,?,?,?,?,?,?,?)",
                    (request_id_var.get(), conversation_id_var.get(), result.plan.original,
                     result.plan.intent.value,
                     json.dumps([o.model_dump() for o in result.outcomes]), len(result.records),
                     duration_ms, iso()))
                qid = int(cur.lastrowid)
                for i, r in enumerate(result.records, 1):
                    c.execute(
                        "INSERT INTO sources(research_query_id, rank, source, external_id, title,"
                        " doi, pmid, url, year, evidence_type, metadata_json)"
                        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (qid, i, r.source, r.source_id, r.title, r.doi, r.pmid, r.best_url(),
                         r.year, r.evidence_type.value,
                         json.dumps({"venue": r.venue, "authors": r.authors[:10],
                                     "also_in": r.also_in}, ensure_ascii=False)))
            return qid
        except Exception:
            log.exception("research.persist_failed")
            return None
