"""NCBI E-utilities sources: PubMed, ClinVar, Gene."""

from __future__ import annotations

import re
from typing import Any, ClassVar

from elara.core.errors import SourceError
from elara.research.http import ResearchHttp
from elara.research.models import EvidenceType, Record, ResearchIntent, ResearchPlan
from elara.research.sources.base import ResearchSource

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class _Eutils(ResearchSource):
    db: ClassVar[str]
    health_url = f"{EUTILS}/einfo.fcgi?retmode=json"

    def __init__(self, http: ResearchHttp, api_key: str | None = None, email: str | None = None):
        super().__init__(http)
        self.api_key = api_key
        self.email = email

    def _common(self) -> dict[str, str]:
        p = {"retmode": "json", "tool": "elara"}
        if self.api_key:
            p["api_key"] = self.api_key
        if self.email:
            p["email"] = self.email
        return p

    async def _esearch(self, term: str, limit: int, **extra: str) -> list[str]:
        data = await self.http.get_json(f"{EUTILS}/esearch.fcgi", source=self.name, params={
            **self._common(), "db": self.db, "term": term, "retmax": str(limit), **extra})
        try:
            return list(data["esearchresult"]["idlist"])
        except (KeyError, TypeError) as e:
            raise SourceError(f"{self.name}: unexpected esearch response") from e

    async def _esummary(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        data = await self.http.get_json(f"{EUTILS}/esummary.fcgi", source=self.name, params={
            **self._common(), "db": self.db, "id": ",".join(ids)})
        result = data.get("result") or {}
        return [result[uid] | {"uid": uid} for uid in result.get("uids", []) if uid in result]


def _year(s: str | None) -> int | None:
    m = re.search(r"(19|20)\d{2}", s or "")
    return int(m.group(0)) if m else None


def pubmed_evidence(pubtypes: list[str]) -> EvidenceType:
    pt = {p.lower() for p in pubtypes}
    if pt & {"review", "systematic review", "meta-analysis", "scoping review"}:
        return EvidenceType.REVIEW
    if "preprint" in pt:
        return EvidenceType.PREPRINT
    if pt & {"editorial", "comment", "letter", "news", "published erratum"}:
        return EvidenceType.SECONDARY
    if pt & {"journal article", "clinical trial", "randomized controlled trial",
             "observational study", "case reports", "comparative study"}:
        return EvidenceType.PRIMARY_RESEARCH
    return EvidenceType.UNKNOWN


class PubMedSource(_Eutils):
    name = "pubmed"
    description = "PubMed biomedical literature (NCBI E-utilities)"
    intents = {ResearchIntent.LITERATURE, ResearchIntent.VARIANT, ResearchIntent.GENE}
    db = "pubmed"

    async def search(self, plan: ResearchPlan, limit: int) -> list[Record]:
        extra = {"sort": "pub_date" if plan.recent else "relevance"}
        if plan.min_year:
            extra |= {"datetype": "pdat", "mindate": str(plan.min_year), "maxdate": "3000"}
        ids = await self._esearch(plan.query, limit, **extra)
        out = []
        for d in await self._esummary(ids):
            ids_map = {a.get("idtype"): a.get("value") for a in d.get("articleids", [])}
            title = (d.get("title") or "").strip()
            if not title:
                continue
            out.append(Record(
                source=self.name, source_id=d["uid"], title=title,
                authors=[a.get("name", "") for a in d.get("authors", []) if a.get("name")],
                year=_year(d.get("pubdate")), venue=d.get("fulljournalname") or d.get("source"),
                doi=ids_map.get("doi"), pmid=d["uid"], pmcid=ids_map.get("pmc"),
                url=f"https://pubmed.ncbi.nlm.nih.gov/{d['uid']}/",
                evidence_type=pubmed_evidence(d.get("pubtype", [])),
                extra={"pubtype": d.get("pubtype", [])}))
        return out


class ClinVarSource(_Eutils):
    name = "clinvar"
    description = "ClinVar clinical significance of human variants (NCBI)"
    intents = {ResearchIntent.VARIANT}
    db = "clinvar"

    async def search(self, plan: ResearchPlan, limit: int) -> list[Record]:
        term = plan.identifiers.get("rsid") or plan.identifiers.get("hgvs") or plan.query
        ids = await self._esearch(term, limit)
        out = []
        for d in await self._esummary(ids):
            cls = d.get("germline_classification") or d.get("clinical_significance") or {}
            genes = [g.get("symbol") for g in d.get("genes", []) if g.get("symbol")]
            traits = [t.get("trait_name") for t in d.get("trait_set", []) if t.get("trait_name")]
            significance = cls.get("description") or "not provided"
            out.append(Record(
                source=self.name, source_id=d["uid"], title=d.get("title") or f"ClinVar {d['uid']}",
                url=f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{d['uid']}/",
                evidence_type=EvidenceType.DATABASE_RECORD,
                abstract=(f"Clinical significance: {significance}. Review status: "
                          f"{cls.get('review_status', 'n/a')}. Last evaluated: "
                          f"{cls.get('last_evaluated', 'n/a')}. Genes: "
                          f"{', '.join(genes) or 'n/a'}. Conditions: "
                          f"{', '.join(traits[:5]) or 'n/a'}."),
                extra={"accession": d.get("accession"), "significance": significance,
                       "review_status": cls.get("review_status"), "genes": genes,
                       "conditions": traits[:10]}))
        return out


class NCBIGeneSource(_Eutils):
    name = "ncbi_gene"
    description = "NCBI Gene records (human)"
    intents = {ResearchIntent.GENE}
    db = "gene"

    async def search(self, plan: ResearchPlan, limit: int) -> list[Record]:
        gene = plan.identifiers.get("gene")
        term = f"{gene}[sym] AND human[orgn]" if gene else f"{plan.query} AND human[orgn]"
        ids = await self._esearch(term, min(limit, 3))
        out = []
        for d in await self._esummary(ids):
            if d.get("status") not in (None, "", "0", 0) and not d.get("name"):
                continue
            out.append(Record(
                source=self.name, source_id=d["uid"],
                title=f"{d.get('name')} — {d.get('description', '')}".strip(" —"),
                url=f"https://www.ncbi.nlm.nih.gov/gene/{d['uid']}",
                evidence_type=EvidenceType.DATABASE_RECORD,
                abstract=(d.get("summary") or None),
                extra={"symbol": d.get("name"), "chromosome": d.get("chromosome"),
                       "map_location": d.get("maplocation"),
                       "organism": (d.get("organism") or {}).get("scientificname")}))
        return out
