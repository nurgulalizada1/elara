"""Literature sources: Europe PMC, Crossref, Semantic Scholar."""

from __future__ import annotations

from elara.research.http import ResearchHttp
from elara.research.models import EvidenceType, Record, ResearchIntent, ResearchPlan
from elara.research.sources.base import ResearchSource, clean_text
from elara.research.sources.ncbi import pubmed_evidence

PREPRINT_VENUES = ("biorxiv", "medrxiv", "arxiv", "research square", "preprints.org", "ssrn")


class EuropePMCSource(ResearchSource):
    name = "europepmc"
    description = "Europe PMC literature incl. preprints and abstracts"
    intents = {ResearchIntent.LITERATURE}
    health_url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=test"
                  "&format=json&pageSize=1")

    async def search(self, plan: ResearchPlan, limit: int) -> list[Record]:
        query = plan.query
        if plan.min_year:
            query = f"({query}) AND PUB_YEAR:[{plan.min_year} TO 3000]"
        params = {"query": query, "format": "json", "pageSize": str(limit), "resultType": "core"}
        if plan.recent:
            params["sort"] = "P_PDATE_D desc"
        data = await self.http.get_json(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search", params=params,
            source=self.name)
        out = []
        for r in (data.get("resultList") or {}).get("result", []):
            title = clean_text(r.get("title"))
            if not title:
                continue
            pubtypes = (r.get("pubTypeList") or {}).get("pubType") or []
            if isinstance(pubtypes, str):
                pubtypes = [pubtypes]
            evidence = (EvidenceType.PREPRINT if r.get("source") == "PPR"
                        else pubmed_evidence(pubtypes))
            journal = ((r.get("journalInfo") or {}).get("journal") or {}).get("title") \
                or r.get("journalTitle")
            authors = [a.get("fullName") for a in (r.get("authorList") or {}).get("author", [])
                       if a.get("fullName")] or [
                a.strip() for a in (r.get("authorString") or "").split(",") if a.strip()]
            year = r.get("pubYear")
            out.append(Record(
                source=self.name, source_id=f"{r.get('source')}:{r.get('id')}", title=title,
                authors=authors, year=int(year) if year and str(year).isdigit() else None,
                venue=journal, doi=r.get("doi"), pmid=r.get("pmid"), pmcid=r.get("pmcid"),
                url=f"https://europepmc.org/article/{r.get('source')}/{r.get('id')}",
                abstract=clean_text(r.get("abstractText")), evidence_type=evidence,
                citation_count=r.get("citedByCount")))
        return out


class CrossrefSource(ResearchSource):
    name = "crossref"
    description = "Crossref DOI metadata across all publishers"
    intents = {ResearchIntent.LITERATURE}
    health_url = "https://api.crossref.org/works?rows=0"

    def __init__(self, http: ResearchHttp, mailto: str | None = None):
        super().__init__(http)
        self.mailto = mailto

    async def search(self, plan: ResearchPlan, limit: int) -> list[Record]:
        params = {"query": plan.query, "rows": str(limit),
                  "select": "DOI,title,author,issued,container-title,type,URL,abstract,"
                            "is-referenced-by-count"}
        if plan.min_year:
            params["filter"] = f"from-pub-date:{plan.min_year}"
        if plan.recent:
            params |= {"sort": "published", "order": "desc"}
        if self.mailto:
            params["mailto"] = self.mailto
        data = await self.http.get_json("https://api.crossref.org/works", params=params,
                                        source=self.name)
        out = []
        for it in (data.get("message") or {}).get("items", []):
            title = clean_text((it.get("title") or [None])[0])
            if not title:
                continue
            parts = ((it.get("issued") or {}).get("date-parts") or [[None]])[0]
            ctype = it.get("type", "")
            evidence = {"posted-content": EvidenceType.PREPRINT,
                        "book": EvidenceType.SECONDARY, "book-chapter": EvidenceType.SECONDARY,
                        "dataset": EvidenceType.DATABASE_RECORD}.get(ctype, EvidenceType.UNKNOWN)
            if evidence == EvidenceType.UNKNOWN and "review" in title.lower():
                evidence = EvidenceType.REVIEW
            out.append(Record(
                source=self.name, source_id=it.get("DOI", title), title=title,
                authors=[" ".join(x for x in (a.get("given"), a.get("family")) if x)
                         for a in it.get("author", []) if a.get("family")],
                year=parts[0] if parts and isinstance(parts[0], int) else None,
                venue=(it.get("container-title") or [None])[0], doi=it.get("DOI"),
                url=it.get("URL"), abstract=clean_text(it.get("abstract")),
                evidence_type=evidence, citation_count=it.get("is-referenced-by-count"),
                extra={"crossref_type": ctype}))
        return out


class SemanticScholarSource(ResearchSource):
    name = "semantic_scholar"
    description = "Semantic Scholar academic graph"
    intents = {ResearchIntent.LITERATURE}
    health_url = "https://api.semanticscholar.org/graph/v1/paper/search?query=test&limit=1"

    def __init__(self, http: ResearchHttp, api_key: str | None = None):
        super().__init__(http)
        self.api_key = api_key

    async def search(self, plan: ResearchPlan, limit: int) -> list[Record]:
        params = {"query": plan.query, "limit": str(limit),
                  "fields": "title,authors,year,venue,externalIds,url,abstract,"
                            "publicationTypes,citationCount"}
        if plan.min_year:
            params["year"] = f"{plan.min_year}-"
        headers = {"x-api-key": self.api_key} if self.api_key else None
        data = await self.http.get_json("https://api.semanticscholar.org/graph/v1/paper/search",
                                        params=params, headers=headers, source=self.name)
        out = []
        for p in data.get("data") or []:
            if not p.get("title"):
                continue
            ext = p.get("externalIds") or {}
            types = {t.lower() for t in (p.get("publicationTypes") or [])}
            venue = p.get("venue") or None
            if "review" in types:
                evidence = EvidenceType.REVIEW
            elif (venue and venue.lower().startswith(PREPRINT_VENUES)) or (
                    ext.get("ArXiv") and not ext.get("DOI")):
                evidence = EvidenceType.PREPRINT
            elif types & {"journalarticle", "clinicaltrial", "casereport", "study"}:
                evidence = EvidenceType.PRIMARY_RESEARCH
            else:
                evidence = EvidenceType.UNKNOWN
            out.append(Record(
                source=self.name, source_id=p.get("paperId", ""), title=p["title"].strip(),
                authors=[a.get("name") for a in p.get("authors", []) if a.get("name")],
                year=p.get("year"), venue=venue, doi=ext.get("DOI"),
                pmid=str(ext["PubMed"]) if ext.get("PubMed") else None, url=p.get("url"),
                abstract=p.get("abstract"), evidence_type=evidence,
                citation_count=p.get("citationCount")))
        return out
