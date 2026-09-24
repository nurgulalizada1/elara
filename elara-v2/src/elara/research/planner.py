"""Research intent classification and query planning (deterministic, optional LLM rewrite)."""

from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, Field

from elara.core.logging import get_logger
from elara.providers.base import ChatMessage, CompletionRequest
from elara.providers.service import LLMService
from elara.research.models import ResearchIntent, ResearchPlan

log = get_logger(__name__)

RSID = re.compile(r"\brs\d{3,12}\b", re.I)
VARIANT_ID = re.compile(r"\b(?:chr)?([0-9]{1,2}|X|Y|MT?)[-:](\d{1,9})[-:]([ACGT]+)[-:>]([ACGT]+)\b",
                        re.I)
HGVS = re.compile(r"\b(?:N[MCGRP]_\d+(?:\.\d+)?:)?[cgpn]\.[A-Za-z0-9_*>+\-]+\b")
GENE_SYMBOL = re.compile(r"\b([A-Z][A-Z0-9]{1,9}(?:-[A-Z0-9]+)?)\b")
NOT_GENES = {"DNA", "RNA", "PCR", "CRISPR", "MRNA", "HIV", "COVID", "USA", "WHO", "AI", "API",
             "PDF", "OK", "ELARA", "UK", "EU", "SARS", "NGS", "GWAS", "SNP", "SNV", "CNV",
             "ATAC", "SEQ", "SCRNA", "LLM", "FDA", "NIH", "HPC", "GPU", "ML", "RT", "QPCR",
             "NCBI", "PMID", "PMIDS", "PMC", "DOI", "DOIS", "ID", "IDS", "URL", "EPMC",
             "GRCH38", "GRCH37", "HGVS", "OMIM"}
# Explicitly named sources ("search PubMed for ..."). Order matters: longer names first.
SOURCE_ALIASES: list[tuple[str, re.Pattern[str]]] = [
    ("ncbi_gene", re.compile(r"\b(?:ncbi['’]?s?\s+gene|entrez\s+gene)\b", re.I)),
    ("europepmc", re.compile(r"\beurope\s*pmc\b|\bepmc\b", re.I)),
    ("semantic_scholar", re.compile(r"\bsemantic\s*scholar\b", re.I)),
    ("pubmed", re.compile(r"\bpub\s*med\b|\bmedline\b", re.I)),
    ("crossref", re.compile(r"\bcross\s*ref\b", re.I)),
    ("clinvar", re.compile(r"\bclin\s*var\b", re.I)),
    ("ensembl", re.compile(r"\bensembl\b", re.I)),
    ("gnomad", re.compile(r"\bgnomad\b", re.I)),
]
_LIMIT = re.compile(r"\b(?:first|top)\s+(\d{1,2})\b|\b(\d{1,2})\s+(?:results|papers|articles|"
                    r"hits|records|studies)\b|\bilk\s+(\d{1,2})\b", re.I)
_OUTPUT_CLAUSE = re.compile(
    r"\s*(?:,|\band\b)?\s*\b(?:give|show|list|return|tell|send)\s+me\b.*$|"
    r"\s*\bwith\s+(?:their\s+)?(?:titles?|pmids?|pubmed\s+ids?|ids?|dois?|links?)\b.*$", re.I)
_SOURCE_PHRASE = re.compile(r"\b(?:in|on|from|using|via)?\s*(?:the\s+)?(?:ncbi['’]?s?\s+gene|"
                            r"entrez\s+gene|europe\s*pmc|semantic\s*scholar|pub\s*med|medline|"
                            r"cross\s*ref|clin\s*var|ensembl|gnomad|ncbi)\b(?:[-'’]?(?:də|da|dən|dan|"
                            r"de|den|te|ta|ten|tan)\b)?(?:\s+(?:database|entry|record|search))?",
                            re.I)
_VARIANT_WORDS = re.compile(r"\b(variant|variants|mutation|mutations|pathogenic|clinvar|gnomad|"
                            r"allele frequency|variantı|variantlar|mutasiya|mutasyon|varyant)\w*",
                            re.I)
_GENE_WORDS = re.compile(r"\b(gene|genes|geni|genin|gen|genler|ensembl|locus)\b", re.I)
_RECENT = re.compile(r"\b(recent|latest|new|newest|current|this year|son|ən son|yeni|güncel|"
                     r"yakın zamanda|son zamanlarda)\b", re.I)
_FILLER = re.compile(
    r"\b(please|find|search|look up|lookup|get|show me|give me|tell me about|what does|"
    r"(the )?(most )?(recent|latest|new|newest) (research|studies|papers|articles|literature|"
    r"publications|evidence)( on| about| regarding| for| into)?|research|studies|papers|"
    r"articles|literature|publications|about|regarding|on the topic of|"
    r"haqqında|barəsində|üzrə|son|ən son|yeni|tədqiqat\w*|məqalə\w*|araşdırma\w*|tap|axtar|"
    r"göstər|hakkında|ile ilgili|makale\w*|bul|ara|getir|güncel)\b", re.I)


class RewrittenQuery(BaseModel):
    query: str = Field(description="Concise English search query (keywords, no filler)")


def extract_identifiers(text: str) -> dict[str, str]:
    ids: dict[str, str] = {}
    if m := RSID.search(text):
        ids["rsid"] = m.group(0).lower()
    if m := VARIANT_ID.search(text):
        chrom = m.group(1).upper().replace("MT", "M")
        ids["variant_id"] = f"{chrom}-{m.group(2)}-{m.group(3).upper()}-{m.group(4).upper()}"
    if m := HGVS.search(text):
        ids["hgvs"] = m.group(0)
    for cand in GENE_SYMBOL.findall(text):
        if cand not in NOT_GENES and not RSID.fullmatch(cand) and any(c.isalpha() for c in cand) \
                and len(cand) >= 2 and not cand.isdigit():
            ids["gene"] = cand
            break
    return ids


def classify_intent(text: str, ids: dict[str, str]) -> ResearchIntent:
    if ids.get("rsid") or ids.get("variant_id") or ids.get("hgvs") or (
            _VARIANT_WORDS.search(text) and ids.get("gene")):
        return ResearchIntent.VARIANT
    if ids.get("gene") and (_GENE_WORDS.search(text) or len(text.split()) <= 3):
        return ResearchIntent.GENE
    return ResearchIntent.LITERATURE


def requested_sources(text: str) -> list[str]:
    return [name for name, pat in SOURCE_ALIASES if pat.search(text)]


def requested_limit(text: str) -> int | None:
    m = _LIMIT.search(text)
    if not m:
        return None
    n = int(next(g for g in m.groups() if g))
    return n if 1 <= n <= 50 else None


def clean_query(text: str) -> str:
    q = _OUTPUT_CLAUSE.sub("", text)
    q = _SOURCE_PHRASE.sub(" ", q)
    q = _LIMIT.sub(" ", q)
    q = _FILLER.sub(" ", q)
    q = re.sub(r"^(?:\s*\b(?:and|or|for|on|about|in|of|the|an?|entry|record|və|ve|ilə|ile)\b\s*)+", "",
               q.strip(), flags=re.I)
    q = re.sub(r"[?!.,;:\"“”«»]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q or text.strip()


class QueryPlanner:
    # Primary sources are always queried; supplementary ones only if the primaries return
    # fewer than ResearchEngine.min_primary_results records.
    SOURCES = {
        ResearchIntent.LITERATURE: ["pubmed", "europepmc"],
        ResearchIntent.VARIANT: ["clinvar", "gnomad", "ensembl", "pubmed"],
        ResearchIntent.GENE: ["ncbi_gene", "ensembl", "pubmed"],
    }
    SUPPLEMENTARY = {
        ResearchIntent.LITERATURE: ["semantic_scholar", "crossref"],
        ResearchIntent.VARIANT: [],
        ResearchIntent.GENE: [],
    }

    def __init__(self, llm: LLMService | None = None):
        self.llm = llm

    async def plan(self, text: str, language: str = "en",
                   sources: list[str] | None = None) -> ResearchPlan:
        ids = extract_identifiers(text)
        intent = classify_intent(text, ids)
        recent = bool(_RECENT.search(text))
        query = clean_query(text)
        if intent == ResearchIntent.VARIANT and ids.get("rsid"):
            query = ids["rsid"] + (f" {ids['gene']}" if ids.get("gene") else "")
        elif intent == ResearchIntent.GENE and ids.get("gene"):
            query = ids["gene"]
        elif language != "en" and self.llm and self.llm.available:
            query = await self._translate(query) or query
        named = requested_sources(text)
        if sources:
            chosen, supplementary, explicit = sources, [], False
        elif named:
            chosen, supplementary, explicit = named, [], True
        else:
            chosen, supplementary, explicit = self.SOURCES[intent], self.SUPPLEMENTARY[intent], False
        return ResearchPlan(intent=intent, query=query, original=text, sources=chosen,
                            supplementary=supplementary, explicit_sources=explicit,
                            limit=requested_limit(text), recent=recent,
                            min_year=date.today().year - 3 if recent else None, identifiers=ids)

    async def _translate(self, query: str) -> str | None:
        """Tier-1 (cheap model) rewrite of a non-English query into English keywords."""
        try:
            out = await self.llm.structured(CompletionRequest(
                system="Rewrite the user's research topic as a concise English literature-search "
                       "query (scientific terms, no filler words).",
                messages=[ChatMessage(role="user", text=query)], max_tokens=200),
                RewrittenQuery, tier="fast", purpose="research_query_rewrite")
            return out.query.strip() or None
        except Exception as e:  # rewriting is an optimisation; never fatal
            log.warning("research.rewrite_failed", extra={"error": str(e)})
            return None
