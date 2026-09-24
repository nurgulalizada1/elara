import httpx
import pytest

from elara.providers import LLMService
from elara.research import EvidenceType, QueryPlanner, ResearchEngine, ResearchIntent, Synthesizer
from elara.research.http import ResearchHttp
from elara.research.planner import clean_query, extract_identifiers
from elara.research.sources import build_sources
from elara.research.synthesis import validate_citations
from tests import research_fixtures as fx
from tests.fakes import ScriptedProvider, text


def make_engine(settings, db=None, overrides=None, calls=None, llm=None, min_primary=3):
    client = httpx.AsyncClient(transport=httpx.MockTransport(fx.handler(overrides, calls)))
    http = ResearchHttp(client, db, backoff_s=0, max_retries=1)
    return ResearchEngine(build_sources(http, settings), QueryPlanner(llm), db,
                          min_primary_results=min_primary)


class TestPlanner:
    async def test_literature(self):
        plan = await QueryPlanner().plan("Find recent research about single-cell RNA sequencing")
        assert plan.intent == ResearchIntent.LITERATURE
        assert plan.recent and plan.min_year
        assert plan.query == "single-cell RNA sequencing"
        assert plan.sources[0] == "pubmed"

    async def test_variant_and_gene(self):
        plan = await QueryPlanner().plan("Is rs80357906 in BRCA1 pathogenic?")
        assert plan.intent == ResearchIntent.VARIANT and plan.identifiers["rsid"] == "rs80357906"
        assert "clinvar" in plan.sources and "gnomad" in plan.sources
        plan = await QueryPlanner().plan("BRCA1 gene")
        assert plan.intent == ResearchIntent.GENE and plan.query == "BRCA1"

    def test_identifiers(self):
        ids = extract_identifiers("chr17:43057062:T>TG and NM_007294.4:c.68_69del")
        assert ids["variant_id"] == "17-43057062-T-TG"
        assert ids["hgvs"].startswith("NM_007294.4:c.")
        assert "gene" not in extract_identifiers("CRISPR and PCR in DNA")

    def test_clean_query_multilingual(self):
        assert clean_query("CRISPR haqqında son tədqiqatları tap") == "CRISPR"
        assert clean_query("CRISPR hakkında makaleleri bul") == "CRISPR"

    async def test_non_english_query_rewritten_by_cheap_model(self, settings):
        settings = settings.model_copy(update={"model_fast": "fast"})
        prov = ScriptedProvider([text('{"query": "single-cell RNA sequencing"}')])
        planner = QueryPlanner(LLMService(prov, settings))
        plan = await planner.plan("tək hüceyrəli RNT sekvenləşdirmə haqqında məqalələr", "az")
        assert plan.query == "single-cell RNA sequencing"
        assert prov.requests[0].model == "fast"


class TestEngine:
    async def test_literature_search_dedupes_and_labels(self, settings, db):
        engine = make_engine(settings, db, min_primary=99)  # force supplementary sources
        res = await engine.research("single-cell RNA sequencing review")
        titles = [r.title for r in res.records]
        # The review appears in pubmed, europepmc and semantic scholar -> merged once.
        assert sum("review of methods" in t for t in titles) == 1
        review = next(r for r in res.records if "review of methods" in r.title)
        assert review.evidence_type == EvidenceType.REVIEW
        assert set(review.also_in) >= {"europepmc", "semantic_scholar"}
        preprint = next(r for r in res.records if "clustering" in r.title)
        assert preprint.evidence_type == EvidenceType.PREPRINT
        assert all(o.ok for o in res.outcomes)
        with db.connect() as c:
            assert c.execute("SELECT count(*) FROM sources").fetchone()[0] == len(res.records)
            assert c.execute("SELECT result_count FROM research_queries").fetchone()[0] == len(
                res.records)

    async def test_variant_search_uses_databases(self, settings):
        engine = make_engine(settings)
        res = await engine.research("rs80357906 BRCA1 variant")
        sources = {r.source for r in res.records}
        assert {"clinvar", "gnomad", "ensembl"} <= sources
        clin = next(r for r in res.records if r.source == "clinvar")
        assert clin.extra["significance"] == "Pathogenic"
        assert res.records[0].evidence_type == EvidenceType.DATABASE_RECORD
        gn = next(r for r in res.records if r.source == "gnomad")
        assert gn.extra["allele_frequency"] == pytest.approx(3 / 1461000)

    async def test_gene_search(self, settings):
        res = await make_engine(settings).research("BRCA1 gene")
        assert {"ncbi_gene", "ensembl"} <= {r.source for r in res.records}

    async def test_source_failure_is_isolated_and_reported(self, settings):
        engine = make_engine(settings, overrides={
            "semanticscholar": httpx.Response(429, json={"message": "Too Many Requests"}),
            "crossref": httpx.ConnectError("down")}, min_primary=99)
        res = await engine.research("single-cell RNA sequencing")
        assert set(res.failed_sources) == {"semantic_scholar", "crossref"}
        assert res.records  # other sources still delivered

    async def test_malformed_response_does_not_crash(self, settings):
        engine = make_engine(settings, overrides={"europepmc": httpx.Response(200, json=[1, 2])})
        res = await engine.research("single-cell RNA sequencing")
        assert "europepmc" in res.failed_sources

    async def test_cache_avoids_repeat_calls(self, settings, db):
        calls: list = []
        engine = make_engine(settings, db, calls=calls)
        await engine.research("BRCA1 gene")
        n = len(calls)
        await engine.research("BRCA1 gene")
        assert len(calls) == n

    async def test_retry_then_success(self, settings):
        state = {"n": 0}

        def flaky(req):
            state["n"] += 1
            if state["n"] == 1:
                return httpx.Response(503)
            return httpx.Response(200, json=fx.ENSEMBL_GENE)

        client = httpx.AsyncClient(transport=httpx.MockTransport(flaky))
        http = ResearchHttp(client, backoff_s=0, max_retries=2)
        data = await http.get_json("https://rest.ensembl.org/lookup/symbol/homo_sapiens/BRCA1",
                                   source="ensembl")
        assert data["id"] == "ENSG00000012048" and state["n"] == 2


class TestSynthesis:
    def test_validate_citations(self):
        txt = "Claim [1]. Other [7]. Range [1, 9]. DOI 10.1234/fake.99 and PMID: 99999999."
        out, removed = validate_citations(txt, 2, {"10.1038/x"})
        assert "[7]" not in out and "[1]" in out and "10.1234/fake" not in out
        assert "99999999" not in out and len(removed) == 4

    async def test_llm_synthesis_cannot_fabricate(self, settings):
        settings = settings.model_copy(update={"model_strong": "strong"})
        res = await make_engine(settings).research("single-cell RNA sequencing")
        prov = ScriptedProvider([text("scRNA-seq is reviewed in [1] and invented in [42] "
                                      "(doi 10.9999/made.up).")])
        syn = await Synthesizer(LLMService(prov, settings)).synthesize("q", res, "en")
        assert syn.used_llm and "[42]" not in syn.text and "10.9999" not in syn.text
        assert "[1] Single-cell RNA sequencing" in syn.text  # generated reference list
        assert "untrusted_content" in prov.requests[0].messages[0].text

    async def test_no_results_says_so(self, settings):
        engine = make_engine(settings, overrides={
            "esearch": httpx.Response(200, json={"esearchresult": {"idlist": []}}),
            "europepmc": httpx.Response(200, json={"resultList": {"result": []}}),
            "crossref": httpx.Response(200, json={"message": {"items": []}}),
            "semanticscholar": httpx.Response(200, json={"data": []})})
        res = await engine.research("xyzzy nonexistent topic")
        syn = await Synthesizer(None).synthesize("q", res, "en")
        assert "found no results" in syn.text and not syn.used_llm

    async def test_all_sources_down_is_not_reported_as_no_results(self, settings):
        import httpx as _h
        engine = make_engine(settings, overrides={"": _h.ConnectError("blocked")})
        res = await engine.research("single-cell RNA sequencing")
        syn = await Synthesizer(None).synthesize("q", res, "en")
        assert "couldn't reach any research source" in syn.text and "no results" not in syn.text

    async def test_fallback_without_llm_lists_sources(self, settings):
        res = await make_engine(settings).research("single-cell RNA sequencing")
        syn = await Synthesizer(LLMService(None, settings)).synthesize("q", res, "az")
        assert "[1]" in syn.text and "Dil modeli olmadığı" in syn.text


async def test_research_tools(settings, db):
    from elara.tools.base import Origin, ToolContext
    from elara.tools.builtin.research_tools import ResearchSearchTool, SourceSearchTool
    engine = make_engine(settings)
    out = await ResearchSearchTool(engine).run(
        ResearchSearchTool.Input(query="rs80357906"), ToolContext(origin=Origin.LLM))
    assert out.intent == "variant" and out.results
    clinvar = SourceSearchTool(engine, engine.sources["clinvar"])
    assert clinvar.name == "clinvar_search" and not clinvar.exposed_to_llm
    out = await clinvar.run(clinvar.Input(query="BRCA1 c.68_69del"), ToolContext(origin=Origin.API))
    assert out.results[0].source == "clinvar"


class TestExplicitSources:
    @pytest.mark.parametrize("text,sources,query,limit", [
        ("Search PubMed for BRCA1 breast cancer and give me the first 3 results with their "
         "titles and PubMed IDs.", ["pubmed"], "BRCA1 breast cancer", 3),
        ("Find the NCBI Gene entry for TP53.", ["ncbi_gene"], "TP53", None),
        ("Search Europe PMC and Crossref for CRISPR base editing", ["europepmc", "crossref"],
         "CRISPR base editing", None),
        ("top 10 papers on Alzheimer amyloid in PubMed", ["pubmed"], "Alzheimer amyloid", 10),
        ("PubMed-də CRISPR haqqında ilk 5 məqaləni tap", ["pubmed"], "CRISPR", 5),
    ])
    async def test_plan(self, text, sources, query, limit):
        plan = await QueryPlanner().plan(text)
        assert plan.explicit_sources and plan.sources == sources
        assert plan.query == query and plan.limit == limit and plan.supplementary == []

    async def test_default_literature_plan_is_tiered(self):
        plan = await QueryPlanner().plan("Find recent papers about single-cell RNA sequencing.")
        assert not plan.explicit_sources and plan.sources == ["pubmed", "europepmc"]
        assert plan.supplementary == ["semantic_scholar", "crossref"]

    def test_source_names_are_not_gene_symbols(self):
        for word in ("NCBI", "PMID", "DOI", "EPMC"):
            assert extract_identifiers(f"Find the {word} entry for TP53")["gene"] == "TP53"
