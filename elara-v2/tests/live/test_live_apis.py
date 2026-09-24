"""Real network calls to the research APIs and (if a key is set) the LLM provider.

Run on your machine:  ELARA_LIVE_TESTS=1 pytest tests/live -v
These verify that the recorded response shapes used in unit tests still match reality.
"""

import os

import httpx
import pytest

from elara.config.settings import Settings
from elara.research import EvidenceType, QueryPlanner, ResearchEngine
from elara.research.http import ResearchHttp
from elara.research.sources import build_sources

pytestmark = pytest.mark.live


@pytest.fixture
async def engine(tmp_path):
    async with httpx.AsyncClient() as client:
        settings = Settings(data_dir=tmp_path, provider="none")
        http = ResearchHttp(client, None, min_intervals={"eutils.ncbi.nlm.nih.gov": 0.4})
        yield ResearchEngine(build_sources(http, settings), QueryPlanner())


async def test_literature_sources(engine):
    res = await engine.research("single-cell RNA sequencing review")
    assert res.records, res.outcomes
    ok = {o.source for o in res.outcomes if o.ok}
    assert {"pubmed", "europepmc"} <= ok, res.outcomes
    assert any(r.evidence_type == EvidenceType.REVIEW for r in res.records)


async def test_variant_sources(engine):
    res = await engine.research("rs80357906 BRCA1 variant")
    ok = {o.source for o in res.outcomes if o.ok}
    assert {"clinvar", "ensembl"} <= ok, res.outcomes
    assert "gnomad" in ok, [o for o in res.outcomes if o.source == "gnomad"]


async def test_gene_sources(engine):
    res = await engine.research("BRCA1 gene")
    assert {"ncbi_gene", "ensembl"} <= {r.source for r in res.records}


@pytest.mark.skipif(not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")),
                    reason="no LLM API key")
async def test_llm_provider_roundtrip(tmp_path):
    from elara.core.container import build_container
    c = build_container(Settings(data_dir=tmp_path, read_dirs=[tmp_path], write_dirs=[tmp_path]))
    try:
        assert await c.llm.provider.check()
        r = await c.assistant.handle("Use the calculator to compute 123.5 * 7, then answer "
                                     "in one short sentence.")
        assert r.used_llm and "864.5" in r.text, r
        assert any(t.tool == "calculator" for t in r.tool_calls)
    finally:
        await c.aclose()
