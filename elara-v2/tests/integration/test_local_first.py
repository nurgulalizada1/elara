"""Regression tests for the real-machine validation findings (local-first, token-minimizing).

The ScriptedProvider starts with an empty script: any unexpected LLM call is recorded in
provider.requests (and would fail), so `provider.requests == []` proves no tokens were spent.
"""

import httpx
import pytest
from pydantic import SecretStr

from elara.core.container import build_container
from elara.core.errors import PathNotAllowed
from elara.memory import MemorySource
from elara.research.http import ResearchHttp
from elara.security.paths import PathGuard
from elara.security.untrusted import Trust
from tests import research_fixtures as fx
from tests.fakes import ScriptedProvider, text


def remember(app, content):
    out = app.memory.remember(content, source=MemorySource.USER_EXPLICIT, trust=Trust.USER)
    assert out.saved, out.reason


# ---------------------------------------------------------------- 1. memory without LLM --
@pytest.mark.parametrize("stored,question", [
    ("my permanent test preference is Azerbaijani answers.",
     "What language do I prefer for answers?"),
    ("user prefers Azerbaijani answers", "What language do I prefer?"),
    ("my favorite programming language is Python", "What is my favorite programming language?"),
    ("mən astrobiologiya ilə maraqlanıram", "Mən nə ilə maraqlanıram?"),
    ("Mən Azərbaycan dilində cavab almağa üstünlük verirəm",
     "Cavabları hansı dildə almağa üstünlük verirəm?"),
])
async def test_memory_answers_locally(app, provider, stored, question):
    remember(app, stored)
    r = await app.assistant.handle(question)  # new conversation
    assert r.tier == 0 and not r.used_llm and r.resolver == "memory", r
    assert stored.rstrip(".") in r.text
    assert provider.requests == []


async def test_explicit_store_then_local_retrieval_in_new_process_like_flow(app, provider):
    r1 = await app.assistant.handle("Remember that my permanent test preference is "
                                    "Azerbaijani answers.")
    assert r1.memory_events and not r1.used_llm
    r2 = await app.assistant.handle("What language do I prefer for answers?")
    assert r2.conversation_id != r1.conversation_id
    assert (r2.tier, r2.used_llm, r2.resolver) == (0, False, "memory")
    assert "Azerbaijani" in r2.text and provider.requests == []


@pytest.mark.parametrize("question", ["How do I install Python on Ubuntu?",
                                      "What is my blood type?"])
async def test_unrelated_questions_are_not_answered_from_memory(app, provider, question):
    remember(app, "my favorite programming language is Python")
    provider.push(text("LLM answer"))
    r = await app.assistant.handle(question)
    assert r.used_llm and r.resolver == "llm" and r.text == "LLM answer"


async def test_memory_query_without_llm_says_not_found(settings):
    c = build_container(settings, use_env_provider=False)
    r = await c.assistant.handle("What is my blood type?")
    assert r.tier == 0 and not r.used_llm and "anything saved" in r.text
    await c.aclose()


# -------------------------------------------- 2. conversation context is not persisted --
async def test_conversation_context_answers_locally_but_is_not_persisted(app, provider):
    provider.push(text("Noted!"))
    r1 = await app.assistant.handle("My temporary test name is ELARA_VALIDATION.")
    assert r1.memory_events == [] and app.memory.store.list() == []
    # same conversation: short-term context answers it, without the LLM
    r2 = await app.assistant.handle("What is my temporary test name?", r1.conversation_id)
    assert (r2.tier, r2.used_llm, r2.resolver) == (0, False, "conversation")
    assert "ELARA_VALIDATION" in r2.text and len(provider.requests) == 1
    # new conversation: not known locally -> falls through (LLM has no such memory)
    provider.push(text("I don't know your temporary test name."))
    r3 = await app.assistant.handle("What is my temporary test name?")
    assert r3.resolver == "llm" and "ELARA_VALIDATION" not in provider.requests[-1].system
    assert "ELARA_VALIDATION" not in r3.text


@pytest.mark.parametrize("statement", ["My exam is tomorrow", "I prefer tea over coffee",
                                       "My name is Aysel", "Mənim adım Nurgüldür",
                                       "Sabah imtahanım var"])
async def test_ordinary_statements_never_create_memory(app, provider, statement):
    provider.push(text("ok"))
    r = await app.assistant.handle(statement)
    assert r.memory_events == [] and app.memory.store.list() == []


@pytest.mark.parametrize("command", ["Remember that I prefer tea over coffee",
                                     "Yadda saxla ki, mən çayı qəhvədən üstün tuturam",
                                     "Hatırla ki çayı severim",
                                     "Save to memory: I prefer tea over coffee"])
async def test_explicit_memory_commands_still_store(app, provider, command):
    r = await app.assistant.handle(command)
    assert r.memory_events and len(app.memory.store.list()) == 1 and not r.used_llm


# ------------------------------------------------------------------ 3. named paths ------
@pytest.fixture
def named_env(tmp_path):
    ws = tmp_path / "ws"
    proj = ws / "realproject"
    (proj / "src").mkdir(parents=True)
    (proj / "readme.txt").write_text("project readme")
    (ws / "project").mkdir()  # decoy: a relative dir with the same name
    (ws / "project" / "decoy.txt").write_text("decoy")
    outside = tmp_path / "outside"
    outside.mkdir()
    guard = PathGuard([ws], [ws], home=tmp_path,
                      named_paths={"project": proj, "secret": outside})
    return guard, ws, proj, outside


def test_named_path_resolution(named_env):
    guard, ws, proj, outside = named_env
    assert guard.resolve("project") == proj
    assert guard.resolve("Project") == proj
    assert guard.resolve("my project folder") == proj
    assert guard.resolve("project/readme.txt") == proj / "readme.txt"
    assert guard.resolve("project/src") == proj / "src"
    assert guard.resolve("workspace") == ws
    assert guard.resolve(str(ws / "project" / "decoy.txt")) == ws / "project" / "decoy.txt"
    assert guard.resolve("notes.txt") == ws / "notes.txt"  # plain relative unchanged
    assert guard.resolve("./project") == ws / "project"    # explicit relative unchanged


@pytest.mark.parametrize("raw", ["project/../../outside", "project/../../../etc/passwd",
                                 "secret", "secret/x.txt", "/etc/hostname", "/etc"])
def test_named_paths_keep_the_jail(named_env, raw):
    with pytest.raises(PathNotAllowed):
        named_env[0].resolve(raw)


async def test_list_files_in_named_project(settings, tmp_path):
    ws = settings.write_dirs[0]
    proj = ws / "validation-project"
    proj.mkdir()
    (proj / "inside.txt").write_text("x")
    (ws / "project").mkdir()
    (ws / "project" / "decoy.txt").write_text("x")
    c = build_container(settings.model_copy(update={"named_paths": {"project": proj}}),
                        use_env_provider=False)
    for msg in ("List the files in project.", "list files in my project folder",
                "Show files in project"):
        r = await c.assistant.handle(msg)
        assert "inside.txt" in r.text and "decoy.txt" not in r.text, (msg, r.text)
        assert not r.used_llm and r.tier == 0
    r = await c.assistant.handle("Read project/inside.txt")
    assert r.tool_calls[0].status == "ok" and not r.used_llm
    await c.aclose()


# -------------------------------------------------------------- 8. security preserved --
async def test_security_protections_still_hold(settings):
    c = build_container(settings, use_env_provider=False)
    ws = settings.write_dirs[0]
    (ws / "keep.txt").write_text("original")
    ex, ctx = c.executor, __import__("elara.tools.base", fromlist=["ToolContext"])
    user = ctx.ToolContext(origin=ctx.Origin.API)
    assert (await ex.execute("read_file", {"path": "/etc/hostname"}, user)).status == "denied"
    assert (await ex.execute("list_files", {"path": "/etc"}, user)).status == "denied"
    assert (await ex.execute("read_file", {"path": "../../../../etc/passwd"}, user)).status == \
        "denied"
    assert (await ex.execute("write_file", {"path": "/tmp/elara-x.txt", "content": "x"},
                             user)).status == "denied"
    out = await ex.execute("write_file", {"path": "keep.txt", "content": "new",
                                          "mode": "overwrite"}, user)
    assert out.status == "needs_confirmation" and (ws / "keep.txt").read_text() == "original"
    missing = await ex.execute("read_file", {"path": "nope.txt"}, user)
    assert missing.status == "error" and "not a file" in missing.error
    missing_dir = await ex.execute("list_files", {"path": "no-such-dir"}, user)
    assert missing_dir.status == "error"
    py = await ex.execute("run_python", {"code": "print(1)"}, ctx.ToolContext(
        origin=ctx.Origin.API, confirmed=True))
    assert py.status == "denied" and "disabled" in py.error
    await c.aclose()


# ------------------------------------------------------------ 4. PubMed direct search --
PUBMED_Q = ("Search PubMed for BRCA1 breast cancer and give me the first 3 results with "
            "their titles and PubMed IDs.")


def research_app(settings, overrides=None, calls=None, provider=None, **settings_update):
    rc = httpx.AsyncClient(transport=httpx.MockTransport(fx.handler(overrides, calls)))
    s = settings.model_copy(update={"model_fast": "f", "model_strong": "s", **settings_update})
    return build_container(s, provider=provider or ScriptedProvider(), research_client=rc)


async def test_pubmed_explicit_search_is_direct_ordered_and_llm_free(settings):
    calls: list[str] = []
    ranked = {"esearchresult": {"idlist": ["38000002", "38000001"]}}  # PubMed's own order
    summary = {"result": {**fx.PUBMED_ESUMMARY["result"], "uids": ["38000002", "38000001"]}}
    c = research_app(settings, calls=calls, overrides={
        "esearch": httpx.Response(200, json=ranked),
        "esummary": httpx.Response(200, json=summary)})
    r = await c.assistant.handle(PUBMED_Q)
    assert not r.used_llm and r.resolver == "research" and c.llm.provider.requests == []
    hosts = {httpx.URL(u).host for u in calls}
    assert hosts == {"eutils.ncbi.nlm.nih.gov"}  # no Europe PMC / Crossref / S2 substitution
    esearch = next(u for u in calls if "esearch" in u)
    assert "retmax=3" in esearch and "term=BRCA1+breast+cancer" in esearch
    lines = [ln for ln in r.text.splitlines() if ln[:2] in ("1.", "2.")]
    assert "PMID: 38000002" in lines[0] and "PMID: 38000001" in lines[1]  # order preserved
    assert "PubMed results" in r.text
    assert [s["pmid"] for s in r.sources] == ["38000002", "38000001"]


async def test_pubmed_unavailable_is_reported_and_fallback_is_labelled(settings):
    c = research_app(settings, overrides={"eutils": httpx.ConnectError("blocked")})
    r = await c.assistant.handle(PUBMED_Q)
    assert "PubMed is unavailable" in r.text and "network unreachable" in r.text
    assert "not presenting other sources' results as PubMed" in r.text
    assert "NOT from PubMed" in r.text and "(Europe PMC)" in r.text  # clearly labelled
    assert "PubMed results" not in r.text and not r.used_llm


async def test_pubmed_empty_result_is_not_padded(settings):
    c = research_app(settings, overrides={
        "esearch": httpx.Response(200, json={"esearchresult": {"idlist": []}})})
    r = await c.assistant.handle(PUBMED_Q)
    assert "PubMed returned no results" in r.text and "PMID" not in r.text


# -------------------------------------------------------------------- 5. NCBI Gene ------
@pytest.mark.parametrize("failure,reason", [
    (httpx.ConnectError("proxy denied"), "network unreachable or blocked"),
    (httpx.Response(403, text="Forbidden"), "access denied"),
])
async def test_ncbi_gene_unavailable_is_not_reported_as_success(settings, failure, reason):
    c = research_app(settings, overrides={"eutils": failure})
    r = await c.assistant.handle("Find the NCBI Gene entry for TP53.")
    assert "NCBI Gene is unavailable" in r.text and reason in r.text
    assert "7157" not in r.text and not r.used_llm
    # a live result from another source is shown only with an explicit label
    assert "NOT from NCBI Gene" in r.text and "(Ensembl)" in r.text


async def test_ncbi_gene_live_success(settings):
    c = research_app(settings)
    r = await c.assistant.handle("Find the NCBI Gene entry for BRCA1.")
    assert "NCBI Gene results" in r.text and "ncbi.nlm.nih.gov/gene/672" in r.text
    assert "unavailable" not in r.text


# ------------------------------------------------------------- 6. Semantic Scholar ------
async def test_semantic_scholar_skipped_when_primary_sources_suffice(settings):
    calls: list[str] = []
    many = {"esearchresult": {"idlist": ["1", "2", "3"]}}
    summ = {"result": {"uids": ["1", "2", "3"], **{u: {"uid": u, "title": f"Paper {u}",
                                                       "pubdate": "2024", "pubtype": []}
                                                   for u in "123"}}}
    c = research_app(settings, calls=calls, overrides={
        "esearch": httpx.Response(200, json=many), "esummary": httpx.Response(200, json=summ)})
    await c.research.research("single-cell RNA sequencing")
    assert not any("semanticscholar" in u or "crossref" in u for u in calls)


async def test_semantic_scholar_rate_limit_cooldown_and_no_fabrication(settings):
    calls: list[str] = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(fx.handler(
        {"semanticscholar": httpx.Response(429, headers={"retry-after": "120"})}, calls)))
    http = ResearchHttp(client, None, backoff_s=0, max_retries=2)
    from elara.research import QueryPlanner, ResearchEngine
    from elara.research.sources import build_sources
    engine = ResearchEngine(build_sources(http, settings), QueryPlanner())
    plan = await engine.planner.plan("Search Semantic Scholar for single-cell RNA sequencing")
    res = await engine.run(plan)
    assert res.records == [] and res.outcomes[0].error_kind == "rate_limited"
    first = sum("semanticscholar" in u for u in calls)
    assert first == 1  # long Retry-After: not retried in a loop
    res2 = await engine.run(plan)
    assert sum("semanticscholar" in u for u in calls) == first  # cooling down: no new call
    assert "rate limited" in res2.outcomes[0].error


async def test_semantic_scholar_rate_limit_message(settings):
    c = research_app(settings, overrides={"semanticscholar": httpx.Response(429)})
    r = await c.assistant.handle("Search Semantic Scholar for CRISPR base editing")
    assert "Semantic Scholar is unavailable right now (rate limited)" in r.text
    assert "NOT from Semantic Scholar" in r.text  # anything else shown is labelled


async def test_semantic_scholar_api_key_is_sent(settings):
    seen = {}

    def handler(req):
        seen["key"] = req.headers.get("x-api-key")
        return httpx.Response(200, json=fx.S2)

    rc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    s = settings.model_copy(update={"semantic_scholar_api_key": SecretStr("s2-test-key")})
    c = build_container(s, research_client=rc, use_env_provider=False)
    await c.research.run(await c.research.planner.plan("Search Semantic Scholar for CRISPR"))
    assert seen["key"] == "s2-test-key"


def test_semantic_scholar_key_env_aliases(monkeypatch, tmp_path):
    from elara.config.settings import Settings
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "abc123")
    assert Settings(data_dir=tmp_path).semantic_scholar_api_key.get_secret_value() == "abc123"


# ------------------------------------------------------------------- 7. routing ---------
@pytest.mark.parametrize("message", [
    "What's 17 * 42?", "saat neçədir?", "Salam ELARA", "What can you do?",
    "Remember that I like jazz", "Mənim haqqımda nə bilirsən?", "list files",
    "Read notes.txt", "Read /etc/hostname", "show ~/.ssh/id_rsa", "Thanks!", "Search PubMed for BRCA1 breast cancer",
    "Find the NCBI Gene entry for BRCA1.", "rs80357906 in ClinVar",
])
async def test_local_capabilities_never_call_the_llm(settings, message):
    (settings.write_dirs[0] / "notes.txt").write_text("hello")
    c = research_app(settings)
    r = await c.assistant.handle(message)
    assert not r.used_llm and c.llm.provider.requests == [], (message, r.resolver, r.text)


async def test_cached_research_is_labelled_as_cached(settings, db):
    c = research_app(settings)
    await c.assistant.handle("Find the NCBI Gene entry for BRCA1.")
    r = await c.assistant.handle("Find the NCBI Gene entry for BRCA1.")
    assert "from local cache" in r.text


@pytest.mark.parametrize("message", ["Read /etc/hostname", "Read ../../etc/passwd",
                                     "show ~/.ssh/id_rsa"])
async def test_reads_outside_the_jail_are_denied_locally(settings, message):
    c = research_app(settings)
    r = await c.assistant.handle(message)
    assert r.tool_calls[0].tool == "read_file" and r.tool_calls[0].status == "denied"
    assert not r.used_llm and c.llm.provider.requests == []


def test_preference_statement_is_classified_as_preference(app):
    remember(app, "my permanent test preference is Azerbaijani answers.")
    assert app.memory.store.list()[0].kind == "preference"
