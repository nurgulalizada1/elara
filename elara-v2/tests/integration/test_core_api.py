"""The application-independent Core API (ElaraCore.process), used by every frontend."""

import json

import httpx

from elara.cli import main as cli
from elara.core.service import CoreRequest, CoreResult, ElaraCore
from tests import research_fixtures as fx
from tests.fakes import ScriptedProvider, text


def open_core(settings, provider=None, research_client=None) -> ElaraCore:
    s = settings.model_copy(update={"model_fast": "f", "model_strong": "s"})
    return ElaraCore.open(s, provider=provider or ScriptedProvider(),
                          research_client=research_client)


async def test_core_processes_text_without_cli(settings):
    async with open_core(settings) as core:
        r = await core.process(CoreRequest(text="Salam ELARA", channel="test"))
    assert isinstance(r, CoreResult) and r.text == "Salam! Necəsən?"
    assert r.channel == "test" and r.request_id and r.conversation_id
    assert r.build["version"] and r.duration_ms >= 0 and r.intent == "greeting"


async def test_calculator_through_core(settings):
    async with open_core(settings) as core:
        r = await core.process(CoreRequest(text="What's 17 * 42?"))
    assert (r.text, r.tier, r.used_llm, r.resolver) == ("17 * 42 = 714", 0, False, "deterministic")


async def test_memory_query_through_core(settings):
    async with open_core(settings) as core:
        await core.process(CoreRequest(text="Remember that my favorite language is Python."))
        r = await core.process(CoreRequest(text="What is my favorite programming language?"))
        assert core.container.llm.provider.requests == []
    assert (r.tier, r.used_llm, r.resolver) == (0, False, "memory") and "Python" in r.text


async def test_pubmed_explicit_source_through_core(settings):
    rc = httpx.AsyncClient(transport=httpx.MockTransport(fx.handler()))
    async with open_core(settings, research_client=rc) as core:
        r = await core.process(CoreRequest(text="Search PubMed for BRCA1 breast cancer and give "
                                                "me the first 3 results with their PubMed IDs."))
        assert core.container.llm.provider.requests == []
    assert (r.resolver, r.used_llm, r.tier) == ("research", False, 3)
    assert "PubMed results" in r.text and [s["source"] for s in r.sources] == ["pubmed"] * 2


async def test_llm_reasoning_through_core(settings):
    provider = ScriptedProvider([text("TCP is reliable; UDP is not.")])
    async with open_core(settings, provider=provider) as core:
        r = await core.process(CoreRequest(text="Compare TCP and UDP in one sentence."))
    assert r.used_llm and r.resolver == "llm" and r.tier == 2
    assert r.text == "TCP is reliable; UDP is not." and provider.requests[0].model == "s"


def test_cli_ask_is_an_adapter_over_core(settings, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_settings", lambda: settings)
    assert cli.main(["ask", "What's 17 * 42?"]) == 0
    assert capsys.readouterr().out.strip() == "17 * 42 = 714"
    assert cli.main(["ask", "--json", "What's 17 * 42?"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["text"] == "17 * 42 = 714" and payload["channel"] == "cli"
    assert payload["resolver"] == "deterministic" and payload["build"]["version"]
    assert set(CoreResult.model_fields) <= set(payload)
