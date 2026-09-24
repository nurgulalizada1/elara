import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from elara.api.app import create_app
from elara.core.container import build_container
from tests import research_fixtures as fx
from tests.fakes import ScriptedProvider, text


@pytest.fixture
def client(settings):
    rc = httpx.AsyncClient(transport=httpx.MockTransport(fx.handler()))
    settings = settings.model_copy(update={"model_fast": "f", "model_strong": "s"})
    prov = ScriptedProvider()
    c = build_container(settings, provider=prov, research_client=rc)
    with TestClient(create_app(container=c)) as tc:
        tc.provider = prov
        tc.container = c
        yield tc


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["database"] is True
    assert r.headers["x-request-id"].startswith("req_")


def test_chat_and_context(client):
    r = client.post("/chat", json={"message": "What's 17 * 42?"})
    body = r.json()
    assert r.status_code == 200 and body["text"] == "17 * 42 = 714" and body["tier"] == 0
    client.provider.push(text("Hello there"))
    r2 = client.post("/chat", json={"message": "Tell me something nice",
                                    "conversation_id": body["conversation_id"]})
    assert r2.json()["text"] == "Hello there" and r2.json()["used_llm"]
    msgs = client.get(f"/conversations/{body['conversation_id']}/messages").json()
    assert len(msgs) == 4


def test_memory_endpoints(client):
    r = client.post("/memory/store", json={"content": "I prefer concise answers"})
    assert r.status_code == 200 and r.json()["action"] == "created"
    mid = r.json()["memory"]["id"]
    assert client.post("/memory/search", json={"query": "concise"}).json()[0]["id"] == mid
    assert client.patch(f"/memory/{mid}", json={"content": "I prefer detailed answers"}
                        ).json()["content"] == "I prefer detailed answers"
    assert client.delete(f"/memory/{mid}").status_code == 200
    assert client.delete(f"/memory/{mid}").status_code == 404
    bad = client.post("/memory/store", json={"content": "my password is hunter2"})
    assert bad.status_code == 422 and "never store" in bad.json()["error"]["message"]


def test_tools_endpoints(client, settings):
    tools = client.get("/tools").json()
    names = {t["name"] for t in tools}
    assert {"calculator", "read_file", "research_search", "pubmed_search"} <= names
    r = client.post("/tools/calculator", json={"arguments": {"expression": "2**10"}})
    assert r.status_code == 200 and r.json()["output"]["result"] == 1024
    (settings.write_dirs[0] / "x.txt").write_text("x")
    r = client.post("/tools/delete_file", json={"arguments": {"path": "x.txt"}})
    assert r.status_code == 409 and "confirmed" in r.json()["confirmation"]["how"]
    assert (settings.write_dirs[0] / "x.txt").exists()
    r = client.post("/tools/delete_file", json={"arguments": {"path": "x.txt"},
                                                "confirmed": True})
    assert r.status_code == 200 and not (settings.write_dirs[0] / "x.txt").exists()
    assert client.post("/tools/nope", json={}).status_code == 404
    assert client.post("/tools/calculator", json={"arguments": {}}).status_code == 422
    assert client.post("/tools/run_python", json={"arguments": {"code": "1"},
                                                  "confirmed": True}).status_code == 403


def test_confirmation_endpoint(client, settings):
    (settings.write_dirs[0] / "y.txt").write_text("y")
    from tests.fakes import tool
    client.provider.push(tool("delete_file", {"path": "y.txt"}))
    r = client.post("/chat", json={"message": "delete y.txt please"}).json()
    action = r["pending_action"]["id"]
    done = client.post(f"/confirmations/{action}", json={"approve": True}).json()
    assert done["tool_calls"][0]["status"] == "ok"
    assert not (settings.write_dirs[0] / "y.txt").exists()


def test_research_endpoint(client):
    r = client.post("/research", json={"query": "BRCA1 gene", "synthesize": False})
    assert r.status_code == 200
    assert {x["source"] for x in r.json()["records"]} >= {"ncbi_gene", "ensembl"}


def test_config_has_no_secrets(client):
    body = client.get("/config").text
    assert "sk-" not in body and "provider" in body


def test_body_limit_and_validation(client):
    r = client.post("/chat", content=b'{"message": "' + b"x" * 70_000 + b'"}',
                    headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_auth_required_when_token_set(settings):
    settings = settings.model_copy(update={"api_token": SecretStr("s3cret-token")})
    c = build_container(settings, use_env_provider=False)
    with TestClient(create_app(container=c)) as tc:
        assert tc.get("/health").status_code == 200  # health stays open
        assert tc.get("/config").status_code == 401
        assert tc.get("/config", headers={"authorization": "Bearer wrong"}).status_code == 401
        assert tc.get("/config", headers={"authorization": "Bearer s3cret-token"}
                      ).status_code == 200
