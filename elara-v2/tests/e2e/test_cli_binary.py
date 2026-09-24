"""End-to-end: run the real `elara ask --json` executable as a subprocess.

Mirrors the real-machine setup: configuration only via environment variables (no .env),
`project` named path = the workspace root, a working (stand-in) Anthropic endpoint that
records every call, and research APIs made unreachable through a dead proxy. A request that
wrongly falls through to the LLM therefore shows up as a recorded call and `used_llm=true`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

LLM_TEXT = "LLM ANSWER: Gene ID 7157 https://www.ncbi.nlm.nih.gov/gene/7157"


@pytest.fixture(scope="module")
def fake_llm():
    calls: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            body = json.dumps({"model": "stand-in", "stop_reason": "end_turn",
                               "usage": {"input_tokens": 1, "output_tokens": 1},
                               "content": [{"type": "text", "text": LLM_TEXT}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", calls
    server.shutdown()


@pytest.fixture
def run(tmp_path, fake_llm):
    url, calls = fake_llm
    ws = tmp_path / "ws"
    (ws / "project").mkdir(parents=True)  # decoy: must NOT be what "project" means
    (ws / "project" / "decoy.txt").write_text("decoy")
    (ws / "notes.txt").write_text("hello from the workspace")
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path),
        "HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "", "NO_PROXY": "127.0.0.1",
        "ELARA_DATA_DIR": str(tmp_path / "data"),
        "ELARA_READ_DIRS": json.dumps([str(ws)]), "ELARA_WRITE_DIRS": json.dumps([str(ws)]),
        "ELARA_NAMED_PATHS": json.dumps({"project": str(ws)}),
        "ELARA_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "sk-ant-e2e-test-key",
        "ELARA_ANTHROPIC_BASE_URL": url, "ELARA_LLM_MAX_RETRIES": "0",
        "ELARA_RESEARCH_TIMEOUT_S": "5",
    }
    script = Path(sys.executable).with_name("elara")
    cmd = [str(script)] if script.exists() else [sys.executable, "-m", "elara.cli.main"]

    def _run(message: str) -> tuple[dict, int]:
        before = len(calls)
        out = subprocess.run([*cmd, "ask", "--json", message], env=env, cwd=tmp_path,
                             capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        return json.loads(out.stdout), len(calls) - before

    return _run


@pytest.mark.parametrize("message", ["List the files in project.", "list files in project",
                                     "list files in my project folder", "list files in workspace"])
def test_named_project_path_through_cli(run, message):
    reply, llm_calls = run(message)
    assert reply["resolver"] == "tool" and reply["tool_calls"][0]["status"] == "ok", reply
    assert "notes.txt" in reply["text"] and "not a directory" not in reply["text"]
    assert not reply["used_llm"] and llm_calls == 0


def test_named_project_file_through_cli(run):
    reply, llm_calls = run("Read project/notes.txt")
    assert reply["tool_calls"][0]["status"] == "ok" and "hello from the workspace" in reply["text"]
    assert llm_calls == 0


def test_named_path_cannot_escape_through_cli(run):
    reply, llm_calls = run("Read project/../../../etc/hostname")
    assert reply["tool_calls"][0]["status"] == "denied" and llm_calls == 0


def test_explicit_pubmed_never_reaches_llm_through_cli(run):
    reply, llm_calls = run("Search PubMed for BRCA1 breast cancer and give me the first 3 "
                           "results with their titles and PubMed IDs.")
    assert (reply["resolver"], reply["used_llm"], reply["tier"]) == ("research", False, 3)
    assert llm_calls == 0 and reply["sources"] == []
    assert "PubMed is unavailable" in reply["text"]
    assert "Europe PMC" not in reply["text"] or "NOT from PubMed" in reply["text"]


def test_explicit_ncbi_gene_never_reaches_llm_through_cli(run):
    reply, llm_calls = run("Find the NCBI Gene entry for TP53.")
    assert (reply["resolver"], reply["used_llm"], reply["tier"]) == ("research", False, 3)
    assert llm_calls == 0 and reply["sources"] == []
    assert "NCBI Gene is unavailable" in reply["text"] and "7157" not in reply["text"]


@pytest.mark.parametrize("message", ["Europe PMC results for CRISPR base editing",
                                     "Search Crossref for CRISPR base editing",
                                     "Search Semantic Scholar for CRISPR base editing",
                                     "ClinVar record for rs80357906"])
def test_every_explicit_source_fails_locally(run, message):
    reply, llm_calls = run(message)
    assert reply["resolver"] == "research" and not reply["used_llm"] and llm_calls == 0
    assert "unavailable" in reply["text"]


def test_control_llm_is_reachable_and_build_is_reported(run):
    """Proves the harness would catch a fall-through: open chat does reach the stand-in."""
    reply, llm_calls = run("Tell me a short joke")
    assert reply["used_llm"] and llm_calls == 1 and reply["text"] == LLM_TEXT
    assert reply["build"]["version"] and reply["build"]["code_path"].endswith("elara")
