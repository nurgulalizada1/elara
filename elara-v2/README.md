# ELARA

ELARA is a local-first personal AI assistant. It speaks Azerbaijani first, plus English and Turkish.
It runs on your Linux machine. Its state (conversations, memories, audit logs, cache) lives in one SQLite file.
An LLM is used only when a request needs one.

```
You: Salam ELARA
ELARA: Salam! Necəsən?
You: What's 17 * 42?
ELARA: 17 * 42 = 714                         ← no model call
You: Remember that I prefer concise answers
ELARA: Got it, I'll remember: I prefer concise answers
You: Find recent research about single-cell RNA sequencing
ELARA: …synthesis with [1]…[n] citations, generated only from records the APIs returned…
```

## What works today

| Area | Status |
|------|--------|
| CLI chat (`elara`), one-shot (`elara ask`), diagnostics (`elara doctor`) | ✓ |
| HTTP API (`elara serve`, FastAPI, optional bearer token) | ✓ |
| LLM providers: Anthropic and OpenAI, over httpx: retries, streaming, structured output, token tracking | ✓ |
| Tiered routing: deterministic, then fast model, then strong model, then the research workflow | ✓ |
| Memory: explicit only ("remember/save …"), answered locally without the LLM; corrections, deletion, expiry, FTS5 search, provenance | ✓ |
| Tools (23): calculator, time, files (list/read incl. PDF/write/search/info/delete), open_path, run_python (off by default), web_fetch, memory tools, research tools | ✓ |
| Research sources: PubMed, Europe PMC, Crossref, Semantic Scholar, ClinVar, NCBI Gene, Ensembl, gnomAD | ✓ (unit-tested on recorded response shapes; see *Live verification*) |
| Security: untrusted-content envelopes, injection detection, taint-based permission escalation, path jail, SSRF guard, secret redaction | ✓ |
| Voice: energy VAD, utterance segmentation, transcript quality gate, espeak-ng TTS, faster-whisper adapter, wake-word adapter | interfaces + components; **no microphone loop yet** |

## Install

Requires Python 3.12+ on Linux.

```bash
cd elara-v2
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[pdf,dev]"
cp .env.example .env         # then add ANTHROPIC_API_KEY (or OPENAI_API_KEY)
elara doctor
elara                        # chat
```

Without an API key, ELARA still handles greetings, math, time, memory, files and research searches
(it lists the raw records instead of writing a synthesis). Requests that need an LLM get a clear message.

## Commands

```
elara                      interactive chat (/help /memory /tools /status /clear /new /exit)
elara ask "message"        one-shot (add --json for the full structured reply)
elara doctor [--offline] [--tests]
elara serve [--host --port]  HTTP API on 127.0.0.1:8765; binding elsewhere requires ELARA_API_TOKEN
elara memory list|search|add|delete ...
elara tools [name --args '{"...": ...}' [--yes]]
elara research "BRCA1 gene"
```

## API

`GET /health` · `POST /chat` · `POST /confirmations/{id}` · `GET /conversations` ·
`GET /conversations/{id}/messages` · `GET /memory` · `POST /memory/search` · `POST /memory/store` ·
`PATCH /memory/{id}` · `DELETE /memory/{id}` · `GET /tools` · `POST /tools/{name}` ·
`POST /research` · `GET /config` · `GET /status`. Interactive docs: `http://127.0.0.1:8765/docs`.

## Tests

```bash
pytest                              # ~240 tests, offline, ~15 s
ELARA_LIVE_TESTS=1 pytest tests/live  # real research APIs (+ LLM round-trip if a key is set)
ruff check src tests
```

### Live verification

This version was built in a sandbox that could not reach the scientific APIs or a real LLM key.
The following has been exercised for real:
- the CLI and HTTP server end to end;
- the OpenAI provider against a local OpenAI-compatible endpoint (tool call, then answer) over a real socket.

The Anthropic, PubMed, Europe PMC, Crossref, Semantic Scholar, ClinVar, NCBI Gene, Ensembl and gnomAD clients follow each API's documented request and response format. They are tested against recorded response shapes.
Run `tests/live` once on your machine to confirm them. gnomAD's GraphQL schema is the one most likely to need an adjustment.

## Documentation

[ARCHITECTURE](docs/ARCHITECTURE.md) · [SECURITY](docs/SECURITY.md) · [MEMORY](docs/MEMORY.md) ·
[TOOLS](docs/TOOLS.md) · [RESEARCH](docs/RESEARCH.md) · [VOICE](docs/VOICE.md) ·
[DEVELOPMENT](docs/DEVELOPMENT.md) · [BUILD_PLAN](docs/BUILD_PLAN.md)
