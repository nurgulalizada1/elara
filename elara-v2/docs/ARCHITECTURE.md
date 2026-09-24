# Architecture

## Layout

```
src/elara/
  config/        Settings (env / .env), the only place default model names live
  core/          logging (JSON + redaction), errors, request context, language detection,
                 container.py (composition root: builds every service with explicit deps)
  database/      SQLite connection factory, versioned SQL migrations, audit log
  providers/     LLMProvider interface; Anthropic + OpenAI (httpx); LLMService (tiers, usage,
                 structured output)
  conversation/  conversation/message persistence, working state, i18n message catalogue
  agent/         intent classifier, router, prompts, LLM tool loop, Assistant (orchestrator)
  memory/        models, SQLite/FTS5 store, decision policy, MemoryService
  tools/         Tool contract, registry, permission policy, confirmations, executor, builtin/*
  research/      planner, HTTP layer (retry/throttle/cache), sources/*, engine, synthesis
  security/      injection detector, untrusted envelopes, path jail, SSRF guard, input cleaning
  api/           FastAPI app + schemas
  cli/           argparse entry point, REPL, doctor
  voice/         VAD, segmenter, quality gate, STT/TTS/wake-word adapters, pipeline
```

The CLI and API are thin shells. Both call `build_container(settings)` and then talk to
`Assistant`, `MemoryService`, `ToolExecutor` and `ResearchEngine`. No business logic lives in an
interface layer.

## Request flow

```
text ─► clean/validate ─► language detect ─► pending confirmation? ── yes/no ─► execute/cancel
                                   │
                                   ▼
                       IntentClassifier (deterministic, az/en/tr)
                                   │
                                   ▼
                                Router
      ┌──────────────┬─────────────┴───────┬──────────────────────────┐
   Tier 0         Tier 1 (fast)        Tier 2 (strong)            Tier 3 (research)
 deterministic    AgentLoop            AgentLoop                  planner → sources (parallel)
 handlers:        (LLM + tools)        (LLM + tools)              → validate/dedupe/rank
 greet, math,                                                     → citation-safe synthesis
 time, memory,
 files, open
                                   │
                                   ▼
                          persist messages/state ─► reply (with `resolver`)
```

## Resolution order (local-first, token-minimizing)

The LLM is the fallback reasoning engine, not the default. Before a request reaches a model,
the assistant tries, in order:

| Step | Resolver | Examples | `resolver` |
|------|----------|----------|------------|
| A | deterministic handlers | greetings, arithmetic, time, help, memory store/forget/list | `deterministic` |
| B | long-term memory | "What language do I prefer?" → the stored preference | `memory` |
| C | current conversation | "What is my test name?" after saying it earlier in the same conversation | `conversation` |
| D | local tools | list/read files, open a named folder | `tool` |
| E | cached results | research HTTP cache (labelled "from local cache") | `research` |
| F | research/API tools | "Search PubMed for …", "NCBI Gene entry for TP53", rsIDs | `research` |
| — | LLM (fast, then strong) | open-ended chat, reasoning, synthesis of multi-source literature | `llm` |

Every reply carries `tier`, `used_llm` and `resolver`, so clients and tests can verify that no
tokens were spent. Personal questions (B/C) are only answered locally when a stored statement
covers most of the question's content words. Otherwise they fall through, so unrelated
questions ("How do I install Python?") are not answered from memory.

* **Tier 0** covers steps A–D, plus follow-ups such as "tell me more about the second paper",
  which are answered from the stored record. Follow-ups that need reasoning ("explain why…")
  go to the LLM with the record as untrusted data.
* **Tier 1/2** use `AgentLoop`. The model receives tool specs and may call tools. Every call
  goes through `ToolExecutor`, and results come back as `tool_result` blocks. The loop stops
  at `agent_max_steps` or when an action needs confirmation. Tier 2 is used for long,
  multi-step or "analyze/compare/design…" requests.
* **Tier 3** handles research. When the user names a source ("Search PubMed for…"):
  - only that source is queried, and its own ranking and requested count are kept;
  - identifiers are shown only when the source returned them;
  - there is no LLM step;
  - failures are reported with their kind (unreachable, blocked, rate-limited, timeout);
  - results from other sources are shown only under an explicit "NOT from X" label.

  Otherwise the primary sources (PubMed + Europe PMC for literature) run in parallel.
  Semantic Scholar and Crossref run only when the primaries return fewer than
  `ELARA_RESEARCH_MIN_PRIMARY_RESULTS` records. `Synthesizer` then writes a citation-checked
  answer with the strong model.

## Short-term context vs long-term memory

* **Conversation context** (the messages of one conversation plus its working state) is used
  for follow-ups and step C. It is never promoted to long-term memory automatically.
* **Long-term memory** is written only on explicit requests ("remember that…", "yadda saxla
  ki…", "hatırla ki…", "save to memory: …"), through the API/CLI, or by the model's
  `memory_store` tool under the provenance rules in SECURITY.md.

## Clients (desktop, mobile, voice)

Everything a client needs goes through the HTTP API: `/chat`, `/confirmations`, `/memory`,
`/tools`, `/research`. It is authenticated by a bearer token when bound beyond localhost.
A future Tauri desktop app can bundle `elara serve` as a sidecar. Android/iOS clients talk to
the same API. The API itself needs no changes for this; transport/pairing for remote devices
is a later task.

Voice runs on the device. `VoicePipeline` only transcribes after the local wake-word detector
fires. Audio before the wake word is never transcribed or sent anywhere, and nothing streams
to an LLM API. The pipeline's `respond` callback is `Assistant.handle`, so voice gets the same
local-first routing (see VOICE.md).

## Key decisions

| Decision | Why |
|----------|-----|
| Raw httpx for LLM providers (no vendor SDKs) | One HTTP stack for providers and research, uniform retry/usage accounting, trivially testable with `httpx.MockTransport`, fewer dependencies. Anthropic assistant turns are replayed verbatim (incl. thinking blocks) inside a tool loop. |
| Deterministic intent classification first | Most personal-assistant traffic (greetings, math, memory) needs no model; this saves tokens and latency and works offline. |
| SQLite only (WAL, a new connection per unit of work) | One-user local system; safe across FastAPI threads; FTS5 gives good keyword search. `MemoryStore.search()` is the seam where a vector backend (sqlite-vec) can be added. |
| Composition root (`core/container.py`) | Explicit dependency injection. Tests inject a `ScriptedProvider` and a mocked HTTP transport without monkeypatching. |
| Tool = small class with pydantic `Input`/`Output` | Schema for the LLM comes from the model; validation is automatic; each tool is testable alone. |
| Taint tracking per turn | Once untrusted content (web, file, research, tool output) enters a turn, any write becomes "needs confirmation". Architecture beats keyword filters (see SECURITY.md). |
| Reference list generated by code | The LLM can cite only `[n]` of real records; invented numbers/DOIs/PMIDs are stripped. |

## Data model (SQLite, `database/migrations/0001_initial.sql`)

`users`, `settings`, `conversations` (with `state_json` working memory),
`messages`, `memories` (+ `memories_fts`), `tool_calls`, `pending_actions`,
`research_queries`, `sources`, `model_calls`, `security_events`, `http_cache`,
`schema_migrations`. Migrations are append-only files `NNNN_name.sql`. Each is applied once,
inside a transaction, and never destructively.

## Observability

Structured JSON logs go to `~/.local/share/elara/logs/elara.log` (rotating), and to stderr for the
server. Each record carries `request_id` and `conversation_id`. Events include `model.call` (provider, model, tier, latency,
tokens), `tool.call` (tool, origin, status, permission, taint, duration), `security.event`,
`research.done`, `assistant.route`. Values of configured secrets and common credential
patterns are redacted before output. Durable copies of model calls, tool calls and security
events are in the audit tables.

## Extending

* **New tool**: subclass `Tool` in `tools/builtin/`, declare `Input`, `Output`, `permission`,
  `output_trust`; register it in `core/container.py`; add tests.
* **New research source**: subclass `ResearchSource` in `research/sources/`, map API → `Record`
  (set `evidence_type`), add to `build_sources()` and to `QueryPlanner.SOURCES`.
* **New LLM provider**: subclass `LLMProvider` (`_complete_once`, `stream`, `check`) and add it
  to `build_provider()`. A local OpenAI-compatible server already works via
  `ELARA_PROVIDER=openai ELARA_OPENAI_BASE_URL=http://localhost:.../v1`.
* **New TTS engine**: add a class with `name`, `languages()`, `synthesize()` and register it
  in `voice/tts.py:TTS_PROVIDERS`.
