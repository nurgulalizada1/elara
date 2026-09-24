# ELARA v2 — Build Plan

Clean-room rebuild. Lives in `elara-v2/`; it shares no code with the earlier implementation
in the repository root (which was left untouched).

Each phase ends with: tests green → architecture review → docs updated.

| # | Phase | Deliverables | Status |
|---|-------|--------------|--------|
| 1 | Foundation | pyproject, settings (env/.env, SecretStr), JSON logging + redaction, SQLite + versioned migrations, audit tables, language detection | done |
| 2 | Conversation engine | message/context model, conversation store, provider abstraction (Anthropic, OpenAI over httpx), retry/backoff, usage tracking, streaming, structured output | done |
| 3 | Agent core | deterministic intent classifier, tiered router, tool registry, executor with permissions/confirmations, LLM tool loop | done |
| 4 | Memory | SQLite+FTS5 store, decision policy, extraction (az/en/tr), correction/supersede, deletion, inspection, provenance | done |
| 5 | Tools | calculator, time, files (list/read/write/search/info/delete), open_path, run_python (sandboxed subprocess, off by default), web_fetch (SSRF-guarded) | done |
| 6 | Research | source abstraction, PubMed, Europe PMC, Crossref, Semantic Scholar, ClinVar, NCBI Gene, Ensembl, gnomAD; dedupe/validate/rank; citation-safe synthesis | done |
| 7 | Security hardening | untrusted-content envelopes, injection detector, taint tracking, path jail, output validation | done |
| 8 | Voice | VAD / wake-word / STT / TTS interfaces, energy VAD, transcript quality gate, espeak-ng TTS, faster-whisper adapter | partial — components done; no mic capture loop, no trained wake-word model |
| 9 | API | FastAPI app: health, chat, memory, tools, config, research, confirmations; bearer auth, size limits | done |
| 10 | Integration + doctor | end-to-end tests, `elara doctor`, docs | done |

## Future work (not built yet)
- Microphone capture loop + wake-word model for "Hey ELARA" (needs a trained model).
- Semantic (vector) memory backend via sqlite-vec behind the `MemoryIndex` interface.
- Local LLM provider (e.g. llama.cpp / Ollama HTTP) behind `LLMProvider`.
- Desktop / mobile clients over the HTTP API.
- More scientific sources: UniProt, PDB, AlphaFold, BLAST.

## Verification notes
- Offline suite: all unit + integration tests green; `ruff` clean.
- Exercised for real: CLI chat, `elara ask`, `elara doctor`, `elara serve` + curl, OpenAI provider
  against a local OpenAI-compatible endpoint (tool loop over a real socket).
- Not verifiable in the build sandbox (network policy): live research APIs and a real
  Anthropic/OpenAI key → run `ELARA_LIVE_TESTS=1 pytest tests/live` locally.
