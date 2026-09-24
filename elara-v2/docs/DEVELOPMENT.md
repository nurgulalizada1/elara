# Development

```bash
cd elara-v2
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[pdf,dev]"
pytest                    # full offline suite
ruff check src tests      # lint (pyflakes, bugbear, bandit subset, isort, pyupgrade)
ELARA_LIVE_TESTS=1 pytest tests/live   # real APIs
scripts/smoke.sh          # CLI smoke test against a throwaway data dir
```

## Tests

`tests/unit` covers:
- foundation (settings, DB, migrations, logging redaction, language detection)
- providers (wire formats, retries, streaming, structured output)
- conversation store, security (injection corpus, envelopes, path jail, SSRF), calculator
- memory (policy in 3 languages, store, supersede, delete, expiry, FTS safety)
- tools/executor/permissions/confirmations
- research (planner, engine, dedupe, failures, cache, synthesis anti-fabrication)
- intent/router, API, CLI/doctor, voice

`tests/integration/test_flows.py` runs `Assistant` end to end with `ScriptedProvider`
(`tests/fakes.py`) and mocked research HTTP (`tests/research_fixtures.py`). It covers:
- conversation → routing → tool → response
- context across turns
- memory store → retrieval across conversations
- research → sources → synthesis → follow-up
- prompt injection in a file → blocked memory write
- confirmation flow, tool failure, LLM failure, language, tier selection

Tests never touch real keys or your data. `tests/conftest.py` clears provider env vars and
uses `tmp_path`.

## Conventions

* Small modules; typed interfaces; dependencies passed in (see `core/container.py`).
* Model names only in `config/settings.py:DEFAULT_MODELS` (override via env).
* New DB changes → new `database/migrations/NNNN_name.sql`; never edit applied migrations.
* User-facing deterministic strings → `conversation/i18n.py` (az/en/tr).
* Errors that users should see derive from `core.errors.ElaraError`; never swallow a failure
  and report success.

## Configuration reference

See `.env.example` and `config/settings.py`. All variables use the `ELARA_` prefix, except
`ANTHROPIC_API_KEY` and `OPENAI_API_KEY`. List/dict values are JSON.
