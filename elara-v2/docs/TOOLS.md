# Tools

Every tool is a small class (`tools/base.py:Tool`) with:
- a name, a description, and a category
- a pydantic `Input` and `Output` (their JSON schemas are exposed)
- a `permission` (optionally argument-dependent via `permission_for`)
- a `timeout_s`
- an `output_trust` (`system` or `untrusted`: provenance of the result)
- `safety_notes`

All invocations go through `ToolExecutor`:

lookup → validate input → permission decision → confirmation (if needed) → run with timeout
→ validate output → envelope untrusted output (+ injection scan) → audit (`tool_calls` table + log)

Failures come back as `error`, `invalid`, `denied` or `needs_confirmation`. The model is told
`FAILED …` or `NOT EXECUTED …`, never a fake success.

## Permission policy

| Level | Behaviour |
|-------|-----------|
| `read_only` | runs |
| `low_risk_write` | runs; **needs confirmation** if requested by the model in a tainted turn |
| `high_risk_write` | always needs confirmation |
| `system` | always needs confirmation |

Disabled tools are denied: `ELARA_DISABLED_TOOLS`, and `run_python` unless
`ELARA_ENABLE_CODE_EXECUTION=true`.
Confirm in chat (yes/no · bəli/xeyr · evet/hayır), via `POST /confirmations/{id}`, via
`POST /tools/{name}` with `"confirmed": true`, or with `elara tools NAME --yes`.

## Built-in tools

| Tool | Permission | Output | Notes |
|------|-----------|--------|-------|
| `calculator` | read_only | system | AST whitelist, no `eval`; exponent/factorial limits |
| `current_time` | read_only | system | local date/time/timezone |
| `list_files` | read_only | untrusted | inside read dirs; hidden files opt-in |
| `read_file` | read_only | untrusted | text (binary refused) and PDF text via `pypdf` |
| `search_files` | read_only | untrusted | name glob + content grep, skips hidden/venv/node_modules, bounded |
| `file_info` | read_only | system | type, size, mtime |
| `write_file` | low_risk_write (create) / high_risk_write (overwrite, append existing) | system | write dirs only, atomic replace |
| `delete_file` | high_risk_write | system | single regular files only |
| `open_path` | low_risk_write (folders) / system (files) | system | `xdg-open`; accepts named paths (`ELARA_NAMED_PATHS`, or a memory like "my project folder is ~/code/x") |
| `run_python` | system | untrusted | disabled by default; isolated subprocess (see SECURITY.md) |
| `web_fetch` | read_only | untrusted | public http(s) only, SSRF-guarded, HTML→text |
| `memory_search` | read_only | system | |
| `memory_store` | low_risk_write | system | provenance-checked, see MEMORY.md |
| `memory_delete` | high_risk_write | system | |
| `research_search` | read_only | untrusted | all research sources, auto intent |
| `pubmed_search`, `europepmc_search`, `crossref_search`, `semantic_scholar_search`, `clinvar_search`, `ncbi_gene_search`, `ensembl_search`, `gnomad_search` | read_only | untrusted | single-source; API/CLI only (hidden from the model to keep prompts small) |

Not included on purpose: a generic shell tool, email sending, and general web search. Web
search needs an API provider; its absence is documented rather than faked.

## Using tools directly

```bash
elara tools                                   # list
elara tools calculator --args '{"expression": "2**10"}'
elara tools delete_file --args '{"path": "old.txt"}' --yes
curl -X POST localhost:8765/tools/read_file -H 'content-type: application/json' \
     -d '{"arguments": {"path": "notes.md"}}'
```
