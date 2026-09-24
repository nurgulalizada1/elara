# Memory

All memory lives in SQLite (`memories` + FTS5 index `memories_fts`).

| Type | Meaning | Example | Lifetime |
|------|---------|---------|----------|
| working | the current conversation: recent messages + `conversations.state_json` (language, research references, focused item) | "tell me more about the second one" | per conversation (`/clear`) |
| episodic | dated/temporary personal events | "My exam is tomorrow" | expires automatically (tomorrow → 2 days, next week → 15 days, …) |
| fact | stable facts, optionally keyed | "my name is Aysel" (key `name`) | until changed or deleted |
| preference | shapes behaviour | "I prefer concise answers" | until changed or deleted |

Preferences, keyed facts and current episodic items are always in the system prompt. Other
memories are retrieved by FTS relevance to the current message.

## Decision mechanism (`memory/policy.py`)

1. **Explicit commands** (always considered):
   - English: "remember (that) …", "don't forget …"
   - Azerbaijani: "yadda saxla (ki) …", "… yadda saxla", "unutma …"
   - Turkish: "hatırla (ki) …", "aklında tut …"
2. **No implicit saving.** Ordinary statements ("My exam is tomorrow", "My name is …") stay
   in the conversation's short-term context. They can answer follow-ups in that same
   conversation, but they are **not** written to long-term memory. (`MemoryPolicy.evaluate_statement`
   still classifies such statements; nothing persists them.)
3. **Never saved**: questions ("What is PCR?"), chit-chat, requests, assistant output, and
   anything from tools, files, web or papers.

Validation rejects:
- anything not from the user (`Trust.USER` required)
- passwords, API keys and card numbers
- instruction-like text (injection detector)
- items shorter than 3 or longer than 500 characters

## Operations

| Operation | Chat | CLI | API |
|-----------|------|-----|-----|
| store | "remember that …" | `elara memory add …` | `POST /memory/store` |
| retrieve | "what's my …?", "what language do I prefer?", "mən nə ilə maraqlanıram?" (answered locally, no LLM) / automatic context | `elara memory search …`, `/memory q` | `POST /memory/search` |
| inspect | "what do you know about me?" | `elara memory list`, `/memory` | `GET /memory` |
| correct | restate a keyed fact ("remember my favorite language is Rust") → supersedes the old one | — | `PATCH /memory/{id}` |
| delete | "forget …", "bunu unut" (most recent), ambiguous matches → asks for `/forget <id>` | `elara memory delete <id>`, `/forget <id>` | `DELETE /memory/{id}` |

Deletion is a **hard delete** of the memory and every version it superseded (privacy).
Each memory keeps provenance: `source`, `conversation_id`, `origin_message_id`, `confidence`,
and `created_at` / `updated_at` / `expires_at`.

## Search

FTS5 with `unicode61 remove_diacritics 2`. Query words are prefix-matched (words longer than 6
characters are truncated to 5), which handles Azerbaijani and Turkish suffixes: "dilim" /
"dilin" / "dili". User text is quoted into the FTS query, so FTS syntax injection is not possible.
A vector backend can be added behind `MemoryStore.search()` without changing callers.
