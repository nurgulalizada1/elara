-- ELARA initial schema. Single local user today; user_id columns keep the door open.

CREATE TABLE users (
    id           INTEGER PRIMARY KEY,
    display_name TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE conversations (
    id         TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL DEFAULT 1 REFERENCES users(id),
    title      TEXT,
    state_json TEXT NOT NULL DEFAULT '{}',   -- working memory (e.g. last referenceable results)
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE messages (
    id              INTEGER PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    language        TEXT,
    intent          TEXT,
    tier            INTEGER,
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL
);
CREATE INDEX idx_messages_conv ON messages(conversation_id, id);

CREATE TABLE memories (
    id                INTEGER PRIMARY KEY,
    user_id           INTEGER NOT NULL DEFAULT 1 REFERENCES users(id),
    kind              TEXT NOT NULL CHECK (kind IN ('preference', 'fact', 'episodic')),
    key               TEXT,                 -- normalised slot, e.g. 'name', 'favorite programming language'
    content           TEXT NOT NULL,
    source            TEXT NOT NULL,        -- user_explicit | user_statement | assistant_inferred | api
    conversation_id   TEXT,
    origin_message_id INTEGER,
    confidence        REAL NOT NULL DEFAULT 1.0,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    expires_at        TEXT,
    superseded_by     INTEGER REFERENCES memories(id),
    deleted_at        TEXT
);
CREATE INDEX idx_memories_key ON memories(key) WHERE deleted_at IS NULL AND superseded_by IS NULL;

CREATE VIRTUAL TABLE memories_fts USING fts5(
    content, key, content='memories', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, key) VALUES (new.id, new.content, coalesce(new.key, ''));
END;
CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, key)
    VALUES ('delete', old.id, old.content, coalesce(old.key, ''));
END;
CREATE TRIGGER memories_au AFTER UPDATE OF content, key ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, key)
    VALUES ('delete', old.id, old.content, coalesce(old.key, ''));
    INSERT INTO memories_fts(rowid, content, key) VALUES (new.id, new.content, coalesce(new.key, ''));
END;

CREATE TABLE tool_calls (
    id              INTEGER PRIMARY KEY,
    request_id      TEXT,
    conversation_id TEXT,
    tool_name       TEXT NOT NULL,
    arguments_json  TEXT NOT NULL,
    origin          TEXT NOT NULL,          -- user | llm | api
    permission      TEXT NOT NULL,
    status          TEXT NOT NULL,          -- ok | error | denied | needs_confirmation
    confirmed       INTEGER NOT NULL DEFAULT 0,
    result_summary  TEXT,
    error           TEXT,
    duration_ms     INTEGER,
    created_at      TEXT NOT NULL
);
CREATE INDEX idx_tool_calls_conv ON tool_calls(conversation_id);

CREATE TABLE pending_actions (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT,
    tool_name       TEXT NOT NULL,
    arguments_json  TEXT NOT NULL,
    reason          TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending | confirmed | cancelled | expired
    created_at      TEXT NOT NULL,
    expires_at      TEXT NOT NULL
);

CREATE TABLE research_queries (
    id              INTEGER PRIMARY KEY,
    request_id      TEXT,
    conversation_id TEXT,
    query           TEXT NOT NULL,
    intent          TEXT NOT NULL,
    sources_json    TEXT NOT NULL,          -- per-source outcome
    result_count    INTEGER NOT NULL,
    duration_ms     INTEGER,
    created_at      TEXT NOT NULL
);

CREATE TABLE sources (
    id                INTEGER PRIMARY KEY,
    research_query_id INTEGER NOT NULL REFERENCES research_queries(id) ON DELETE CASCADE,
    rank              INTEGER NOT NULL,
    source            TEXT NOT NULL,
    external_id       TEXT,
    title             TEXT NOT NULL,
    doi               TEXT,
    pmid              TEXT,
    url               TEXT,
    year              INTEGER,
    evidence_type     TEXT NOT NULL,
    metadata_json     TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE model_calls (
    id              INTEGER PRIMARY KEY,
    request_id      TEXT,
    conversation_id TEXT,
    provider        TEXT NOT NULL,
    model           TEXT NOT NULL,
    tier            TEXT,
    purpose         TEXT,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    latency_ms      INTEGER,
    status          TEXT NOT NULL,
    error           TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE security_events (
    id              INTEGER PRIMARY KEY,
    request_id      TEXT,
    conversation_id TEXT,
    kind            TEXT NOT NULL,
    severity        TEXT NOT NULL,
    detail          TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE http_cache (
    key        TEXT PRIMARY KEY,
    status     INTEGER NOT NULL,
    body       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

INSERT INTO users (id, display_name, created_at) VALUES (1, NULL, datetime('now'));
