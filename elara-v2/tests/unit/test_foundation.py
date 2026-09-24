import json
import logging

import pytest
from pydantic import SecretStr

from elara.config.settings import DEFAULT_MODELS, Settings
from elara.core.language import detect_language
from elara.core.logging import JsonFormatter, Redactor, configure_logging, get_logger
from elara.database import Database
from elara.database.audit import AuditLog


class TestSettings:
    def test_provider_auto_resolution(self, tmp_path):
        s = Settings(data_dir=tmp_path, provider="auto")
        assert s.resolved_provider() == "none"
        s = Settings(data_dir=tmp_path, provider="auto", anthropic_api_key=SecretStr("sk-ant-x"))
        assert s.resolved_provider() == "anthropic"
        assert s.model_for("fast") == DEFAULT_MODELS["anthropic"]["fast"]

    def test_model_override(self, tmp_path):
        s = Settings(data_dir=tmp_path, provider="openai", model_strong="my-model")
        assert s.model_for("strong") == "my-model"
        assert s.model_for("fast") == DEFAULT_MODELS["openai"]["fast"]

    def test_env_loading_and_secret_hiding(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecretvalue")
        monkeypatch.setenv("ELARA_READ_DIRS", json.dumps([str(tmp_path)]))
        s = Settings(data_dir=tmp_path)
        assert s.anthropic_api_key.get_secret_value() == "sk-ant-supersecretvalue"
        assert "supersecret" not in repr(s)
        assert "supersecret" not in json.dumps(s.public_view())
        assert s.read_dirs == [tmp_path.resolve()]


class TestDatabase:
    def test_migrate_creates_schema(self, db):
        with db.connect() as c:
            tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for t in ("memories", "conversations", "messages", "tool_calls", "research_queries",
                  "sources", "settings", "users", "model_calls", "security_events"):
            assert t in tables
        assert db.schema_version() >= 1

    def test_migrate_is_idempotent_and_non_destructive(self, db):
        with db.connect() as c:
            c.execute("INSERT INTO settings(key, value, updated_at) VALUES ('k', 'v', 'now')")
        assert db.migrate() == []  # nothing new to apply
        with db.connect() as c:
            assert c.execute("SELECT value FROM settings WHERE key='k'").fetchone()[0] == "v"

    def test_transaction_rolls_back(self, db):
        with pytest.raises(RuntimeError):
            with db.transaction() as c:
                c.execute("INSERT INTO settings(key, value, updated_at) VALUES ('a','b','n')")
                raise RuntimeError("boom")
        with db.connect() as c:
            assert c.execute("SELECT count(*) FROM settings WHERE key='a'").fetchone()[0] == 0

    def test_fts_triggers(self, db):
        with db.connect() as c:
            c.execute("INSERT INTO memories(kind, content, source, created_at, updated_at)"
                      " VALUES ('fact', 'favorite language is Python', 'user_explicit', 'n', 'n')")
            hits = c.execute("SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'python'")
            assert len(hits.fetchall()) == 1

    def test_unwritable_path_raises_database_error(self, tmp_path):
        from elara.core.errors import DatabaseError
        blocker = tmp_path / "file"
        blocker.write_text("x")
        with pytest.raises(DatabaseError):
            Database(blocker / "sub" / "x.db").migrate()

    def test_audit_log(self, db):
        audit = AuditLog(db)
        audit.model_call(provider="p", model="m", tier="fast", purpose="chat", input_tokens=10,
                         output_tokens=5, latency_ms=3, status="ok")
        assert audit.usage_totals() == {"calls": 1, "input_tokens": 10, "output_tokens": 5}


class TestLogging:
    def test_redactor_patterns(self):
        r = Redactor(["mysecretvalue123"])
        out = r("key=mysecretvalue123 and sk-ant-abcdefghijklmnop and Bearer abc.def.ghijkl"
                " password: hunter22")
        assert "mysecretvalue123" not in out
        assert "sk-ant-abcdefghijklmnop" not in out
        assert "abc.def.ghijkl" not in out
        assert "hunter22" not in out

    def test_json_formatter_includes_context_and_redacts(self, tmp_path):
        from elara.core.context import request_scope
        configure_logging("INFO", json_format=True, console=False,
                          log_file=tmp_path / "l.log", secrets=["topsecretkey99"])
        with request_scope("req_1", "conv_1"):
            get_logger("t").info("hello", extra={"k": "topsecretkey99"})
        for h in logging.getLogger("elara").handlers:
            h.flush()
        line = json.loads((tmp_path / "l.log").read_text().strip().splitlines()[-1])
        assert line["request_id"] == "req_1" and line["conversation_id"] == "conv_1"
        assert line["k"] == "[REDACTED]"
        assert isinstance(JsonFormatter(), logging.Formatter)


@pytest.mark.parametrize("text,expected", [
    ("Salam ELARA, necəsən?", "az"),
    ("Mənim sevimli dilim Python-dur", "az"),
    ("Bu gün hava necədir?", "az"),
    ("Merhaba, nasılsın?", "tr"),
    ("Bugün ne yapıyorsun?", "tr"),
    ("Hello, how are you?", "en"),
    ("Find recent research about single-cell RNA sequencing", "en"),
    ("What's 17 * 42?", "en"),
])
def test_language_detection(text, expected):
    assert detect_language(text) == expected


def test_language_fallback_on_ambiguous():
    assert detect_language("17*42", fallback="tr") == "tr"
    assert detect_language("OK", fallback="az") == "az"
