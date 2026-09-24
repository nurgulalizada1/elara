from __future__ import annotations

from pathlib import Path

import pytest

from elara.config.settings import Settings
from elara.database import Database


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Tests never read the developer's real keys, .env or data dir."""
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ELARA_PROVIDER", "ELARA_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return Settings(
        data_dir=tmp_path / "data",
        read_dirs=[ws],
        write_dirs=[ws],
        provider="none",
        log_json=True,
    )


@pytest.fixture
def db(settings: Settings) -> Database:
    d = Database(settings.db_path)
    d.migrate()
    return d
