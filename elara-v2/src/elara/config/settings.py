"""Centralised configuration.

All configuration comes from environment variables (prefix ``ELARA_``) or a ``.env``
file. Secrets are held as ``SecretStr`` so they never appear in reprs or logs.
Default model identifiers live *only* here (``DEFAULT_MODELS``).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["auto", "anthropic", "openai", "none"]
Language = Literal["az", "en", "tr"]

# The only place concrete model names appear. Override with ELARA_MODEL_FAST/STRONG.
DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "anthropic": {"fast": "claude-haiku-4-5", "strong": "claude-opus-5"},
    "openai": {"fast": "gpt-4o-mini", "strong": "gpt-4o"},
}


def _default_data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "elara"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ELARA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- general ---
    data_dir: Path = Field(default_factory=_default_data_dir)
    default_language: Language = "az"
    log_level: str = "INFO"
    log_json: bool = True

    # --- LLM providers ---
    provider: ProviderName = "auto"
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "ELARA_ANTHROPIC_API_KEY")
    )
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "ELARA_OPENAI_API_KEY")
    )
    anthropic_base_url: str = "https://api.anthropic.com"
    openai_base_url: str = "https://api.openai.com/v1"
    model_fast: str | None = None
    model_strong: str | None = None
    # None = do not send a temperature (some models reject sampling parameters).
    temperature: float | None = None
    max_output_tokens: int = 4096
    llm_timeout_s: float = 90.0
    llm_max_retries: int = 3
    agent_max_steps: int = 6
    context_max_messages: int = 20

    # --- files ---
    read_dirs: list[Path] = Field(default_factory=lambda: [Path.home()])
    write_dirs: list[Path] = Field(default_factory=lambda: [Path.home() / "elara-workspace"])
    named_paths: dict[str, Path] = Field(default_factory=dict)
    max_file_read_bytes: int = 1_000_000

    # --- research ---
    ncbi_api_key: SecretStr | None = None
    semantic_scholar_api_key: SecretStr | None = None
    contact_email: str | None = None
    research_timeout_s: float = 20.0
    research_max_results: int = 5
    http_cache_ttl_s: int = 6 * 3600

    # --- api ---
    api_host: str = "127.0.0.1"
    api_port: int = 8765
    api_token: SecretStr | None = None
    max_request_bytes: int = 64_000
    max_message_chars: int = 8_000

    # --- security ---
    enable_code_execution: bool = False
    disabled_tools: list[str] = Field(default_factory=list)
    allow_private_network_fetch: bool = False
    confirmation_ttl_s: int = 600

    # --- voice ---
    tts_provider: str = "espeak"
    tts_voices: dict[str, str] = Field(
        default_factory=lambda: {"az": "az", "en": "en-us", "tr": "tr"}
    )
    stt_provider: str = "faster-whisper"
    stt_model: str = "small"

    @field_validator("data_dir", mode="after")
    @classmethod
    def _expand_data_dir(cls, v: Path) -> Path:
        return v.expanduser()

    @field_validator("read_dirs", "write_dirs", mode="after")
    @classmethod
    def _expand_dirs(cls, v: list[Path]) -> list[Path]:
        return [p.expanduser().resolve() for p in v]

    @field_validator("named_paths", mode="after")
    @classmethod
    def _expand_named(cls, v: dict[str, Path]) -> dict[str, Path]:
        return {k.lower(): p.expanduser().resolve() for k, p in v.items()}

    @property
    def db_path(self) -> Path:
        return self.data_dir / "elara.db"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "logs" / "elara.log"

    def resolved_provider(self) -> str:
        """Pick the provider: explicit setting, else the first one with a key."""
        if self.provider != "auto":
            return self.provider
        if self.anthropic_api_key and self.anthropic_api_key.get_secret_value():
            return "anthropic"
        if self.openai_api_key and self.openai_api_key.get_secret_value():
            return "openai"
        return "none"

    def model_for(self, tier: Literal["fast", "strong"]) -> str | None:
        provider = self.resolved_provider()
        override = self.model_fast if tier == "fast" else self.model_strong
        if override:
            return override
        return DEFAULT_MODELS.get(provider, {}).get(tier)

    def secret_values(self) -> list[str]:
        """All configured secret values, for log redaction."""
        out = []
        for s in (self.anthropic_api_key, self.openai_api_key, self.ncbi_api_key,
                  self.semantic_scholar_api_key, self.api_token):
            if s is not None and s.get_secret_value():
                out.append(s.get_secret_value())
        return out

    def public_view(self) -> dict:
        """Configuration safe to expose via API / CLI (no secrets)."""
        return {
            "data_dir": str(self.data_dir),
            "default_language": self.default_language,
            "provider": self.resolved_provider(),
            "model_fast": self.model_for("fast"),
            "model_strong": self.model_for("strong"),
            "anthropic_key_set": bool(self.anthropic_api_key),
            "openai_key_set": bool(self.openai_api_key),
            "read_dirs": [str(p) for p in self.read_dirs],
            "write_dirs": [str(p) for p in self.write_dirs],
            "named_paths": {k: str(v) for k, v in self.named_paths.items()},
            "enable_code_execution": self.enable_code_execution,
            "disabled_tools": self.disabled_tools,
            "api": {"host": self.api_host, "port": self.api_port,
                    "auth_required": bool(self.api_token)},
            "tts_provider": self.tts_provider,
            "stt_provider": self.stt_provider,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
