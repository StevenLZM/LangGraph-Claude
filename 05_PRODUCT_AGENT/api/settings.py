from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "production-agent-customer-service"
    app_version: str = "0.1.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    llm_mode: str = "offline_stub"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    primary_model: str = "claude-sonnet-4-6"
    fallback_model: str = "gpt-4o-mini"

    redis_url: str = ""
    database_url: str = ""
    checkpointer_db: str = str(ROOT / "data" / "sessions.db")
    memory_db: str = str(ROOT / "data" / "memory.db")

    langchain_tracing_v2: bool = False
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str = ""
    langchain_project: str = "production-agent-customer-service"


settings = Settings()
