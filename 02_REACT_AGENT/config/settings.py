"""Runtime settings for the 02 ReAct Agent project."""
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

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_max_model: str = "deepseek-v4-pro"
    deepseek_light_model: str = "deepseek-v4-flash"

    tavily_api_key: str = ""

    max_react_iterations: int = 10
    max_plan_steps: int = 8
    python_timeout_seconds: int = 5

    mcp_config_path: str = str(ROOT / ".mcp.json")


settings = Settings()
