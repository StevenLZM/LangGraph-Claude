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

    llm_mode: str = "deepseek"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_max_model: str = "deepseek-v4-pro"
    deepseek_light_model: str = "deepseek-v4-flash"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = ""
    primary_model: str = "claude-sonnet-4-6"
    fallback_model: str = "gpt-4o-mini"

    redis_url: str = ""
    database_url: str = ""
    checkpointer_db: str = str(ROOT / "data" / "sessions.db")
    memory_db: str = str(ROOT / "data" / "memory.db")
    chat_request_db: str = str(ROOT / "data" / "chat_requests.db")
    message_outbox_db: str = str(ROOT / "data" / "message_outbox.db")
    storage_backend: str = "sqlite"
    checkpointer_backend: str = "none"
    checkpointer_url: str = ""
    checkpointer_setup: bool = True

    rocketmq_enabled: bool = False
    rocketmq_endpoint: str = "localhost:9876"
    rocketmq_producer_group: str = "PID_05_PRODUCT_AGENT"
    rocketmq_access_key: str = ""
    rocketmq_secret_key: str = ""
    rocketmq_normal_topic: str = "agent-customer-service-normal-v1"
    rocketmq_fifo_topic: str = "agent-customer-service-fifo-v1"
    rocketmq_delay_topic: str = "agent-customer-service-delay-v1"
    rocketmq_transaction_topic: str = "agent-customer-service-tx-v1"

    user_rate_limit_per_minute: int = 10
    global_qps_limit: int = 100
    single_request_token_budget: int = 4000
    global_hourly_token_budget: int = 500000

    langchain_tracing_v2: bool = False
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str = ""
    langchain_project: str = "production-agent-customer-service"
    observability_env: str = "local"
    quality_alert_threshold: int = 70


settings = Settings()
