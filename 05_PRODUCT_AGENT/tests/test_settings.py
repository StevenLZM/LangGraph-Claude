from __future__ import annotations

from api.settings import Settings


def test_settings_defaults_are_local_development_friendly():
    settings = Settings(_env_file=None)

    assert settings.app_name == "production-agent-customer-service"
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000
    assert settings.llm_mode == "deepseek"
    assert settings.redis_url == ""
    assert settings.database_url == ""
    assert settings.storage_backend == "sqlite"
    assert settings.checkpointer_backend == "none"
    assert settings.checkpointer_url == ""
    assert settings.checkpointer_setup is True
    assert settings.user_rate_limit_per_minute == 10
    assert settings.global_qps_limit == 100
    assert settings.single_request_token_budget == 4000
    assert settings.global_hourly_token_budget == 500000
    assert settings.observability_env == "local"
    assert settings.quality_alert_threshold == 70
