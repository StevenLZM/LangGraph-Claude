from __future__ import annotations

from api.settings import Settings


def test_settings_defaults_are_local_development_friendly():
    settings = Settings()

    assert settings.app_name == "production-agent-customer-service"
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000
    assert settings.llm_mode == "offline_stub"
    assert settings.redis_url == ""
    assert settings.database_url == ""
