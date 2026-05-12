from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def test_health_returns_runtime_status():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "production-agent-customer-service"
    assert payload["version"]
    assert payload["graph_ready"] is True
    assert payload["dependencies"] == {
        "api": "ok",
        "redis": "not_configured",
        "database": "not_configured",
        "memory": "sqlite",
        "checkpointer": "none",
        "llm": "deepseek",
        "llm_startup_error": "",
        "rate_limiter": "memory",
        "rocketmq": "disabled",
    }
