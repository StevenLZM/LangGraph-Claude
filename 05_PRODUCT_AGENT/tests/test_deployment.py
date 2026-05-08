from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_dockerfile_runs_fastapi_application():
    dockerfile = _read("Dockerfile")

    assert "FROM python:" in dockerfile
    assert "pip install --no-cache-dir -r requirements.txt" in dockerfile
    assert "uvicorn" in dockerfile
    assert "api.main:app" in dockerfile
    assert "EXPOSE 8000" in dockerfile


def test_compose_defines_m5_runtime_stack():
    compose = _read("docker-compose.yml")

    for service_name in ("api:", "redis:", "postgres:", "prometheus:", "grafana:", "locust:"):
        assert service_name in compose
    assert "REDIS_URL=redis://redis:6379/0" in compose
    assert "pgvector/pgvector:pg16" in compose
    assert "./infra/prometheus.yml:/etc/prometheus/prometheus.yml" in compose
    assert "./infra/grafana/provisioning:/etc/grafana/provisioning" in compose
    assert "profiles:" in compose
    assert "loadtest" in compose


def test_prometheus_scrapes_api_metrics_endpoint():
    prometheus = _read("infra/prometheus.yml")

    assert "job_name: customer-service-api" in prometheus
    assert "metrics_path: /metrics" in prometheus
    assert "api:8000" in prometheus


def test_grafana_provisioning_includes_prometheus_dashboard():
    datasource = _read("infra/grafana/provisioning/datasources/prometheus.yml")
    dashboard_provider = _read("infra/grafana/provisioning/dashboards/customer-service.yml")
    dashboard = _read("infra/grafana/dashboards/customer-service.json")

    assert "Prometheus" in datasource
    assert "http://prometheus:9090" in datasource
    assert "/var/lib/grafana/dashboards" in dashboard_provider
    for metric in (
        "agent_requests_total",
        "agent_response_time_seconds",
        "agent_tokens_total",
        "agent_errors_total",
        "agent_human_transfers_total",
        "agent_quality_score",
    ):
        assert metric in dashboard


def test_locustfile_covers_core_customer_service_scenarios():
    locustfile = _read("load_tests/locustfile.py")

    assert "HttpUser" in locustfile
    assert "GET /health" in locustfile
    assert "ORD123456" in locustfile
    assert "退款" in locustfile
    assert "转人工" in locustfile
    assert "/metrics" in locustfile


def test_env_example_does_not_contain_real_secrets():
    env_example = _read(".env.example")

    assert "LANGCHAIN_API_KEY=" in env_example
    assert "GRAFANA_ADMIN_PASSWORD=admin" in env_example
    assert "REDIS_URL=" in env_example
    assert "lsv2_" not in env_example
    assert "sk-" not in env_example
