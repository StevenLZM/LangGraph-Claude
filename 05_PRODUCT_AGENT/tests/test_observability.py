from __future__ import annotations

from fastapi.testclient import TestClient

import api.main as main


def _post_chat(
    client: TestClient,
    *,
    user_id: str = "observability_user",
    session_id: str = "observability_session",
    message: str = "我的订单 ORD123456 到哪了？",
):
    return client.post(
        "/chat",
        json={"user_id": user_id, "session_id": session_id, "message": message},
    )


def test_metrics_endpoint_exposes_core_prometheus_metrics():
    client = TestClient(main.app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "agent_requests_total" in body
    assert "agent_response_time_seconds" in body
    assert "agent_tokens_total" in body
    assert "agent_active_sessions" in body
    assert "agent_quality_score" in body


def test_chat_records_request_token_latency_and_quality_metrics():
    from monitoring.metrics import reset_observability_metrics

    reset_observability_metrics()
    client = TestClient(main.app)

    response = _post_chat(client)

    assert response.status_code == 200
    body = client.get("/metrics").text
    assert 'agent_requests_total{status="success"} 1.0' in body
    assert 'agent_tokens_total{type="estimated"}' in body
    assert "agent_response_time_seconds_count 1.0" in body
    assert "agent_quality_score_count 1.0" in body
    assert "agent_active_sessions 1.0" in body


def test_quality_evaluator_flags_low_quality_answer_and_records_alert():
    from monitoring.evaluator import AutoQualityEvaluator

    evaluator = AutoQualityEvaluator(alert_threshold=70)

    result = evaluator.evaluate(
        question="我的订单 ORD123456 到哪了？",
        answer="不知道。",
        context={"order_context": {"order_id": "ORD123456"}},
    )

    assert result.score < 70
    assert result.passed is False
    assert result.issues
    assert evaluator.alert_events
    assert evaluator.alert_events[-1]["score"] == result.score
    assert evaluator.alert_events[-1]["level"] == "warning"


def test_langsmith_trace_config_includes_session_and_user_metadata():
    from monitoring.tracing import build_trace_config

    config = build_trace_config(
        session_id="trace_session",
        user_id="trace_user",
        environment="test",
        app_version="0.1-test",
    )

    assert config["tags"] == ["customer-service", "session:trace_session", "user:trace_user"]
    assert config["metadata"]["session_id"] == "trace_session"
    assert config["metadata"]["user_id"] == "trace_user"
    assert config["metadata"]["environment"] == "test"
    assert config["metadata"]["version"] == "0.1-test"
