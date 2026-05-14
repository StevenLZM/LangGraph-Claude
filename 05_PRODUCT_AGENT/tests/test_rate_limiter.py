from __future__ import annotations

import importlib
import uuid

import pytest
from fastapi.testclient import TestClient

import api.main as main


def _rate_limiter_class():
    try:
        return importlib.import_module("api.middleware.rate_limiter").RateLimiter
    except (ImportError, AttributeError) as exc:
        pytest.fail(f"RateLimiter is not implemented: {exc}")


def _post_chat(client: TestClient, *, user_id: str, session_id: str, message: str = "我的订单 ORD123456 到哪了？"):
    return client.post(
        "/chat",
        json={
            "user_id": user_id,
            "session_id": session_id,
            "request_id": uuid.uuid4().hex,
            "message": message,
        },
    )


def test_same_user_eleventh_request_per_minute_returns_429():
    client = TestClient(main.app)

    for index in range(10):
        response = _post_chat(client, user_id="limited_user", session_id=f"limited_session_{index}")
        assert response.status_code == 200

    response = _post_chat(client, user_id="limited_user", session_id="limited_session_11")

    assert response.status_code == 429
    assert response.json()["detail"]["error"] == "rate_limit_exceeded"
    assert "请求过于频繁" in response.json()["detail"]["message"]
    assert response.json()["detail"]["retry_after"] > 0


def test_global_qps_limit_returns_503(monkeypatch):
    RateLimiter = _rate_limiter_class()
    limiter = RateLimiter(redis_url="", global_qps_limit=1)
    monkeypatch.setattr(main, "rate_limiter", limiter)
    client = TestClient(main.app)

    first = _post_chat(client, user_id="qps_user_1", session_id="qps_session_1")
    second = _post_chat(client, user_id="qps_user_2", session_id="qps_session_2")

    assert first.status_code == 200
    assert second.status_code == 503
    assert second.json()["detail"]["error"] == "global_qps_exceeded"
    assert "服务繁忙" in second.json()["detail"]["message"]


def test_global_token_budget_exceeded_returns_degraded_chat_response(monkeypatch):
    RateLimiter = _rate_limiter_class()
    limiter = RateLimiter(
        redis_url="",
        single_request_token_budget=4000,
        global_hourly_token_budget=1,
    )
    monkeypatch.setattr(main, "rate_limiter", limiter)
    client = TestClient(main.app)

    response = _post_chat(
        client,
        user_id="budget_user",
        session_id="budget_session",
        message="我的订单 ORD123456 到哪了？",
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["degraded"] is True
    assert payload["degrade_reason"] == "global_token_budget_exceeded"
    assert "简化回复" in payload["answer"]
    assert payload["token_used"] > 1


def test_single_request_token_budget_exceeded_returns_degraded_chat_response(monkeypatch):
    RateLimiter = _rate_limiter_class()
    limiter = RateLimiter(
        redis_url="",
        single_request_token_budget=2,
        global_hourly_token_budget=500000,
    )
    monkeypatch.setattr(main, "rate_limiter", limiter)
    client = TestClient(main.app)

    response = _post_chat(
        client,
        user_id="single_budget_user",
        session_id="single_budget_session",
        message="我需要查询订单 ORD123456 的物流状态和退款进度",
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["degraded"] is True
    assert payload["degrade_reason"] == "single_request_token_budget_exceeded"
    assert "简化回复" in payload["answer"]
