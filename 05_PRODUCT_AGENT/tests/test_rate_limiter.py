from __future__ import annotations

import importlib
import asyncio
import uuid

import pytest
from fastapi import HTTPException
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


class FakeRedisLua:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.eval_calls: list[tuple[str, int, str, int, int]] = []

    async def eval(self, script: str, numkeys: int, key: str, ttl_seconds: int, limit: int) -> list[int]:
        self.eval_calls.append((script, numkeys, key, int(ttl_seconds), int(limit)))
        count = self.counts.get(key, 0) + 1
        self.counts[key] = count
        allowed = 1 if count <= int(limit) else 0
        return [allowed, count, int(ttl_seconds)]


def _redis_limiter(**kwargs):
    RateLimiter = _rate_limiter_class()
    fake_redis = FakeRedisLua()
    limiter = RateLimiter(redis_url="redis://test", **kwargs)
    limiter._redis = fake_redis
    limiter._redis_import_error = None
    return limiter, fake_redis


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


def test_redis_user_rate_limit_uses_lua_script():
    limiter, fake_redis = _redis_limiter(user_rate_limit_per_minute=1, clock=lambda: 120.0)

    asyncio.run(limiter.check_user_rate("redis_user"))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(limiter.check_user_rate("redis_user"))

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["error"] == "rate_limit_exceeded"
    assert exc_info.value.detail["retry_after"] == 120
    assert len(fake_redis.eval_calls) == 2
    assert fake_redis.eval_calls[0][2:] == ("ratelimit:user:redis_user:2", 120, 1)


def test_redis_global_qps_limit_uses_lua_script():
    limiter, fake_redis = _redis_limiter(global_qps_limit=1, clock=lambda: 1234.0)

    asyncio.run(limiter.check_global_qps())
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(limiter.check_global_qps())

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["error"] == "global_qps_exceeded"
    assert exc_info.value.detail["retry_after"] == 1
    assert len(fake_redis.eval_calls) == 2
    assert fake_redis.eval_calls[0][2:] == ("qps:1234", 2, 1)


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
