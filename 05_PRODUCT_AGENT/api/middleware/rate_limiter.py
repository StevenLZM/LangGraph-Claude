from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException


@dataclass(frozen=True)
class TokenBudgetExceeded(Exception):
    reason: str
    token_count: int


class RateLimiter:
    def __init__(
        self,
        *,
        redis_url: str,
        user_rate_limit_per_minute: int = 10,
        global_qps_limit: int = 100,
        single_request_token_budget: int = 4000,
        global_hourly_token_budget: int = 500000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.redis_url = redis_url
        self.user_rate_limit_per_minute = user_rate_limit_per_minute
        self.global_qps_limit = global_qps_limit
        self.single_request_token_budget = single_request_token_budget
        self.global_hourly_token_budget = global_hourly_token_budget
        self._clock = clock
        self._user_counts: dict[str, tuple[int, float]] = {}
        self._qps_counts: dict[str, tuple[int, float]] = {}
        self._token_counts: dict[str, tuple[int, float]] = {}
        self._redis: Any | None = None
        self._redis_import_error: Exception | None = None

        if redis_url:
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(redis_url, decode_responses=True)
            except Exception as exc:  # pragma: no cover - depends on optional redis install
                self._redis_import_error = exc

    @property
    def backend(self) -> str:
        return "redis" if self.redis_url else "memory"

    async def check_user_rate(self, user_id: str) -> None:
        if self._redis_backend_enabled():
            await self._check_user_rate_redis(user_id)
            return

        now = self._clock()
        key = f"ratelimit:user:{user_id}:{int(now // 60)}"
        count, expires_at = self._increment_memory_counter(self._user_counts, key, ttl_seconds=60)
        if count > self.user_rate_limit_per_minute:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": "请求过于频繁，请稍后再试。",
                    "retry_after": self._retry_after(expires_at),
                },
            )

    async def check_global_qps(self) -> None:
        if self._redis_backend_enabled():
            await self._check_global_qps_redis()
            return

        key = f"qps:{int(self._clock())}"
        count, expires_at = self._increment_memory_counter(self._qps_counts, key, ttl_seconds=1)
        if count > self.global_qps_limit:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "global_qps_exceeded",
                    "message": "服务繁忙，请稍后重试。",
                    "retry_after": self._retry_after(expires_at),
                },
            )

    async def reserve_token_budget(self, token_count: int) -> None:
        if token_count > self.single_request_token_budget:
            raise TokenBudgetExceeded("single_request_token_budget_exceeded", token_count)

        if self._redis_backend_enabled():
            await self._reserve_token_budget_redis(token_count)
            return

        key = f"token_budget:{int(self._clock() // 3600)}"
        current, expires_at = self._token_counts.get(key, (0, self._clock() + 3600))
        if self._clock() >= expires_at:
            current = 0
            expires_at = self._clock() + 3600
        if current + token_count > self.global_hourly_token_budget:
            raise TokenBudgetExceeded("global_token_budget_exceeded", token_count)
        self._token_counts[key] = (current + token_count, expires_at)

    def reset(self) -> None:
        self._user_counts.clear()
        self._qps_counts.clear()
        self._token_counts.clear()

    def _redis_backend_enabled(self) -> bool:
        if not self.redis_url:
            return False
        if self._redis_import_error is not None or self._redis is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "rate_limiter_unavailable",
                    "message": "限流服务暂不可用，请稍后重试。",
                },
            )
        return True

    async def _check_user_rate_redis(self, user_id: str) -> None:
        assert self._redis is not None
        key = f"ratelimit:user:{user_id}:{int(self._clock() // 60)}"
        try:
            pipe = self._redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, 120)
            result = await pipe.execute()
            count = int(result[0])
            if count > self.user_rate_limit_per_minute:
                ttl = await self._redis.ttl(key)
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "rate_limit_exceeded",
                        "message": "请求过于频繁，请稍后再试。",
                        "retry_after": max(1, int(ttl)),
                    },
                )
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - exercised with real Redis
            raise self._redis_unavailable() from exc

    async def _check_global_qps_redis(self) -> None:
        assert self._redis is not None
        key = f"qps:{int(self._clock())}"
        try:
            count = int(await self._redis.incr(key))
            await self._redis.expire(key, 2)
            if count > self.global_qps_limit:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "global_qps_exceeded",
                        "message": "服务繁忙，请稍后重试。",
                        "retry_after": 1,
                    },
                )
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - exercised with real Redis
            raise self._redis_unavailable() from exc

    async def _reserve_token_budget_redis(self, token_count: int) -> None:
        assert self._redis is not None
        key = f"token_budget:{int(self._clock() // 3600)}"
        try:
            current = int(await self._redis.get(key) or 0)
            if current + token_count > self.global_hourly_token_budget:
                raise TokenBudgetExceeded("global_token_budget_exceeded", token_count)
            await self._redis.incrby(key, token_count)
            await self._redis.expire(key, 7200)
        except TokenBudgetExceeded:
            raise
        except Exception as exc:  # pragma: no cover - exercised with real Redis
            raise self._redis_unavailable() from exc

    def _increment_memory_counter(
        self,
        counters: dict[str, tuple[int, float]],
        key: str,
        *,
        ttl_seconds: int,
    ) -> tuple[int, float]:
        now = self._clock()
        count, expires_at = counters.get(key, (0, now + ttl_seconds))
        if now >= expires_at:
            count = 0
            expires_at = now + ttl_seconds
        count += 1
        counters[key] = (count, expires_at)
        return count, expires_at

    def _retry_after(self, expires_at: float) -> int:
        return max(1, math.ceil(expires_at - self._clock()))

    @staticmethod
    def _redis_unavailable() -> HTTPException:
        return HTTPException(
            status_code=503,
            detail={
                "error": "rate_limiter_unavailable",
                "message": "限流服务暂不可用，请稍后重试。",
            },
        )
