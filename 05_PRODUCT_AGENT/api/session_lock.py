from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SessionLock:
    session_id: str
    token: str
    backend: str
    key: str
    memory_lock: asyncio.Lock | None = None


class SessionLockManager:
    def __init__(
        self,
        *,
        redis_url: str,
        ttl_seconds: float = 120.0,
        wait_timeout_seconds: float = 5.0,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.wait_timeout_seconds = wait_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self._redis: Any | None = None
        self._redis_import_error: Exception | None = None
        self._memory_locks: dict[str, asyncio.Lock] = {}

        if redis_url:
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(redis_url, decode_responses=True)
            except Exception as exc:  # pragma: no cover - depends on optional redis install
                self._redis_import_error = exc

    @property
    def backend(self) -> str:
        return "redis" if self.redis_url else "memory"

    async def acquire(self, session_id: str) -> SessionLock | None:
        if self.redis_url:
            return await self._acquire_redis(session_id)
        return await self._acquire_memory(session_id)

    async def release(self, lock: SessionLock) -> None:
        if lock.backend == "redis":
            await self._release_redis(lock)
            return
        if lock.memory_lock is not None and lock.memory_lock.locked():
            lock.memory_lock.release()

    async def _acquire_memory(self, session_id: str) -> SessionLock | None:
        lock = self._memory_locks.setdefault(session_id, asyncio.Lock())
        try:
            await asyncio.wait_for(lock.acquire(), timeout=self.wait_timeout_seconds)
        except TimeoutError:
            return None
        return SessionLock(
            session_id=session_id,
            token="memory",
            backend="memory",
            key=f"memory:{session_id}",
            memory_lock=lock,
        )

    async def _acquire_redis(self, session_id: str) -> SessionLock | None:
        if self._redis_import_error is not None or self._redis is None:
            raise RuntimeError("redis is required for distributed session locks")

        key = f"chat:session_lock:{session_id}"
        token = uuid.uuid4().hex
        deadline = asyncio.get_running_loop().time() + self.wait_timeout_seconds
        while True:
            acquired = await self._redis.set(
                key,
                token,
                nx=True,
                px=int(self.ttl_seconds * 1000),
            )
            if acquired:
                return SessionLock(session_id=session_id, token=token, backend="redis", key=key)
            if asyncio.get_running_loop().time() >= deadline:
                return None
            await asyncio.sleep(self.poll_interval_seconds)

    async def _release_redis(self, lock: SessionLock) -> None:
        if self._redis is None:
            return
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        end
        return 0
        """
        await self._redis.eval(script, 1, lock.key, lock.token)
