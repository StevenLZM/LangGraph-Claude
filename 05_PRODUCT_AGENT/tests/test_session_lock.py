from __future__ import annotations

import asyncio

from api.session_lock import SessionLockManager


def test_memory_session_lock_serializes_same_session():
    async def scenario() -> None:
        manager = SessionLockManager(redis_url="", wait_timeout_seconds=0.01)

        first = await manager.acquire("session_001")
        second = await manager.acquire("session_001")

        assert first is not None
        assert second is None

        await manager.release(first)
        third = await manager.acquire("session_001")

        assert third is not None
        await manager.release(third)

    asyncio.run(scenario())
