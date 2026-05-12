from __future__ import annotations

from fastapi import APIRouter

from memory.long_term import UserMemoryManager
from memory.session_store import SessionStore
from messaging.outbox import MessageOutboxStore


def create_admin_router(
    *,
    session_store: SessionStore,
    user_memory_manager: UserMemoryManager,
    message_outbox_store: MessageOutboxStore | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/admin", tags=["admin"])

    @router.get("/sessions")
    async def list_sessions(limit: int = 100) -> dict:
        return {"sessions": session_store.list_public_sessions(limit=limit)}

    @router.get("/users/{user_id}/memories")
    async def list_user_memories(user_id: str, limit: int = 100) -> dict:
        return {
            "user_id": user_id,
            "memories": user_memory_manager.list_memories(user_id, limit=limit),
        }

    @router.get("/stats/transfers")
    async def transfer_stats() -> dict:
        return session_store.summarize_transfers()

    @router.get("/messages/outbox")
    async def list_message_outbox(limit: int = 100) -> dict:
        if message_outbox_store is None:
            return {"events": []}
        return {"events": message_outbox_store.list_events(limit=limit)}

    return router
