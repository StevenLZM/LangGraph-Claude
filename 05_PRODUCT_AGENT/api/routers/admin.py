from __future__ import annotations

from fastapi import APIRouter

from memory.long_term import UserMemoryManager
from memory.semantic import SemanticMemoryStore
from memory.session_store import SessionStore
from memory.summary import PostgresSummaryMemoryStore, SummaryMemoryStore
from messaging.outbox import MessageOutboxStore, PostgresMessageOutboxStore


def create_admin_router(
    *,
    session_store: SessionStore,
    user_memory_manager: UserMemoryManager,
    summary_memory_store: SummaryMemoryStore | PostgresSummaryMemoryStore | None = None,
    semantic_memory_store: SemanticMemoryStore | None = None,
    message_outbox_store: MessageOutboxStore | PostgresMessageOutboxStore | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/admin", tags=["admin"])

    @router.get("/sessions")
    async def list_sessions(limit: int = 100) -> dict:
        return {"sessions": session_store.list_public_sessions(limit=limit)}

    @router.get("/users/{user_id}/memories")
    async def list_user_memories(user_id: str, limit: int = 100) -> dict:
        semantic_memories = []
        if semantic_memory_store is not None:
            semantic_memories = semantic_memory_store.list_user(user_id, limit=limit)
        return {
            "user_id": user_id,
            "memories": user_memory_manager.list_memories(user_id, limit=limit),
            "semantic_memories": semantic_memories,
        }

    @router.get("/sessions/{session_id}/summary")
    async def get_session_summary(session_id: str) -> dict:
        if summary_memory_store is None:
            return {"session_id": session_id, "summary": None}
        summary = summary_memory_store.load_summary(session_id)
        return {"session_id": session_id, "summary": summary.to_dict() if summary else None}

    @router.get("/stats/transfers")
    async def transfer_stats() -> dict:
        return session_store.summarize_transfers()

    @router.get("/messages/outbox")
    async def list_message_outbox(limit: int = 100) -> dict:
        if message_outbox_store is None:
            return {"events": []}
        return {"events": message_outbox_store.list_events(limit=limit)}

    return router
