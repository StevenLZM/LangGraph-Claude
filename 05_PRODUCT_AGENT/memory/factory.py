from __future__ import annotations

from typing import Any

from memory.long_term import PostgresUserMemoryManager, UserMemoryManager
from memory.semantic import MilvusSemanticMemoryStore, NoopSemanticMemoryStore, SQLiteSemanticMemoryStore
from memory.session_store import PostgresSessionStore, SessionStore
from memory.summary import PostgresSummaryMemoryStore, SummaryMemoryStore


def build_session_store(settings: Any) -> SessionStore | PostgresSessionStore:
    backend = settings.storage_backend.casefold()
    if backend == "sqlite":
        return SessionStore(settings.memory_db)
    if backend == "postgres":
        if not settings.database_url:
            raise ValueError("DATABASE_URL is required when STORAGE_BACKEND=postgres")
        return PostgresSessionStore(settings.database_url)
    raise ValueError(f"Unsupported STORAGE_BACKEND: {settings.storage_backend}")


def build_user_memory_manager(settings: Any) -> UserMemoryManager | PostgresUserMemoryManager:
    backend = settings.storage_backend.casefold()
    if backend == "sqlite":
        return UserMemoryManager(settings.memory_db)
    if backend == "postgres":
        if not settings.database_url:
            raise ValueError("DATABASE_URL is required when STORAGE_BACKEND=postgres")
        return PostgresUserMemoryManager(settings.database_url)
    raise ValueError(f"Unsupported STORAGE_BACKEND: {settings.storage_backend}")


def build_summary_memory_store(settings: Any) -> SummaryMemoryStore | PostgresSummaryMemoryStore:
    backend = settings.storage_backend.casefold()
    if backend == "sqlite":
        return SummaryMemoryStore(settings.summary_memory_db)
    if backend == "postgres":
        if not settings.database_url:
            raise ValueError("DATABASE_URL is required when STORAGE_BACKEND=postgres")
        return PostgresSummaryMemoryStore(settings.database_url)
    raise ValueError(f"Unsupported STORAGE_BACKEND: {settings.storage_backend}")


def build_semantic_memory_store(
    settings: Any,
) -> NoopSemanticMemoryStore | SQLiteSemanticMemoryStore | MilvusSemanticMemoryStore:
    backend = getattr(settings, "semantic_memory_backend", "disabled").casefold()
    if backend in {"", "disabled", "none"}:
        return NoopSemanticMemoryStore()
    if backend == "sqlite":
        return SQLiteSemanticMemoryStore(settings.semantic_memory_db)
    if backend == "milvus":
        return MilvusSemanticMemoryStore(
            uri=settings.milvus_memory_uri,
            collection_name=settings.milvus_memory_collection,
            dimension=settings.memory_embedding_dimension,
        )
    raise ValueError(f"Unsupported SEMANTIC_MEMORY_BACKEND: {settings.semantic_memory_backend}")
