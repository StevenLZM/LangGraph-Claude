from __future__ import annotations

from typing import Any

from memory.long_term import PostgresUserMemoryManager, UserMemoryManager
from memory.session_store import PostgresSessionStore, SessionStore


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
