from __future__ import annotations

from contextlib import ExitStack
from typing import Any


_CHECKPOINTER_STACK = ExitStack()


def build_checkpointer(
    settings: Any,
    *,
    postgres_saver_cls: Any | None = None,
    redis_saver_cls: Any | None = None,
) -> Any | None:
    backend = settings.checkpointer_backend.casefold()
    if backend == "none":
        return None
    if backend == "postgres":
        return _build_postgres_checkpointer(settings, postgres_saver_cls=postgres_saver_cls)
    if backend == "redis":
        return _build_redis_checkpointer(settings, redis_saver_cls=redis_saver_cls)
    raise ValueError(f"Unsupported CHECKPOINTER_BACKEND: {settings.checkpointer_backend}")


def close_checkpointer_resources() -> None:
    _CHECKPOINTER_STACK.close()


def _build_postgres_checkpointer(settings: Any, *, postgres_saver_cls: Any | None) -> Any:
    url = settings.checkpointer_url or settings.database_url
    if not url:
        raise ValueError("DATABASE_URL or CHECKPOINTER_URL is required when CHECKPOINTER_BACKEND=postgres")
    saver_cls = postgres_saver_cls or _import_postgres_saver()
    saver = _CHECKPOINTER_STACK.enter_context(saver_cls.from_conn_string(url))
    if settings.checkpointer_setup:
        saver.setup()
    return saver


def _build_redis_checkpointer(settings: Any, *, redis_saver_cls: Any | None) -> Any:
    url = settings.checkpointer_url or settings.redis_url
    if not url:
        raise ValueError("REDIS_URL or CHECKPOINTER_URL is required when CHECKPOINTER_BACKEND=redis")
    saver_cls = redis_saver_cls or _import_redis_saver()
    saver = _CHECKPOINTER_STACK.enter_context(saver_cls.from_conn_string(url))
    if settings.checkpointer_setup:
        saver.setup()
    return saver


def _import_postgres_saver() -> Any:
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("langgraph-checkpoint-postgres is required for Postgres checkpointing") from exc
    return PostgresSaver


def _import_redis_saver() -> Any:
    try:
        from langgraph.checkpoint.redis import RedisSaver
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("langgraph-checkpoint-redis is required for Redis checkpointing") from exc
    return RedisSaver
