from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from messaging.events import RocketMQEvent


class MessageOutboxStore:
    backend = "sqlite"

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensure_schema()

    def enqueue(self, event: RocketMQEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO message_outbox(
                    event_id, event_type, topic, tag, aggregate_id, event_json,
                    status, attempts, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, 'pending', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    event.event_id,
                    event.event_type,
                    event.topic,
                    event.tag,
                    event.aggregate_id,
                    event.to_json(),
                ),
            )

    def mark_published(self, event_id: str, *, message_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE message_outbox
                SET status='published',
                    message_id=?,
                    last_error='',
                    attempts=attempts + 1,
                    updated_at=CURRENT_TIMESTAMP
                WHERE event_id=?
                """,
                (message_id, event_id),
            )

    def mark_failed(self, event_id: str, *, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE message_outbox
                SET status='enqueue_failed',
                    last_error=?,
                    attempts=attempts + 1,
                    updated_at=CURRENT_TIMESTAMP
                WHERE event_id=?
                """,
                (error, event_id),
            )

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT event_id, event_type, topic, tag, aggregate_id, event_json,
                       status, attempts, message_id, last_error, created_at, updated_at
                FROM message_outbox
                WHERE event_id=?
                """,
                (event_id,),
            ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["event"] = json.loads(payload["event_json"])
        return payload

    def list_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event_id, event_type, topic, tag, aggregate_id, status,
                       attempts, message_id, last_error, created_at, updated_at
                FROM message_outbox
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        if self.db_path != ":memory:":
            Path(self.db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS message_outbox (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    aggregate_id TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    message_id TEXT NOT NULL DEFAULT '',
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )


class PostgresMessageOutboxStore:
    backend = "postgres"

    def __init__(self, database_url: str, *, connect: Any | None = None) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL is required for Postgres message outbox storage")
        self.database_url = database_url
        self._connect_factory = connect
        self._schema_ready = False

    def enqueue(self, event: RocketMQEvent) -> None:
        self._ensure_schema_once()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO message_outbox(
                    event_id, event_type, topic, tag, aggregate_id, event_json,
                    status, attempts, created_at, updated_at
                )
                VALUES(%s, %s, %s, %s, %s, %s, 'pending', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(event_id) DO NOTHING
                """,
                (
                    event.event_id,
                    event.event_type,
                    event.topic,
                    event.tag,
                    event.aggregate_id,
                    event.to_json(),
                ),
            )

    def mark_published(self, event_id: str, *, message_id: str) -> None:
        self._ensure_schema_once()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE message_outbox
                SET status='published',
                    message_id=%s,
                    last_error='',
                    attempts=attempts + 1,
                    updated_at=CURRENT_TIMESTAMP
                WHERE event_id=%s
                """,
                (message_id, event_id),
            )

    def mark_failed(self, event_id: str, *, error: str) -> None:
        self._ensure_schema_once()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE message_outbox
                SET status='enqueue_failed',
                    last_error=%s,
                    attempts=attempts + 1,
                    updated_at=CURRENT_TIMESTAMP
                WHERE event_id=%s
                """,
                (error, event_id),
            )

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        self._ensure_schema_once()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT event_id, event_type, topic, tag, aggregate_id, event_json,
                       status, attempts, message_id, last_error, created_at, updated_at
                FROM message_outbox
                WHERE event_id=%s
                """,
                (event_id,),
            ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["event"] = json.loads(payload["event_json"])
        return payload

    def list_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        self._ensure_schema_once()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event_id, event_type, topic, tag, aggregate_id, status,
                       attempts, message_id, last_error, created_at, updated_at
                FROM message_outbox
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _connect(self) -> Any:
        if self._connect_factory is not None:
            return self._connect_factory()
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("psycopg is required for Postgres message outbox storage") from exc
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS message_outbox (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    aggregate_id TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    message_id TEXT NOT NULL DEFAULT '',
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _ensure_schema_once(self) -> None:
        if self._schema_ready:
            return
        self._ensure_schema()
        self._schema_ready = True


def build_message_outbox_store(settings: Any) -> MessageOutboxStore | PostgresMessageOutboxStore:
    backend = settings.storage_backend.casefold()
    if backend == "sqlite":
        return MessageOutboxStore(settings.message_outbox_db)
    if backend == "postgres":
        if not settings.database_url:
            raise ValueError("DATABASE_URL is required when STORAGE_BACKEND=postgres")
        return PostgresMessageOutboxStore(settings.database_url)
    raise ValueError(f"Unsupported STORAGE_BACKEND: {settings.storage_backend}")
