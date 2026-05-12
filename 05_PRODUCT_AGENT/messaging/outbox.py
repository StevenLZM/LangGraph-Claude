from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from messaging.events import RocketMQEvent


class MessageOutboxStore:
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

