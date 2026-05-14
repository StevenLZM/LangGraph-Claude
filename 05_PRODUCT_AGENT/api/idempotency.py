from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChatRequestRecord:
    user_id: str
    session_id: str
    request_id: str
    message_hash: str
    status: str
    response_json: str = ""
    error_status_code: int = 0
    error_detail_json: str = ""
    created_at: str = ""
    updated_at: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.status in {"succeeded", "failed"}


@dataclass(frozen=True)
class StartedChatRequest:
    acquired: bool
    record: ChatRequestRecord


class IdempotencyConflict(Exception):
    """Raised when the same request_id is reused for a different message."""


def chat_message_hash(*, user_id: str, session_id: str, message: str) -> str:
    payload = json.dumps(
        {
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ChatRequestStore:
    backend = "sqlite"

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensure_schema()

    def start_request(
        self,
        *,
        user_id: str,
        session_id: str,
        request_id: str,
        message_hash: str,
    ) -> StartedChatRequest:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO chat_requests(
                    user_id, session_id, request_id, message_hash, status,
                    response_json, error_status_code, error_detail_json,
                    created_at, updated_at
                )
                VALUES(?, ?, ?, ?, 'processing', '', 0, '', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (user_id, session_id, request_id, message_hash),
            )
            acquired = cursor.rowcount == 1
        record = self.get_request(
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
        )
        if record is None:  # pragma: no cover - defensive; insert or select should have produced a row
            raise RuntimeError("chat request record was not created")
        if record.message_hash != message_hash:
            raise IdempotencyConflict
        return StartedChatRequest(acquired=acquired, record=record)

    def complete_success(
        self,
        *,
        user_id: str,
        session_id: str,
        request_id: str,
        response: dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE chat_requests
                SET status='succeeded',
                    response_json=?,
                    error_status_code=0,
                    error_detail_json='',
                    updated_at=CURRENT_TIMESTAMP
                WHERE user_id=? AND session_id=? AND request_id=?
                """,
                (
                    json.dumps(response, ensure_ascii=False),
                    user_id,
                    session_id,
                    request_id,
                ),
            )

    def complete_failure(
        self,
        *,
        user_id: str,
        session_id: str,
        request_id: str,
        status_code: int,
        detail: Any,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE chat_requests
                SET status='failed',
                    response_json='',
                    error_status_code=?,
                    error_detail_json=?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE user_id=? AND session_id=? AND request_id=?
                """,
                (
                    int(status_code),
                    json.dumps(detail, ensure_ascii=False),
                    user_id,
                    session_id,
                    request_id,
                ),
            )

    def get_request(
        self,
        *,
        user_id: str,
        session_id: str,
        request_id: str,
    ) -> ChatRequestRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT user_id, session_id, request_id, message_hash, status,
                       response_json, error_status_code, error_detail_json,
                       created_at, updated_at
                FROM chat_requests
                WHERE user_id=? AND session_id=? AND request_id=?
                """,
                (user_id, session_id, request_id),
            ).fetchone()
        if row is None:
            return None
        return ChatRequestRecord(**dict(row))

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
                CREATE TABLE IF NOT EXISTS chat_requests (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    message_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_json TEXT NOT NULL DEFAULT '',
                    error_status_code INTEGER NOT NULL DEFAULT 0,
                    error_detail_json TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(user_id, session_id, request_id)
                )
                """
            )


class PostgresChatRequestStore:
    backend = "postgres"

    def __init__(self, database_url: str, *, connect: Any | None = None) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL is required for Postgres request storage")
        self.database_url = database_url
        self._connect_factory = connect
        self._schema_ready = False

    def start_request(
        self,
        *,
        user_id: str,
        session_id: str,
        request_id: str,
        message_hash: str,
    ) -> StartedChatRequest:
        self._ensure_schema_once()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO chat_requests(
                    user_id, session_id, request_id, message_hash, status,
                    response_json, error_status_code, error_detail_json,
                    created_at, updated_at
                )
                VALUES(%s, %s, %s, %s, 'processing', '', 0, '', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, session_id, request_id) DO NOTHING
                """,
                (user_id, session_id, request_id, message_hash),
            )
            acquired = cursor.rowcount == 1
        record = self.get_request(
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
        )
        if record is None:  # pragma: no cover - defensive
            raise RuntimeError("chat request record was not created")
        if record.message_hash != message_hash:
            raise IdempotencyConflict
        return StartedChatRequest(acquired=acquired, record=record)

    def complete_success(
        self,
        *,
        user_id: str,
        session_id: str,
        request_id: str,
        response: dict[str, Any],
    ) -> None:
        self._ensure_schema_once()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE chat_requests
                SET status='succeeded',
                    response_json=%s,
                    error_status_code=0,
                    error_detail_json='',
                    updated_at=CURRENT_TIMESTAMP
                WHERE user_id=%s AND session_id=%s AND request_id=%s
                """,
                (
                    json.dumps(response, ensure_ascii=False),
                    user_id,
                    session_id,
                    request_id,
                ),
            )

    def complete_failure(
        self,
        *,
        user_id: str,
        session_id: str,
        request_id: str,
        status_code: int,
        detail: Any,
    ) -> None:
        self._ensure_schema_once()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE chat_requests
                SET status='failed',
                    response_json='',
                    error_status_code=%s,
                    error_detail_json=%s,
                    updated_at=CURRENT_TIMESTAMP
                WHERE user_id=%s AND session_id=%s AND request_id=%s
                """,
                (
                    int(status_code),
                    json.dumps(detail, ensure_ascii=False),
                    user_id,
                    session_id,
                    request_id,
                ),
            )

    def get_request(
        self,
        *,
        user_id: str,
        session_id: str,
        request_id: str,
    ) -> ChatRequestRecord | None:
        self._ensure_schema_once()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT user_id, session_id, request_id, message_hash, status,
                       response_json, error_status_code, error_detail_json,
                       created_at, updated_at
                FROM chat_requests
                WHERE user_id=%s AND session_id=%s AND request_id=%s
                """,
                (user_id, session_id, request_id),
            ).fetchone()
        if row is None:
            return None
        return ChatRequestRecord(**dict(row))

    def _connect(self) -> Any:
        if self._connect_factory is not None:
            return self._connect_factory()
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("psycopg is required for Postgres request storage") from exc
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_requests (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    message_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_json TEXT NOT NULL DEFAULT '',
                    error_status_code INTEGER NOT NULL DEFAULT 0,
                    error_detail_json TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(user_id, session_id, request_id)
                )
                """
            )
            conn.execute(
                """
                ALTER TABLE chat_requests
                ADD COLUMN IF NOT EXISTS message_hash TEXT NOT NULL DEFAULT ''
                """
            )

    def _ensure_schema_once(self) -> None:
        if self._schema_ready:
            return
        self._ensure_schema()
        self._schema_ready = True


def build_chat_request_store(settings: Any) -> ChatRequestStore | PostgresChatRequestStore:
    backend = settings.storage_backend.casefold()
    if backend == "sqlite":
        return ChatRequestStore(settings.chat_request_db)
    if backend == "postgres":
        if not settings.database_url:
            raise ValueError("DATABASE_URL is required when STORAGE_BACKEND=postgres")
        return PostgresChatRequestStore(settings.database_url)
    raise ValueError(f"Unsupported STORAGE_BACKEND: {settings.storage_backend}")
