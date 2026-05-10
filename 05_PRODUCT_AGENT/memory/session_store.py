from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


def message_to_dict(message: BaseMessage) -> dict[str, Any]:
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    elif isinstance(message, SystemMessage):
        role = "system"
    else:
        role = getattr(message, "type", "message")
    return {
        "role": role,
        "content": str(message.content),
        "additional_kwargs": dict(getattr(message, "additional_kwargs", {}) or {}),
    }


def message_from_dict(payload: dict[str, Any]) -> BaseMessage:
    role = payload.get("role", "")
    content = payload.get("content", "")
    additional_kwargs = payload.get("additional_kwargs") or {}
    if role == "user":
        return HumanMessage(content=content, additional_kwargs=additional_kwargs)
    if role == "assistant":
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    return SystemMessage(content=content, additional_kwargs=additional_kwargs)


class SessionStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensure_schema()

    def save_session(
        self,
        *,
        session_id: str,
        user_id: str,
        messages: list[BaseMessage],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata = dict(metadata or {})
        message_payload = [message_to_dict(message) for message in messages]
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(session_id, user_id, messages_json, metadata_json, updated_at)
                VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id=excluded.user_id,
                    messages_json=excluded.messages_json,
                    metadata_json=excluded.metadata_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    session_id,
                    user_id,
                    json.dumps(message_payload, ensure_ascii=False),
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )

    def load_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id, user_id, messages_json, metadata_json, updated_at
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        messages = [message_from_dict(item) for item in json.loads(row["messages_json"] or "[]")]
        metadata = json.loads(row["metadata_json"] or "{}")
        return {
            "session_id": row["session_id"],
            "user_id": row["user_id"],
            "messages": messages,
            "metadata": metadata,
            "updated_at": row["updated_at"],
        }

    def get_public_session(self, session_id: str) -> dict[str, Any] | None:
        loaded = self.load_session(session_id)
        if loaded is None:
            return None
        messages = loaded["messages"]
        metadata = loaded["metadata"]
        return {
            "session_id": loaded["session_id"],
            "user_id": loaded["user_id"],
            "messages": [message_to_dict(message) for message in messages],
            "window_size": len(messages),
            "total_turns": sum(1 for message in messages if isinstance(message, HumanMessage)),
            "summary": metadata.get("summary", ""),
            "needs_human_transfer": metadata.get("needs_human_transfer", False),
            "transfer_reason": metadata.get("transfer_reason", ""),
            "token_used": metadata.get("token_used", 0),
            "quality_score": metadata.get("quality_score"),
            "quality_evaluation": metadata.get("quality_evaluation"),
            "quality_alert": metadata.get("quality_alert", False),
            "updated_at": loaded["updated_at"],
        }

    def list_public_sessions(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, user_id, messages_json, metadata_json, updated_at
                FROM sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        sessions = []
        for row in rows:
            messages = [message_from_dict(item) for item in json.loads(row["messages_json"] or "[]")]
            metadata = json.loads(row["metadata_json"] or "{}")
            sessions.append(
                {
                    "session_id": row["session_id"],
                    "user_id": row["user_id"],
                    "updated_at": row["updated_at"],
                    "window_size": len(messages),
                    "total_turns": sum(1 for message in messages if isinstance(message, HumanMessage)),
                    "needs_human_transfer": metadata.get("needs_human_transfer", False),
                    "transfer_reason": metadata.get("transfer_reason", ""),
                    "quality_score": metadata.get("quality_score"),
                    "token_used": metadata.get("token_used", 0),
                    "quality_alert": metadata.get("quality_alert", False),
                }
            )
        return sessions

    def summarize_transfers(self) -> dict[str, Any]:
        sessions = self.list_public_sessions(limit=10000)
        transfer_reasons: dict[str, int] = {}
        quality_scores = []
        token_total = 0
        low_quality_count = 0
        for session in sessions:
            if session["needs_human_transfer"]:
                reason = session["transfer_reason"] or "未记录原因"
                transfer_reasons[reason] = transfer_reasons.get(reason, 0) + 1
            score = session.get("quality_score")
            if isinstance(score, int):
                quality_scores.append(score)
                if score < 70:
                    low_quality_count += 1
            token_total += int(session.get("token_used") or 0)
        return {
            "total_sessions": len(sessions),
            "human_transfer_count": sum(transfer_reasons.values()),
            "transfer_reasons": transfer_reasons,
            "low_quality_count": low_quality_count,
            "average_quality_score": round(sum(quality_scores) / len(quality_scores), 1)
            if quality_scores
            else 0.0,
            "token_total": token_total,
        }

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
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )


class PostgresSessionStore:
    def __init__(self, database_url: str, *, connect: Callable[[], Any] | None = None) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL is required for Postgres session storage")
        self.database_url = database_url
        self._connect_factory = connect
        self._schema_ready = False

    def save_session(
        self,
        *,
        session_id: str,
        user_id: str,
        messages: list[BaseMessage],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._ensure_schema_once()
        metadata = dict(metadata or {})
        message_payload = [message_to_dict(message) for message in messages]
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(session_id, user_id, messages_json, metadata_json, updated_at)
                VALUES(%s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id=EXCLUDED.user_id,
                    messages_json=EXCLUDED.messages_json,
                    metadata_json=EXCLUDED.metadata_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    session_id,
                    user_id,
                    json.dumps(message_payload, ensure_ascii=False),
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )

    def load_session(self, session_id: str) -> dict[str, Any] | None:
        self._ensure_schema_once()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id, user_id, messages_json, metadata_json, updated_at
                FROM sessions
                WHERE session_id = %s
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        messages = [message_from_dict(item) for item in json.loads(row["messages_json"] or "[]")]
        metadata = json.loads(row["metadata_json"] or "{}")
        return {
            "session_id": row["session_id"],
            "user_id": row["user_id"],
            "messages": messages,
            "metadata": metadata,
            "updated_at": str(row["updated_at"]),
        }

    def get_public_session(self, session_id: str) -> dict[str, Any] | None:
        loaded = self.load_session(session_id)
        if loaded is None:
            return None
        messages = loaded["messages"]
        metadata = loaded["metadata"]
        return {
            "session_id": loaded["session_id"],
            "user_id": loaded["user_id"],
            "messages": [message_to_dict(message) for message in messages],
            "window_size": len(messages),
            "total_turns": sum(1 for message in messages if isinstance(message, HumanMessage)),
            "summary": metadata.get("summary", ""),
            "needs_human_transfer": metadata.get("needs_human_transfer", False),
            "transfer_reason": metadata.get("transfer_reason", ""),
            "token_used": metadata.get("token_used", 0),
            "quality_score": metadata.get("quality_score"),
            "quality_evaluation": metadata.get("quality_evaluation"),
            "quality_alert": metadata.get("quality_alert", False),
            "updated_at": loaded["updated_at"],
        }

    def list_public_sessions(self, *, limit: int = 100) -> list[dict[str, Any]]:
        self._ensure_schema_once()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, user_id, messages_json, metadata_json, updated_at
                FROM sessions
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        sessions = []
        for row in rows:
            messages = [message_from_dict(item) for item in json.loads(row["messages_json"] or "[]")]
            metadata = json.loads(row["metadata_json"] or "{}")
            sessions.append(
                {
                    "session_id": row["session_id"],
                    "user_id": row["user_id"],
                    "updated_at": str(row["updated_at"]),
                    "window_size": len(messages),
                    "total_turns": sum(1 for message in messages if isinstance(message, HumanMessage)),
                    "needs_human_transfer": metadata.get("needs_human_transfer", False),
                    "transfer_reason": metadata.get("transfer_reason", ""),
                    "quality_score": metadata.get("quality_score"),
                    "token_used": metadata.get("token_used", 0),
                    "quality_alert": metadata.get("quality_alert", False),
                }
            )
        return sessions

    def summarize_transfers(self) -> dict[str, Any]:
        sessions = self.list_public_sessions(limit=10000)
        transfer_reasons: dict[str, int] = {}
        quality_scores = []
        token_total = 0
        low_quality_count = 0
        for session in sessions:
            if session["needs_human_transfer"]:
                reason = session["transfer_reason"] or "未记录原因"
                transfer_reasons[reason] = transfer_reasons.get(reason, 0) + 1
            score = session.get("quality_score")
            if isinstance(score, int):
                quality_scores.append(score)
                if score < 70:
                    low_quality_count += 1
            token_total += int(session.get("token_used") or 0)
        return {
            "total_sessions": len(sessions),
            "human_transfer_count": sum(transfer_reasons.values()),
            "transfer_reasons": transfer_reasons,
            "low_quality_count": low_quality_count,
            "average_quality_score": round(sum(quality_scores) / len(quality_scores), 1)
            if quality_scores
            else 0.0,
            "token_total": token_total,
        }

    def _connect(self) -> Any:
        if self._connect_factory is not None:
            return self._connect_factory()
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("psycopg is required for Postgres session storage") from exc
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _ensure_schema_once(self) -> None:
        if self._schema_ready:
            return
        self._ensure_schema()
        self._schema_ready = True
