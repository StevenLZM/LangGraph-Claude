from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

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
            "updated_at": loaded["updated_at"],
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
