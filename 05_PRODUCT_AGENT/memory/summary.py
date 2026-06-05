from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


@dataclass(frozen=True)
class SummaryMemoryRecord:
    session_id: str
    user_id: str
    summary: str
    version: int
    covered_turns: int
    source_event_id: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "summary": self.summary,
            "version": self.version,
            "covered_turns": self.covered_turns,
            "source_event_id": self.source_event_id,
            "updated_at": self.updated_at,
        }


class SummaryMemoryStore:
    backend = "sqlite"

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensure_schema()

    def load_summary(self, session_id: str) -> SummaryMemoryRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id, user_id, summary, version, covered_turns, source_event_id, updated_at
                FROM session_summaries
                WHERE session_id=?
                """,
                (session_id,),
            ).fetchone()
        return _record_from_row(row) if row is not None else None

    def save_summary(
        self,
        *,
        session_id: str,
        user_id: str,
        summary: str,
        covered_turns: int,
        source_event_id: str = "",
    ) -> SummaryMemoryRecord:
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT version FROM session_summaries WHERE session_id=?",
                (session_id,),
            ).fetchone()
            next_version = int(existing["version"]) + 1 if existing else 1
            conn.execute(
                """
                INSERT INTO session_summaries(
                    session_id, user_id, summary, version, covered_turns, source_event_id, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id=excluded.user_id,
                    summary=excluded.summary,
                    version=excluded.version,
                    covered_turns=excluded.covered_turns,
                    source_event_id=excluded.source_event_id,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (session_id, user_id, summary, next_version, covered_turns, source_event_id),
            )
        loaded = self.load_summary(session_id)
        if loaded is None:  # pragma: no cover - defensive
            raise RuntimeError(f"summary for session {session_id} was not saved")
        return loaded

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
                CREATE TABLE IF NOT EXISTS session_summaries (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    covered_turns INTEGER NOT NULL DEFAULT 0,
                    source_event_id TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )


class PostgresSummaryMemoryStore:
    backend = "postgres"

    def __init__(self, database_url: str, *, connect: Callable[[], Any] | None = None) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL is required for Postgres summary memory storage")
        self.database_url = database_url
        self._connect_factory = connect
        self._schema_ready = False

    def load_summary(self, session_id: str) -> SummaryMemoryRecord | None:
        self._ensure_schema_once()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id, user_id, summary, version, covered_turns, source_event_id, updated_at
                FROM session_summaries
                WHERE session_id=%s
                """,
                (session_id,),
            ).fetchone()
        return _record_from_row(row) if row is not None else None

    def save_summary(
        self,
        *,
        session_id: str,
        user_id: str,
        summary: str,
        covered_turns: int,
        source_event_id: str = "",
    ) -> SummaryMemoryRecord:
        self._ensure_schema_once()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT version FROM session_summaries WHERE session_id=%s",
                (session_id,),
            ).fetchone()
            next_version = int(existing["version"]) + 1 if existing else 1
            conn.execute(
                """
                INSERT INTO session_summaries(
                    session_id, user_id, summary, version, covered_turns, source_event_id, updated_at
                )
                VALUES(%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id=EXCLUDED.user_id,
                    summary=EXCLUDED.summary,
                    version=EXCLUDED.version,
                    covered_turns=EXCLUDED.covered_turns,
                    source_event_id=EXCLUDED.source_event_id,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (session_id, user_id, summary, next_version, covered_turns, source_event_id),
            )
        loaded = self.load_summary(session_id)
        if loaded is None:  # pragma: no cover - defensive
            raise RuntimeError(f"summary for session {session_id} was not saved")
        return loaded

    def _connect(self) -> Any:
        if self._connect_factory is not None:
            return self._connect_factory()
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("psycopg is required for Postgres summary memory storage") from exc
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_summaries (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    covered_turns INTEGER NOT NULL DEFAULT 0,
                    source_event_id TEXT NOT NULL DEFAULT '',
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _ensure_schema_once(self) -> None:
        if self._schema_ready:
            return
        self._ensure_schema()
        self._schema_ready = True


def build_conversation_summary(messages: list[BaseMessage], *, limit: int = 240) -> str:
    lines = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "用户"
        elif isinstance(message, AIMessage):
            role = "客服"
        else:
            continue
        content = " ".join(str(message.content).split())
        if content:
            lines.append(f"{role}: {content}")
    if not lines:
        return ""
    summary = "；".join(lines[-8:])
    return summary if len(summary) <= limit else f"{summary[: limit - 3]}..."


def _record_from_row(row: Any) -> SummaryMemoryRecord:
    return SummaryMemoryRecord(
        session_id=str(row["session_id"]),
        user_id=str(row["user_id"]),
        summary=str(row["summary"]),
        version=int(row["version"]),
        covered_turns=int(row["covered_turns"]),
        source_event_id=str(row["source_event_id"] or ""),
        updated_at=str(row["updated_at"] or ""),
    )
