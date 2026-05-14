from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from api.settings import Settings
from api.idempotency import ChatRequestStore, PostgresChatRequestStore, build_chat_request_store
from memory.factory import build_session_store, build_user_memory_manager
from memory.long_term import PostgresUserMemoryManager, UserMemoryManager
from memory.session_store import PostgresSessionStore, SessionStore
from messaging.outbox import MessageOutboxStore, PostgresMessageOutboxStore, build_message_outbox_store


class FakeCursor:
    def __init__(self, *, row=None, rows=None, rowcount: int = 0) -> None:
        self._row = row
        self._rows = list(rows or [])
        self.rowcount = rowcount

    def fetchone(self):
        return self._row

    def fetchall(self):
        return self._rows


class FakePostgresConnection:
    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple]] = []
        self.sessions: dict[str, dict] = {}
        self.memories: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def execute(self, sql: str, params: tuple = ()) -> FakeCursor:
        self.statements.append((sql, params))
        normalized = " ".join(sql.split()).lower()

        if normalized.startswith("insert into sessions"):
            session_id, user_id, messages_json, metadata_json = params
            current = self.sessions.get(session_id)
            self.sessions[session_id] = {
                "session_id": session_id,
                "user_id": user_id,
                "messages_json": messages_json,
                "metadata_json": metadata_json,
                "version": int(current["version"]) + 1 if current else 1,
                "updated_at": "2026-05-10 10:00:00+00",
            }
            return FakeCursor(rowcount=1)

        if "from sessions" in normalized and "where session_id" in normalized:
            return FakeCursor(row=self.sessions.get(params[0]))

        if "from sessions" in normalized and "order by updated_at" in normalized:
            return FakeCursor(rows=list(self.sessions.values()))

        if normalized.startswith("insert into user_memories"):
            user_id, category, content = params
            if any(item["user_id"] == user_id and item["content"] == content for item in self.memories):
                return FakeCursor(rowcount=0)
            self.memories.append(
                {
                    "id": len(self.memories) + 1,
                    "user_id": user_id,
                    "category": category,
                    "content": content,
                    "created_at": "2026-05-10 10:00:00+00",
                }
            )
            return FakeCursor(rowcount=1)

        if normalized.startswith("delete from user_memories"):
            before = len(self.memories)
            self.memories = [item for item in self.memories if item["user_id"] != params[0]]
            return FakeCursor(rowcount=before - len(self.memories))

        if "from user_memories" in normalized and "where user_id" in normalized:
            rows = [item for item in self.memories if item["user_id"] == params[0]]
            rows.sort(key=lambda item: (item["created_at"], item["id"]), reverse=True)
            return FakeCursor(rows=rows[: params[1]] if "limit" in normalized else rows)

        return FakeCursor()


def test_storage_factory_uses_sqlite_by_default(tmp_path):
    settings = Settings(_env_file=None, memory_db=str(tmp_path / "memory.db"))

    assert isinstance(build_session_store(settings), SessionStore)
    assert isinstance(build_user_memory_manager(settings), UserMemoryManager)
    assert isinstance(build_chat_request_store(settings), ChatRequestStore)
    assert isinstance(build_message_outbox_store(settings), MessageOutboxStore)


def test_storage_factory_uses_postgres_when_configured():
    settings = Settings(
        _env_file=None,
        storage_backend="postgres",
        database_url="postgresql://customer_service:secret@postgres/customer_service",
    )

    assert isinstance(build_session_store(settings), PostgresSessionStore)
    assert isinstance(build_user_memory_manager(settings), PostgresUserMemoryManager)
    assert isinstance(build_chat_request_store(settings), PostgresChatRequestStore)
    assert isinstance(build_message_outbox_store(settings), PostgresMessageOutboxStore)


def test_postgres_storage_requires_database_url():
    settings = Settings(_env_file=None, storage_backend="postgres", database_url="")

    with pytest.raises(ValueError, match="DATABASE_URL"):
        build_session_store(settings)

    with pytest.raises(ValueError, match="DATABASE_URL"):
        build_user_memory_manager(settings)


def test_postgres_session_store_preserves_existing_session_contract():
    fake = FakePostgresConnection()
    store = PostgresSessionStore("postgresql://db", connect=lambda: fake)

    store.save_session(
        session_id="session_001",
        user_id="user_001",
        messages=[HumanMessage(content="你好"), AIMessage(content="你好，我是客服")],
        metadata={"summary": "首轮问候", "needs_human_transfer": False},
    )
    loaded = store.load_session("session_001")

    assert loaded is not None
    assert loaded["session_id"] == "session_001"
    assert loaded["user_id"] == "user_001"
    assert loaded["metadata"]["summary"] == "首轮问候"
    assert loaded["messages"][0].content == "你好"
    assert loaded["messages"][1].content == "你好，我是客服"


def test_postgres_user_memory_manager_preserves_existing_memory_contract():
    fake = FakePostgresConnection()
    manager = PostgresUserMemoryManager("postgresql://db", connect=lambda: fake)

    saved = manager.save_from_turn(
        user_id="user_001",
        user_message="我喜欢顺丰配送，以后发货优先顺丰。",
        assistant_answer="已记住你的配送偏好。",
    )
    memories = manager.load_memories("user_001", "你记得我的配送偏好吗？")
    listed = manager.list_memories("user_001")
    deleted = manager.delete_memories("user_001")

    assert saved == 1
    assert memories == ["用户偏好：我喜欢顺丰配送，以后发货优先顺丰。"]
    assert listed[0]["category"] == "delivery_preference"
    assert deleted == 1
