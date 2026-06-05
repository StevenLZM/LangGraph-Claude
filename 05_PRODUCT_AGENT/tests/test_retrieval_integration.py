from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

import api.main as main
from memory.semantic import SQLiteSemanticMemoryStore, StructuredMemory


def test_chat_uses_retrieval_decision_to_load_semantic_memory(tmp_path, monkeypatch):
    semantic_store = SQLiteSemanticMemoryStore(str(tmp_path / "semantic.db"))
    semantic_store.upsert(
        StructuredMemory(
            user_id="semantic_user_001",
            category="delivery_preference",
            entity_type="preference",
            key="shipping_preference",
            value={"carrier": "顺丰"},
            content="用户偏好：以后发货优先顺丰",
            source_session_id="seed_session",
            source_request_id="seed_request",
        )
    )
    monkeypatch.setattr(main, "semantic_memory_store", semantic_store, raising=False)

    response = TestClient(main.app).post(
        "/chat",
        json={
            "user_id": "semantic_user_001",
            "session_id": "semantic_session_001",
            "request_id": uuid.uuid4().hex,
            "message": "给我送货用什么快递？",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "顺丰" in payload["answer"]
    assert "用户偏好：以后发货优先顺丰" in payload["user_memories"]
    assert payload["llm_trace"]["retrieval_decision"]["intent"] == "logistics"
    assert payload["llm_trace"]["retrieval_decision"]["needs_memory"] is True
