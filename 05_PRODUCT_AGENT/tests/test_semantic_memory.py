from __future__ import annotations

from memory.semantic import SQLiteSemanticMemoryStore, StructuredMemory


def test_sqlite_semantic_memory_store_upserts_structured_memory_and_searches(tmp_path):
    store = SQLiteSemanticMemoryStore(str(tmp_path / "semantic.db"))
    memory = StructuredMemory(
        user_id="user_001",
        category="delivery_preference",
        entity_type="preference",
        key="shipping_carrier",
        value={"carrier": "顺丰"},
        content="用户偏好：以后发货优先顺丰",
        source_session_id="session_001",
        source_request_id="req_001",
        confidence=0.88,
    )

    saved = store.upsert(memory)
    results = store.search(
        user_id="user_001",
        query="给我送货用什么快递？",
        filters={"category": ["delivery_preference"]},
    )

    assert saved == 1
    assert len(results) == 1
    assert results[0].content == "用户偏好：以后发货优先顺丰"
    assert results[0].score > 0
    assert results[0].memory.value == {"carrier": "顺丰"}


def test_sqlite_semantic_memory_store_deletes_user_memories(tmp_path):
    store = SQLiteSemanticMemoryStore(str(tmp_path / "semantic.db"))
    store.upsert(
        StructuredMemory(
            user_id="user_001",
            category="preference",
            entity_type="preference",
            key="product",
            value={"product": "AirBuds Pro 2"},
            content="用户偏好：我喜欢 AirBuds Pro 2",
            source_session_id="session_001",
            source_request_id="req_001",
        )
    )

    assert store.delete_user("user_001") == 1
    assert store.search(user_id="user_001", query="AirBuds") == []
