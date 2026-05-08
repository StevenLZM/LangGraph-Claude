from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from memory.long_term import UserMemoryManager
from memory.session_store import SessionStore
from memory.short_term import ContextWindowManager


def test_context_window_summarizes_old_messages_and_keeps_recent_turns():
    messages = []
    for index in range(100):
        messages.append(HumanMessage(content=f"用户第 {index} 轮：我想继续查询订单和物流状态。"))
        messages.append(AIMessage(content=f"客服第 {index} 轮：这是对应的处理结果。"))

    manager = ContextWindowManager(max_messages=16, max_tokens=260)
    trimmed = manager.trim(messages)

    assert isinstance(trimmed[0], SystemMessage)
    assert "早期对话摘要" in trimmed[0].content
    assert len(trimmed) <= 17
    assert trimmed[-1].content == messages[-1].content
    assert manager.count_tokens(trimmed) <= 260


def test_session_store_persists_messages_across_instances(tmp_path):
    db_path = tmp_path / "sessions.db"
    first_store = SessionStore(str(db_path))
    first_store.save_session(
        session_id="session_001",
        user_id="user_001",
        messages=[HumanMessage(content="你好"), AIMessage(content="你好，我是客服")],
        metadata={"needs_human_transfer": False, "summary": "首轮问候"},
    )

    second_store = SessionStore(str(db_path))
    loaded = second_store.load_session("session_001")

    assert loaded is not None
    assert loaded["session_id"] == "session_001"
    assert loaded["user_id"] == "user_001"
    assert loaded["metadata"]["summary"] == "首轮问候"
    assert loaded["messages"][0].content == "你好"
    assert loaded["messages"][1].content == "你好，我是客服"


def test_user_memory_manager_saves_searches_and_deletes_memories(tmp_path):
    db_path = tmp_path / "memories.db"
    manager = UserMemoryManager(str(db_path))

    saved = manager.save_from_turn(
        user_id="user_001",
        user_message="我喜欢顺丰配送，以后发货优先顺丰。",
        assistant_answer="已记住你的配送偏好。",
    )
    memories = manager.load_memories("user_001", "你记得我的配送偏好吗？")

    assert saved == 1
    assert memories
    assert "顺丰" in memories[0]

    recall_saved = manager.save_from_turn(
        user_id="user_001",
        user_message="你记得我的配送偏好吗？",
        assistant_answer="我记得你的偏好。",
    )

    assert recall_saved == 0

    deleted = manager.delete_memories("user_001")

    assert deleted == 1
    assert manager.load_memories("user_001", "配送偏好") == []


def test_delivery_preference_replaces_previous_delivery_preference(tmp_path):
    db_path = tmp_path / "memories.db"
    manager = UserMemoryManager(str(db_path))

    first_saved = manager.save_from_turn(
        user_id="user_001",
        user_message="我喜欢顺丰配送，以后发货优先顺丰。",
        assistant_answer="已记住你的配送偏好。",
    )
    second_saved = manager.save_from_turn(
        user_id="user_001",
        user_message="以后给我发货优先京东物流。",
        assistant_answer="已更新你的配送偏好。",
    )
    memories = manager.load_memories("user_001", "你记得我的配送偏好吗？")

    assert first_saved == 1
    assert second_saved == 1
    assert len(memories) == 1
    assert "京东物流" in memories[0]
    assert "顺丰" not in memories[0]
