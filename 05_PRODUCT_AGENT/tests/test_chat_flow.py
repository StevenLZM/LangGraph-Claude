from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def _chat(message: str, session_id: str = "session_001") -> dict:
    client = TestClient(app)
    response = client.post(
        "/chat",
        json={
            "user_id": "user_001",
            "session_id": session_id,
            "message": message,
        },
    )
    assert response.status_code == 200
    return response.json()


def _chat_for_user(user_id: str, session_id: str, message: str) -> dict:
    client = TestClient(app)
    response = client.post(
        "/chat",
        json={
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
        },
    )
    assert response.status_code == 200
    return response.json()


def test_chat_answers_order_status():
    payload = _chat("我的订单 ORD123456 到哪了？")

    assert payload["session_id"] == "session_001"
    assert "ORD123456" in payload["answer"]
    assert "派送" in payload["answer"]
    assert payload["needs_human_transfer"] is False
    assert payload["transfer_reason"] == ""
    assert payload["order_context"]["order_id"] == "ORD123456"
    assert payload["quality_score"] >= 80


def test_chat_answers_logistics_status():
    payload = _chat("帮我查一下物流 ORD123456")

    assert "顺丰" in payload["answer"]
    assert "上海浦东配送站" in payload["answer"]
    assert payload["order_context"]["tracking_no"] == "SF100200300CN"


def test_chat_answers_product_question():
    payload = _chat("AirBuds Pro 2 还有库存吗？")

    assert "AirBuds Pro 2" in payload["answer"]
    assert "有货" in payload["answer"]
    assert payload["needs_human_transfer"] is False


def test_refund_requires_explicit_confirmation_before_submit():
    first = _chat("我要给订单 ORD123456 退款", session_id="refund_session")
    assert "确认" in first["answer"]
    assert "已提交" not in first["answer"]
    assert first["order_context"]["refund_status"] == "confirmation_required"

    confirmed = _chat("我确认退款 ORD123456", session_id="refund_session")
    assert "退款申请已提交" in confirmed["answer"]
    assert confirmed["order_context"]["refund_status"] == "submitted"


def test_chat_marks_human_transfer_for_complaint_and_legal_issue():
    payload = _chat("我要投诉，最好让法务处理，给我转人工")

    assert payload["needs_human_transfer"] is True
    assert "人工" in payload["answer"]
    assert payload["transfer_reason"] == "用户要求人工或涉及投诉/法律问题"


def test_chat_handles_mixed_english_and_chinese():
    payload = _chat("Where is my order ORD123456？谢谢")

    assert "ORD123456" in payload["answer"]
    assert payload["needs_human_transfer"] is False


def test_chat_rejects_empty_message():
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"user_id": "user_001", "session_id": "session_001", "message": "   "},
    )

    assert response.status_code == 422


def test_session_endpoint_returns_persisted_conversation():
    client = TestClient(app)
    session_id = "persisted_session_001"

    _chat("我的订单 ORD123456 到哪了？", session_id=session_id)
    response = client.get(f"/sessions/{session_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == session_id
    assert payload["user_id"] == "user_001"
    assert payload["total_turns"] == 1
    assert payload["window_size"] == 2
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][-1]["role"] == "assistant"
    assert "ORD123456" in payload["messages"][-1]["content"]


def test_user_memory_recalls_across_sessions_and_can_be_deleted():
    client = TestClient(app)
    user_id = "memory_user_001"

    first = _chat_for_user(user_id, "memory_seed_session", "我喜欢顺丰配送，以后发货优先顺丰")
    assert "已记住" in first["answer"]

    recalled = _chat_for_user(user_id, "memory_recall_session", "你记得我的配送偏好吗？")
    assert "顺丰" in recalled["answer"]
    assert recalled["user_memories"]

    delete_response = client.delete(f"/users/{user_id}/memories")
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] >= 1

    after_delete = _chat_for_user(user_id, "memory_after_delete_session", "你记得我的配送偏好吗？")
    assert after_delete["user_memories"] == []
    assert "顺丰" not in after_delete["answer"]
    assert "暂时没有" in after_delete["answer"]
