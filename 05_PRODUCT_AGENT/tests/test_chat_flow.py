from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app


class FakeChatLLM:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0
        self.messages: list[list[object]] = []

    async def ainvoke(self, messages: list[object]) -> str:
        self.calls += 1
        self.messages.append(messages)
        return self.answer


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


def test_chat_uses_latest_delivery_preference_when_carrier_query_has_no_order_id():
    user_id = "delivery_preference_user_001"

    _chat_for_user(user_id, "delivery_preference_seed_1", "我喜欢顺丰配送，以后发货优先顺丰")
    _chat_for_user(user_id, "delivery_preference_seed_2", "以后给我发货优先京东物流")
    payload = _chat_for_user(user_id, "delivery_preference_query", "给我送货用什么快递？")

    assert "京东物流" in payload["answer"]
    assert "订单号" in payload["answer"]
    assert "顺丰" not in payload["answer"]
    assert payload["llm_trace"]["used_llm"] is False
    assert payload["llm_trace"]["tool_name"] == "delivery_preference"
    assert "已读取配送偏好" in payload["llm_trace"]["reasoning_summary"]


def test_replacing_delivery_preference_refreshes_response_memories():
    user_id = "delivery_preference_refresh_user_001"

    _chat_for_user(user_id, "delivery_preference_refresh_seed_1", "我喜欢顺丰配送，以后发货优先顺丰")
    second = _chat_for_user(user_id, "delivery_preference_refresh_seed_2", "以后给我发货优先京东物流")

    assert second["user_memories"] == ["用户偏好：以后给我发货优先京东物流"]


def test_chat_calls_injected_llm_and_returns_trace(monkeypatch):
    import api.main as main_module

    fake_llm = FakeChatLLM("LLM回答：会优先参考你的京东物流偏好，请提供订单号确认实际承运商。")
    monkeypatch.setattr(main_module.settings, "llm_mode", "hybrid")
    monkeypatch.setattr(main_module, "customer_service_llm", fake_llm)

    response = TestClient(app).post(
        "/chat",
        json={
            "user_id": "llm_user_001",
            "session_id": "llm_session_001",
            "message": "给我送货用什么快递？",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].startswith("LLM回答")
    assert payload["llm_trace"]["used_llm"] is True
    assert payload["llm_trace"]["mode"] == "hybrid"
    assert payload["llm_trace"]["model_used"] == "primary"
    assert payload["llm_trace"]["tool_name"] == "delivery_preference"
    assert fake_llm.calls == 1


def test_hybrid_mode_with_llm_startup_error_keeps_rule_answer(monkeypatch):
    import api.main as main_module

    fake_llm = FakeChatLLM("offline_stub")
    monkeypatch.setattr(main_module.settings, "llm_mode", "hybrid")
    monkeypatch.setattr(main_module, "customer_service_llm", fake_llm)
    monkeypatch.setattr(
        main_module,
        "customer_service_llm_setup",
        SimpleNamespace(startup_error="OPENAI_API_KEY is not configured."),
    )

    response = TestClient(app).post(
        "/chat",
        json={
            "user_id": "llm_startup_error_user_001",
            "session_id": "llm_startup_error_session_001",
            "message": "给我送货用什么快递？",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] != "offline_stub"
    assert "订单号" in payload["answer"]
    assert payload["llm_trace"]["used_llm"] is False
    assert "OPENAI_API_KEY" in payload["llm_trace"]["reasoning_summary"]
    assert fake_llm.calls == 0
