from __future__ import annotations

import uuid
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from api.main import app
from agent.choices import create_refund_choice


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
            "request_id": uuid.uuid4().hex,
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
            "request_id": uuid.uuid4().hex,
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
    assert payload["choices"]["scenario"] == "product"


def test_refund_requires_explicit_confirmation_before_submit():
    first = _chat("我要给订单 ORD123456 退款", session_id="refund_session")
    assert "确认" in first["answer"]
    assert "已提交" not in first["answer"]
    assert first["order_context"]["refund_status"] == "confirmation_required"
    assert first["choices"]["scenario"] == "refund"
    assert [option["id"] for option in first["choices"]["options"]] == [
        "confirm_refund",
        "cancel_refund",
        "human_transfer",
    ]

    confirmed = _chat("确认退款", session_id="refund_session")
    assert "退款申请已提交" in confirmed["answer"]
    assert confirmed["order_context"]["refund_status"] == "submitted"
    assert confirmed["choices"] is None
    assert confirmed["task_status"]["phase"] == "completed"


def test_return_flow_collects_slots_before_confirmation_and_submit():
    session_id = "return_multiturn_session"

    first = _chat("我要退货", session_id=session_id)
    assert first["task_status"]["active_task"] == "refund"
    assert first["task_status"]["phase"] == "collecting_slots"
    assert "order_id" in first["task_status"]["missing_slots"]
    assert "已提交" not in first["answer"]

    second = _chat("ORD123456", session_id=session_id)
    assert second["task_status"]["phase"] == "collecting_slots"
    assert "order_id" not in second["task_status"]["missing_slots"]
    assert "reason" in second["task_status"]["missing_slots"]
    assert "已提交" not in second["answer"]

    third = _chat("不想要了，商品完好", session_id=session_id)
    assert third["task_status"]["phase"] == "awaiting_confirmation"
    assert third["choices"]["scenario"] == "refund"
    assert third["order_context"]["refund_status"] == "confirmation_required"
    assert "已提交" not in third["answer"]

    confirmed = _chat("确认退货", session_id=session_id)
    assert "退款申请已提交" in confirmed["answer"]
    assert confirmed["order_context"]["refund_status"] == "submitted"
    assert confirmed["task_status"]["phase"] == "completed"
    assert confirmed["choices"] is None


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
        json={
            "user_id": "user_001",
            "session_id": "session_001",
            "request_id": uuid.uuid4().hex,
            "message": "   ",
        },
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
    assert payload["llm_trace"]["used_llm"] is True
    assert payload["llm_trace"]["model_used"] == "test-real-llm"
    assert payload["llm_trace"]["tool_name"] == "delivery_preference"
    assert "已读取配送偏好" in payload["llm_trace"]["reasoning_summary"]
    assert payload["choices"]["scenario"] == "logistics"
    assert payload["task_status"]["active_task"] == "logistics"
    assert payload["task_status"]["phase"] == "collecting_slots"
    assert payload["task_status"]["missing_slots"] == ["order_id"]

    follow_up = _chat_for_user(user_id, "delivery_preference_query", "ORD123456")
    assert follow_up["order_context"]["tracking_no"] == "SF100200300CN"
    assert follow_up["choices"] is None
    assert follow_up["task_status"]["phase"] == "completed"


def test_legacy_pending_choice_metadata_converts_to_dialog_state():
    import api.main as main_module

    session_id = "legacy_pending_choice_session"
    choice_set = create_refund_choice(
        "ORD123456",
        {"order_id": "ORD123456", "refund_status": "confirmation_required"},
    )
    main_module.session_store.save_session(
        session_id=session_id,
        user_id="user_001",
        messages=[
            HumanMessage(content="我要给订单 ORD123456 退款"),
            AIMessage(content="请确认是否继续提交退款申请。"),
        ],
        metadata={"pending_choice": choice_set},
        expected_version=0,
    )

    confirmed = _chat("确认退款", session_id=session_id)

    assert "退款申请已提交" in confirmed["answer"]
    assert confirmed["order_context"]["refund_status"] == "submitted"
    assert confirmed["task_status"]["phase"] == "completed"


def test_ten_turn_session_keeps_task_state_stable():
    user_id = "ten_turn_user_001"
    session_id = "ten_turn_session_001"

    first = _chat_for_user(user_id, session_id, "我喜欢顺丰配送，以后发货优先顺丰")
    assert "已记住" in first["answer"]

    second = _chat_for_user(user_id, session_id, "查下物流")
    assert second["task_status"]["active_task"] == "logistics"
    assert second["task_status"]["missing_slots"] == ["order_id"]

    third = _chat_for_user(user_id, session_id, "ORD123456")
    assert third["order_context"]["tracking_no"] == "SF100200300CN"
    assert third["task_status"]["phase"] == "completed"

    fourth = _chat_for_user(user_id, session_id, "AirBuds Pro 2 还有库存吗？")
    assert fourth["choices"]["scenario"] == "product"

    fifth = _chat_for_user(user_id, session_id, "库存")
    assert "有货" in fifth["answer"]

    sixth = _chat_for_user(user_id, session_id, "我要退货")
    assert sixth["task_status"]["active_task"] == "refund"
    assert sixth["task_status"]["phase"] == "collecting_slots"

    seventh = _chat_for_user(user_id, session_id, "取消")
    assert seventh["task_status"]["phase"] == "cancelled"
    assert "取消" in seventh["answer"]

    eighth = _chat_for_user(user_id, session_id, "我要给订单 ORD123456 退款")
    assert eighth["task_status"]["phase"] == "awaiting_confirmation"
    assert eighth["order_context"]["refund_status"] == "confirmation_required"

    ninth = _chat_for_user(user_id, session_id, "确认退款")
    assert ninth["order_context"]["refund_status"] == "submitted"
    assert ninth["task_status"]["phase"] == "completed"

    tenth = _chat_for_user(user_id, session_id, "你记得我的配送偏好吗？")
    assert "顺丰" in tenth["answer"]
    assert tenth["user_memories"]
    assert tenth["task_status"] is None


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
            "request_id": uuid.uuid4().hex,
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


def test_real_llm_startup_error_returns_503_instead_of_rule_answer(monkeypatch):
    import api.main as main_module

    fake_llm = FakeChatLLM("offline_stub")
    monkeypatch.setattr(main_module.settings, "llm_mode", "deepseek")
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
            "request_id": uuid.uuid4().hex,
            "message": "给我送货用什么快递？",
        },
    )

    assert response.status_code == 503
    payload = response.json()
    assert payload["detail"]["error"] == "llm_unavailable"
    assert "OPENAI_API_KEY" in payload["detail"]["reason"]
    assert fake_llm.calls == 0
