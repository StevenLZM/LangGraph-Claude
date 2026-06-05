from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from memory.summary import SummaryMemoryStore, build_conversation_summary


def test_summary_memory_store_saves_rolling_session_summary(tmp_path):
    store = SummaryMemoryStore(str(tmp_path / "summary.db"))

    first = store.save_summary(
        session_id="session_001",
        user_id="user_001",
        summary="用户询问订单 ORD123456，客服回复正在配送。",
        covered_turns=1,
        source_event_id="evt_001",
    )
    second = store.save_summary(
        session_id="session_001",
        user_id="user_001",
        summary="用户询问订单和退款，客服已要求确认退款。",
        covered_turns=2,
        source_event_id="evt_002",
    )
    loaded = store.load_summary("session_001")

    assert first.version == 1
    assert second.version == 2
    assert loaded is not None
    assert loaded.summary == "用户询问订单和退款，客服已要求确认退款。"
    assert loaded.covered_turns == 2
    assert loaded.source_event_id == "evt_002"


def test_build_conversation_summary_extracts_recent_customer_service_context():
    summary = build_conversation_summary(
        [
            HumanMessage(content="我的订单 ORD123456 到哪了？"),
            AIMessage(content="订单 ORD123456 正在配送中。"),
            HumanMessage(content="我要退款"),
            AIMessage(content="请确认是否继续提交退款申请。"),
        ]
    )

    assert "ORD123456" in summary
    assert "退款" in summary
    assert len(summary) <= 240
