from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agent.graph import build_customer_service_graph
from agent.state import CustomerServiceState


def test_graph_compiles_with_m0_nodes():
    graph = build_customer_service_graph(checkpointer=MemorySaver())

    nodes = set(graph.get_graph().nodes.keys())

    assert {
        "context_loader",
        "turn_router",
        "pending_task_resolver",
        "retrieval_decision",
        "slot_filling",
        "confirmation_guard",
        "tool_planner_or_react",
        "tool_executor",
        "response_builder",
        "finalizer",
    }.issubset(nodes)


def test_graph_returns_customer_service_reply():
    graph = build_customer_service_graph(checkpointer=MemorySaver())
    initial_state: CustomerServiceState = {
        "session_id": "session_001",
        "user_id": "user_001",
        "messages": [HumanMessage(content="你好，我想查询订单")],
    }

    result = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": "session_001"}},
    )

    assert result["session_id"] == "session_001"
    assert result["user_id"] == "user_001"
    assert result["needs_human_transfer"] is False
    assert result["transfer_reason"] == ""
    assert result["window_size"] == 2
    assert result["total_turns"] == 1
    assert result["token_used"] >= 1
    assert result["response_time_ms"] >= 0
    assert isinstance(result["messages"][-1], AIMessage)
    assert "客服" in result["messages"][-1].content


def test_graph_missing_refund_slots_stops_before_tool_execution():
    graph = build_customer_service_graph(checkpointer=MemorySaver())
    initial_state: CustomerServiceState = {
        "session_id": "return_slots_session",
        "user_id": "user_001",
        "messages": [HumanMessage(content="我要退货")],
    }

    result = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": "return_slots_session"}},
    )

    assert result["route"] == "new_task"
    assert result["task_status"]["active_task"] == "refund"
    assert result["task_status"]["phase"] == "collecting_slots"
    assert "order_id" in result["task_status"]["missing_slots"]
    assert result["tool_trace"] == []
    assert result["order_context"] is None
    assert "订单号" in result["messages"][-1].content


def test_graph_confirmation_required_stops_before_refund_execution():
    graph = build_customer_service_graph(checkpointer=MemorySaver())
    initial_state: CustomerServiceState = {
        "session_id": "return_confirm_session",
        "user_id": "user_001",
        "messages": [HumanMessage(content="我要退货 ORD123456，不想要了，商品完好")],
    }

    result = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": "return_confirm_session"}},
    )

    assert result["task_status"]["active_task"] == "refund"
    assert result["task_status"]["phase"] == "awaiting_confirmation"
    assert result["order_context"]["refund_status"] == "confirmation_required"
    assert result["choices"]["scenario"] == "refund"
    assert result["tool_trace"] == []
    assert "已提交" not in result["messages"][-1].content


def test_read_only_refund_eligibility_react_runs_policy_and_logistics_tools():
    graph = build_customer_service_graph(checkpointer=MemorySaver())
    initial_state: CustomerServiceState = {
        "session_id": "readonly_refund_eligibility_session",
        "user_id": "user_001",
        "messages": [HumanMessage(content="我这个订单 ORD123456 还能退吗，物流到哪了？")],
        "max_tool_steps": 3,
    }

    result = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": "readonly_refund_eligibility_session"}},
    )

    assert [step["tool_name"] for step in result["tool_trace"]] == [
        "get_order",
        "get_logistics",
        "faq_rag",
    ]
    assert all(step["read_only"] is True for step in result["tool_trace"])
    assert result["task_status"] is None
    assert "rag_sources" in result["order_context"]
    assert "rag_matched" in result["order_context"]
    assert result["order_context"]["rag_backend"]


def test_refund_write_tool_receives_idempotency_key(monkeypatch):
    captured: list[dict[str, str | bool]] = []

    def fake_apply_refund(order_id: str, *, confirmed: bool, idempotency_key: str = "") -> dict:
        captured.append(
            {
                "order_id": order_id,
                "confirmed": confirmed,
                "idempotency_key": idempotency_key,
            }
        )
        return {
            "order_id": order_id,
            "refund_status": "submitted",
            "refund_ticket_id": "RF-TEST",
        }

    monkeypatch.setattr("agent.tool_executor.apply_refund", fake_apply_refund)

    graph = build_customer_service_graph(checkpointer=MemorySaver())
    dialog_state = {
        "active_task": "refund",
        "phase": "awaiting_confirmation",
        "required_slots": ["order_id"],
        "collected_slots": {"order_id": "ORD123456"},
        "missing_slots": [],
        "confirmation": {
            "required": True,
            "confirmed": False,
            "action": "apply_refund",
            "preview": "将为订单 ORD123456 提交退款申请。",
        },
        "attempt_count": 0,
        "expires_at": "2999-01-01T00:00:00+00:00",
        "idempotency_key": "apply_refund:ORD123456",
    }
    initial_state: CustomerServiceState = {
        "session_id": "refund_idempotency_session",
        "user_id": "user_001",
        "messages": [HumanMessage(content="确认退款")],
        "dialog_state": dialog_state,
        "max_tool_steps": 3,
    }

    result = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": "refund_idempotency_session"}},
    )

    assert captured == [
        {
            "order_id": "ORD123456",
            "confirmed": True,
            "idempotency_key": "apply_refund:ORD123456",
        }
    ]
    assert result["task_status"]["phase"] == "completed"


def test_teaching_guide_does_not_describe_current_graph_as_linear():
    guide = Path("Codex_TEACHING_GUIDE.md").read_text(encoding="utf-8")

    assert "LangGraph 当前是线性图" not in guide
