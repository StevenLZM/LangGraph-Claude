from __future__ import annotations

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
