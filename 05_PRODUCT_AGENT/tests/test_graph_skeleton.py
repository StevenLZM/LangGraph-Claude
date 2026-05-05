from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agent.graph import build_customer_service_graph
from agent.state import CustomerServiceState


def test_graph_compiles_with_m0_nodes():
    graph = build_customer_service_graph(checkpointer=MemorySaver())

    nodes = set(graph.get_graph().nodes.keys())

    assert {"context_loader", "agent", "finalizer"}.issubset(nodes)


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
