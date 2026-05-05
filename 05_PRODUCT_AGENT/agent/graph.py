from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.nodes import agent_node, context_loader_node, finalizer_node
from agent.state import CustomerServiceState


def build_customer_service_graph(checkpointer: Any | None = None):
    workflow = StateGraph(CustomerServiceState)

    workflow.add_node("context_loader", context_loader_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("finalizer", finalizer_node)

    workflow.add_edge(START, "context_loader")
    workflow.add_edge("context_loader", "agent")
    workflow.add_edge("agent", "finalizer")
    workflow.add_edge("finalizer", END)

    return workflow.compile(checkpointer=checkpointer)
