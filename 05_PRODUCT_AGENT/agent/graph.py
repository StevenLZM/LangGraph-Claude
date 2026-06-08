from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    confirmation_guard_node,
    context_loader_node,
    finalizer_node,
    pending_task_resolver_node,
    response_builder_node,
    retrieval_decision_node,
    route_after_confirmation_guard,
    route_after_slot_filling,
    route_after_tool_executor,
    route_after_turn_router,
    slot_filling_node,
    tool_executor_node,
    tool_planner_or_react_node,
    turn_router_node,
)
from agent.state import CustomerServiceState


def build_customer_service_graph(checkpointer: Any | None = None):
    workflow = StateGraph(CustomerServiceState)

    workflow.add_node("context_loader", context_loader_node)
    workflow.add_node("turn_router", turn_router_node)
    workflow.add_node("pending_task_resolver", pending_task_resolver_node)
    workflow.add_node("retrieval_decision", retrieval_decision_node)
    workflow.add_node("slot_filling", slot_filling_node)
    workflow.add_node("confirmation_guard", confirmation_guard_node)
    workflow.add_node("tool_planner_or_react", tool_planner_or_react_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("response_builder", response_builder_node)
    workflow.add_node("finalizer", finalizer_node)

    workflow.add_edge(START, "context_loader")
    workflow.add_edge("context_loader", "turn_router")
    workflow.add_conditional_edges(
        "turn_router",
        route_after_turn_router,
        {
            "pending_task_resolver": "pending_task_resolver",
            "retrieval_decision": "retrieval_decision",
            "response_builder": "response_builder",
        },
    )
    workflow.add_edge("pending_task_resolver", "slot_filling")
    workflow.add_edge("retrieval_decision", "slot_filling")
    workflow.add_conditional_edges(
        "slot_filling",
        route_after_slot_filling,
        {
            "response_builder": "response_builder",
            "confirmation_guard": "confirmation_guard",
        },
    )
    workflow.add_conditional_edges(
        "confirmation_guard",
        route_after_confirmation_guard,
        {
            "response_builder": "response_builder",
            "tool_planner_or_react": "tool_planner_or_react",
        },
    )
    workflow.add_edge("tool_planner_or_react", "tool_executor")
    workflow.add_conditional_edges(
        "tool_executor",
        route_after_tool_executor,
        {
            "tool_planner_or_react": "tool_planner_or_react",
            "response_builder": "response_builder",
        },
    )
    workflow.add_edge("response_builder", "finalizer")
    workflow.add_edge("finalizer", END)

    return workflow.compile(checkpointer=checkpointer)
