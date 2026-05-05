from __future__ import annotations

import time

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent.prompts import OFFLINE_M0_REPLY
from agent.service import handle_customer_message
from agent.state import CustomerServiceState


def _count_human_turns(messages: list[BaseMessage]) -> int:
    return sum(1 for message in messages if isinstance(message, HumanMessage))


def context_loader_node(state: CustomerServiceState) -> dict:
    messages = list(state.get("messages") or [])
    return {
        "session_id": state.get("session_id", ""),
        "user_id": state.get("user_id", ""),
        "user_profile": state.get("user_profile", {}),
        "user_memories": state.get("user_memories", []),
        "order_context": state.get("order_context"),
        "needs_human_transfer": state.get("needs_human_transfer", False),
        "transfer_reason": state.get("transfer_reason", ""),
        "token_used": state.get("token_used", 0),
        "response_time_ms": state.get("response_time_ms", 0),
        "quality_score": state.get("quality_score"),
        "tool_name": state.get("tool_name", ""),
        "window_size": len(messages),
        "total_turns": state.get("total_turns", _count_human_turns(messages)),
        "_started_at": time.perf_counter(),
    }


def agent_node(state: CustomerServiceState) -> dict:
    messages = list(state.get("messages") or [])
    latest_human = next((message for message in reversed(messages) if isinstance(message, HumanMessage)), None)
    if latest_human is not None:
        decision = handle_customer_message(str(latest_human.content))
        return {
            "messages": [
                AIMessage(
                    content=decision.answer,
                    additional_kwargs={"tool_name": decision.tool_name},
                )
            ],
            "order_context": decision.order_context,
            "needs_human_transfer": decision.needs_human_transfer,
            "transfer_reason": decision.transfer_reason,
            "quality_score": decision.quality_score,
            "tool_name": decision.tool_name,
            "token_used": max(1, len(str(latest_human.content)) // 2),
        }

    return {
        "messages": [
            AIMessage(
                content=OFFLINE_M0_REPLY,
                additional_kwargs={"mode": "offline_stub"},
            )
        ],
        "needs_human_transfer": False,
        "transfer_reason": "",
        "token_used": 0,
    }


def finalizer_node(state: CustomerServiceState) -> dict:
    messages = list(state.get("messages") or [])
    started_at = state.get("_started_at")
    response_time_ms = 0
    if isinstance(started_at, float):
        response_time_ms = max(0, int((time.perf_counter() - started_at) * 1000))

    return {
        "window_size": len(messages),
        "total_turns": _count_human_turns(messages),
        "response_time_ms": response_time_ms,
    }
