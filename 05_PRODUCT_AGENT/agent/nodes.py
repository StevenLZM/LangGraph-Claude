from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from agent.dialog_state import (
    MAX_SLOT_ATTEMPTS,
    increment_attempt,
    is_expired,
    new_dialog_state,
    normalize_dialog_state,
    refresh_missing_slots,
    with_phase,
)
from agent.intent import (
    extract_order_id,
    is_human_transfer_request,
    is_logistics_query,
    is_memory_recall_query,
    is_preference_statement,
    is_refund_request,
)
from agent.response_builder import build_response_update
from agent.retrieval import rule_based_retrieval_decision
from agent.slot_filling import (
    LOGISTICS_REQUIRED_SLOTS,
    collect_slots,
    has_required_slots,
    is_cancel_request,
    is_confirmation_request,
    is_order_id_only,
    refund_required_slots,
)
from agent.tool_executor import execute_next_tool
from agent.tool_planner import build_tool_plan, current_step, should_continue_react


def _count_human_turns(messages: list[BaseMessage]) -> int:
    return sum(1 for message in messages if isinstance(message, HumanMessage))


def _latest_human_message(messages: list[BaseMessage]) -> str:
    latest = next((message for message in reversed(messages) if isinstance(message, HumanMessage)), None)
    return str(latest.content) if latest is not None else ""


def context_loader_node(state: dict[str, Any]) -> dict[str, Any]:
    messages = list(state.get("messages") or [])
    dialog_state = normalize_dialog_state(state.get("dialog_state"))
    return {
        "session_id": state.get("session_id", ""),
        "user_id": state.get("user_id", ""),
        "user_profile": state.get("user_profile", {}),
        "user_memories": state.get("user_memories", []),
        "memory_summary": state.get("memory_summary", ""),
        "retrieval_decision": state.get("retrieval_decision", {}),
        "dialog_state": dialog_state,
        "task_status": state.get("task_status"),
        "order_context": None,
        "choices": None,
        "pending_choice": None,
        "route": "",
        "latest_user_message": _latest_human_message(messages),
        "tool_plan": {},
        "tool_results": [],
        "tool_trace": [],
        "max_tool_steps": int(state.get("max_tool_steps") or 3),
        "needs_human_transfer": False,
        "transfer_reason": "",
        "token_used": state.get("token_used", 0),
        "response_time_ms": 0,
        "quality_score": None,
        "tool_name": "",
        "window_size": len(messages),
        "total_turns": state.get("total_turns", _count_human_turns(messages)),
        "_started_at": time.perf_counter(),
    }


def turn_router_node(state: dict[str, Any]) -> dict[str, Any]:
    message = str(state.get("latest_user_message") or "")
    dialog_state = normalize_dialog_state(state.get("dialog_state"))
    if dialog_state and is_expired(dialog_state):
        expired = with_phase(dialog_state, "cancelled")
        return {"route": "expired_task", "dialog_state": expired}
    if is_human_transfer_request(message):
        return {"route": "human_transfer"}
    if dialog_state:
        if is_cancel_request(message):
            return {"route": "cancel_pending"}
        if is_confirmation_request(message):
            return {"route": "confirm_pending"}
        if _is_high_priority_new_task(message, dialog_state):
            return {"route": "new_task", "dialog_state": None}
        return {"route": "continue_pending"}
    return {"route": "new_task"}


def pending_task_resolver_node(state: dict[str, Any]) -> dict[str, Any]:
    dialog_state = normalize_dialog_state(state.get("dialog_state"))
    route = str(state.get("route") or "")
    if not dialog_state:
        return {}
    if route in {"cancel_pending", "expired_task"}:
        return {"dialog_state": with_phase(dialog_state, "cancelled")}
    if route == "confirm_pending":
        confirmation = dict(dialog_state.get("confirmation") or {})
        confirmation["confirmed"] = True
        resolved = deepcopy(dialog_state)
        resolved["confirmation"] = confirmation
        return {"dialog_state": resolved}
    return {"dialog_state": dialog_state}


def retrieval_decision_node(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("retrieval_decision"):
        return {"retrieval_decision": state.get("retrieval_decision", {})}
    return {
        "retrieval_decision": rule_based_retrieval_decision(
            str(state.get("latest_user_message") or "")
        ).to_dict()
    }


def slot_filling_node(state: dict[str, Any]) -> dict[str, Any]:
    message = str(state.get("latest_user_message") or "")
    route = str(state.get("route") or "")
    dialog_state = normalize_dialog_state(state.get("dialog_state"))

    if dialog_state and dialog_state.get("phase") == "cancelled":
        return {"dialog_state": dialog_state}

    if dialog_state:
        return {"dialog_state": _advance_existing_task(dialog_state, message, route)}

    if _needs_direct_response(message):
        return {"route": "direct_response"}

    retrieval_intent = str((state.get("retrieval_decision") or {}).get("intent") or "")
    if retrieval_intent == "faq" or _is_read_only_refund_question(message):
        return {}

    new_task = _new_task_from_message(message)
    if new_task:
        return {"dialog_state": _advance_existing_task(new_task, message, route)}

    return {}


def confirmation_guard_node(state: dict[str, Any]) -> dict[str, Any]:
    dialog_state = normalize_dialog_state(state.get("dialog_state"))
    if not dialog_state:
        return {}
    active_task = str(dialog_state.get("active_task") or "")
    phase = str(dialog_state.get("phase") or "")
    if active_task != "refund":
        return {"dialog_state": dialog_state}
    if phase == "executing":
        return {"dialog_state": dialog_state}
    if phase == "awaiting_confirmation":
        confirmation = dict(dialog_state.get("confirmation") or {})
        if confirmation.get("confirmed"):
            return {"dialog_state": with_phase(dialog_state, "executing")}
        return {"dialog_state": dialog_state}
    return {"dialog_state": dialog_state}


def tool_planner_or_react_node(state: dict[str, Any]) -> dict[str, Any]:
    if current_step(state.get("tool_plan")) is not None:
        return {}
    return {"tool_plan": build_tool_plan(state)}


def tool_executor_node(state: dict[str, Any]) -> dict[str, Any]:
    return execute_next_tool(state)


def response_builder_node(state: dict[str, Any]) -> dict[str, Any]:
    return build_response_update(state)


def finalizer_node(state: dict[str, Any]) -> dict[str, Any]:
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


def route_after_turn_router(state: dict[str, Any]) -> str:
    route = str(state.get("route") or "")
    if route in {"continue_pending", "confirm_pending", "cancel_pending", "expired_task"}:
        return "pending_task_resolver"
    if route == "human_transfer":
        return "response_builder"
    return "retrieval_decision"


def route_after_slot_filling(state: dict[str, Any]) -> str:
    if str(state.get("route") or "") == "direct_response":
        return "response_builder"
    dialog_state = normalize_dialog_state(state.get("dialog_state"))
    if not dialog_state:
        return "confirmation_guard"
    phase = str(dialog_state.get("phase") or "")
    if phase in {"collecting_slots", "cancelled", "escalated", "awaiting_confirmation"}:
        return "response_builder" if phase != "awaiting_confirmation" else "confirmation_guard"
    return "confirmation_guard"


def route_after_confirmation_guard(state: dict[str, Any]) -> str:
    dialog_state = normalize_dialog_state(state.get("dialog_state"))
    if not dialog_state:
        return "tool_planner_or_react"
    phase = str(dialog_state.get("phase") or "")
    if phase in {"collecting_slots", "awaiting_confirmation", "cancelled", "escalated"}:
        return "response_builder"
    return "tool_planner_or_react"


def route_after_tool_executor(state: dict[str, Any]) -> str:
    tool_plan = state.get("tool_plan")
    if current_step(tool_plan) is None:
        return "response_builder"
    if not bool((tool_plan or {}).get("react_mode")):
        return "tool_planner_or_react"
    if should_continue_react(tool_plan, max_tool_steps=int(state.get("max_tool_steps") or 3)):
        return "tool_planner_or_react"
    return "response_builder"


def _advance_existing_task(dialog_state: dict[str, Any], message: str, route: str) -> dict[str, Any]:
    active_task = str(dialog_state.get("active_task") or "")
    if route == "cancel_pending":
        return with_phase(dialog_state, "cancelled")
    if active_task == "product":
        updated = collect_slots(dialog_state, message)
        if (updated.get("collected_slots") or {}).get("selected_option"):
            return with_phase(updated, "executing")
        return updated

    updated = collect_slots(dialog_state, message)
    if not has_required_slots(updated):
        attempted = increment_attempt(updated) if route == "continue_pending" else updated
        if int(attempted.get("attempt_count") or 0) > MAX_SLOT_ATTEMPTS:
            return with_phase(attempted, "escalated")
        return with_phase(attempted, "collecting_slots")

    if active_task == "refund":
        confirmation = dict(updated.get("confirmation") or {})
        if confirmation.get("confirmed") or route == "confirm_pending":
            confirmation["confirmed"] = True
            updated["confirmation"] = confirmation
            return with_phase(updated, "executing")
        return with_phase(updated, "awaiting_confirmation")
    if active_task == "logistics":
        return with_phase(updated, "executing")
    return refresh_missing_slots(updated)


def _new_task_from_message(message: str) -> dict[str, Any] | None:
    order_id = extract_order_id(message)
    if is_refund_request(message):
        return new_dialog_state(
            active_task="refund",
            required_slots=refund_required_slots(message),
            collected_slots={"order_id": order_id} if order_id else {},
            confirmation={
                "required": True,
                "confirmed": False,
                "action": "apply_refund",
                "preview": f"将为订单 {order_id} 提交退款申请。" if order_id else "将提交退款申请。",
            },
        )
    if is_logistics_query(message) and not order_id:
        return new_dialog_state(
            active_task="logistics",
            required_slots=LOGISTICS_REQUIRED_SLOTS,
            collected_slots={},
            confirmation={"required": False, "confirmed": False, "action": "get_logistics", "preview": ""},
        )
    return None


def _is_read_only_refund_question(message: str) -> bool:
    if not is_refund_request(message):
        return False
    return any(keyword in message for keyword in ("能退", "还能退", "可以退", "能不能退", "政策"))


def _needs_direct_response(message: str) -> bool:
    return is_memory_recall_query(message) or is_preference_statement(message)


def _is_high_priority_new_task(message: str, dialog_state: dict[str, Any]) -> bool:
    active_task = str(dialog_state.get("active_task") or "")
    if active_task == "logistics" and is_refund_request(message):
        return True
    if active_task != "refund" and is_refund_request(message):
        return True
    if active_task == "product" and not is_order_id_only(message) and (
        is_refund_request(message) or is_logistics_query(message)
    ):
        return True
    return False
