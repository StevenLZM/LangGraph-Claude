from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent.intent import (
    extract_order_id,
    is_faq_query,
    is_logistics_query,
    is_memory_recall_query,
    is_order_query,
    is_product_query,
    is_refund_request,
)
from agent.slot_filling import is_return_request


def build_tool_plan(state: dict[str, Any]) -> dict[str, Any]:
    dialog_state = state.get("dialog_state") or {}
    if dialog_state.get("active_task") == "refund" and dialog_state.get("phase") == "executing":
        slots = dict(dialog_state.get("collected_slots") or {})
        return _plan(
            [
                {
                    "tool_name": "apply_refund",
                    "args": {
                        "order_id": slots.get("order_id", ""),
                        "confirmed": True,
                        "idempotency_key": dialog_state.get("idempotency_key", ""),
                    },
                    "read_only": False,
                }
            ]
        )

    if dialog_state.get("active_task") == "logistics" and dialog_state.get("phase") == "executing":
        order_id = str((dialog_state.get("collected_slots") or {}).get("order_id") or "")
        return _plan(_logistics_steps(order_id))

    if dialog_state.get("active_task") == "product" and dialog_state.get("phase") == "executing":
        return _plan([])

    latest_message = str(state.get("latest_user_message") or "")
    order_id = extract_order_id(latest_message)
    retrieval_decision = dict(state.get("retrieval_decision") or {})
    intent = str(retrieval_decision.get("intent") or "")

    if _is_read_only_refund_question(latest_message) and order_id:
        return _plan(
            [
                {"tool_name": "get_order", "args": {"order_id": order_id}, "read_only": True},
                {"tool_name": "get_logistics", "args": {"order_id": order_id}, "read_only": True},
                {"tool_name": "faq_rag", "args": {"query": latest_message}, "read_only": True},
            ],
            react_mode=True,
        )
    if intent == "faq" or is_faq_query(latest_message):
        return _plan([{"tool_name": "faq_rag", "args": {"query": latest_message}, "read_only": True}])
    if intent == "memory_recall" or is_memory_recall_query(latest_message):
        return _plan([{"tool_name": "load_user_memory", "args": {"query": latest_message}, "read_only": True}])
    if order_id and (intent == "logistics" or is_logistics_query(latest_message)):
        return _plan(_logistics_steps(order_id))
    if order_id and (intent == "order" or is_order_query(latest_message)):
        return _plan([{"tool_name": "get_order", "args": {"order_id": order_id}, "read_only": True}])
    if intent == "product" or is_product_query(latest_message):
        return _plan([{"tool_name": "get_product", "args": {"query": latest_message}, "read_only": True}])
    return _plan([])


def current_step(tool_plan: dict[str, Any] | None) -> dict[str, Any] | None:
    if not tool_plan:
        return None
    steps = list(tool_plan.get("steps") or [])
    index = int(tool_plan.get("current_step") or 0)
    if index < 0 or index >= len(steps):
        return None
    return deepcopy(steps[index])


def advance_plan(tool_plan: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(tool_plan)
    updated["current_step"] = int(updated.get("current_step") or 0) + 1
    if bool(updated.get("react_mode")):
        updated["react_step_count"] = int(updated.get("react_step_count") or 0) + 1
    return updated


def should_continue_react(tool_plan: dict[str, Any] | None, *, max_tool_steps: int) -> bool:
    if not tool_plan or not bool(tool_plan.get("react_mode")):
        return False
    if int(tool_plan.get("react_step_count") or 0) >= max_tool_steps:
        return False
    return current_step(tool_plan) is not None


def final_tool_name(tool_plan: dict[str, Any] | None, tool_results: list[dict[str, Any]]) -> str:
    if tool_results:
        return str(tool_results[-1].get("tool_name") or "")
    if tool_plan:
        steps = list(tool_plan.get("steps") or [])
        if steps:
            return str(steps[-1].get("tool_name") or "")
    return ""


def _plan(steps: list[dict[str, Any]], *, react_mode: bool = False) -> dict[str, Any]:
    return {
        "steps": steps,
        "current_step": 0,
        "react_mode": react_mode,
        "react_step_count": 0,
    }


def _logistics_steps(order_id: str) -> list[dict[str, Any]]:
    return [
        {"tool_name": "get_order", "args": {"order_id": order_id}, "read_only": True},
        {"tool_name": "get_logistics", "args": {"order_id": order_id}, "read_only": True},
    ]


def _is_read_only_refund_question(message: str) -> bool:
    if not is_refund_request(message):
        return False
    if is_return_request(message):
        return False
    return any(keyword in message for keyword in ("能退", "还能退", "可以退", "能不能退", "政策"))
