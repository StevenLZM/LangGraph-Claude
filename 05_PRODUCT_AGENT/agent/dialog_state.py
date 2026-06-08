from __future__ import annotations

import uuid
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any


DEFAULT_TASK_TTL_SECONDS = 15 * 60
MAX_SLOT_ATTEMPTS = 3
TERMINAL_PHASES = {"completed", "cancelled", "escalated"}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def expires_at(ttl_seconds: int = DEFAULT_TASK_TTL_SECONDS) -> str:
    return (utc_now() + timedelta(seconds=ttl_seconds)).isoformat()


def is_expired(dialog_state: dict[str, Any] | None, *, now: datetime | None = None) -> bool:
    if not dialog_state:
        return False
    raw_expires_at = str(dialog_state.get("expires_at") or "")
    if not raw_expires_at:
        return False
    try:
        parsed = datetime.fromisoformat(raw_expires_at)
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (now or utc_now()) >= parsed


def new_dialog_state(
    *,
    active_task: str,
    required_slots: list[str],
    collected_slots: dict[str, Any] | None = None,
    phase: str = "collecting_slots",
    confirmation: dict[str, Any] | None = None,
    ttl_seconds: int = DEFAULT_TASK_TTL_SECONDS,
) -> dict[str, Any]:
    slots = dict(collected_slots or {})
    missing_slots = [slot for slot in required_slots if not _slot_has_value(slots.get(slot))]
    order_id = str(slots.get("order_id") or "")
    action = str((confirmation or {}).get("action") or _default_action(active_task))
    return {
        "active_task": active_task,
        "phase": phase,
        "required_slots": list(required_slots),
        "collected_slots": slots,
        "missing_slots": missing_slots,
        "confirmation": dict(confirmation or _default_confirmation(active_task, action, order_id)),
        "attempt_count": 0,
        "expires_at": expires_at(ttl_seconds),
        "idempotency_key": _idempotency_key(action, order_id),
    }


def normalize_dialog_state(dialog_state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(dialog_state, dict) or not dialog_state.get("active_task"):
        return None
    normalized = deepcopy(dialog_state)
    normalized["active_task"] = str(normalized.get("active_task") or "")
    normalized["phase"] = str(normalized.get("phase") or "collecting_slots")
    normalized["required_slots"] = [str(slot) for slot in normalized.get("required_slots") or []]
    slots = normalized.get("collected_slots") or {}
    normalized["collected_slots"] = dict(slots) if isinstance(slots, dict) else {}
    normalized["missing_slots"] = missing_slots(normalized)
    confirmation = normalized.get("confirmation") or {}
    normalized["confirmation"] = dict(confirmation) if isinstance(confirmation, dict) else {}
    normalized["attempt_count"] = int(normalized.get("attempt_count") or 0)
    normalized["expires_at"] = str(normalized.get("expires_at") or expires_at())
    action = str(normalized["confirmation"].get("action") or _default_action(normalized["active_task"]))
    order_id = str(normalized["collected_slots"].get("order_id") or "")
    normalized["idempotency_key"] = str(normalized.get("idempotency_key") or _idempotency_key(action, order_id))
    return normalized


def load_dialog_state_from_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    dialog_state = normalize_dialog_state(metadata.get("dialog_state"))
    if dialog_state:
        return dialog_state
    return dialog_state_from_pending_choice(metadata.get("pending_choice"))


def dialog_state_from_pending_choice(pending_choice: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(pending_choice, dict):
        return None
    scenario = str(pending_choice.get("scenario") or "")
    payload = dict(pending_choice.get("payload") or {})
    if scenario == "refund":
        order_id = str(payload.get("order_id") or "")
        refund_payload = dict(payload.get("refund") or {})
        if not order_id:
            order_id = str(refund_payload.get("order_id") or "")
        return new_dialog_state(
            active_task="refund",
            required_slots=["order_id"],
            collected_slots={"order_id": order_id},
            phase="awaiting_confirmation",
            confirmation={
                "required": True,
                "confirmed": False,
                "action": "apply_refund",
                "preview": f"将为订单 {order_id} 提交退款申请。",
            },
        )
    if scenario == "logistics":
        return new_dialog_state(
            active_task="logistics",
            required_slots=["order_id"],
            collected_slots={},
            phase="collecting_slots",
            confirmation={"required": False, "confirmed": False, "action": "get_logistics", "preview": ""},
        )
    if scenario == "product":
        product = dict(payload.get("product") or {})
        return new_dialog_state(
            active_task="product",
            required_slots=[],
            collected_slots={"product": product},
            phase="awaiting_selection",
            confirmation={"required": False, "confirmed": False, "action": "get_product", "preview": ""},
        )
    return None


def refresh_missing_slots(dialog_state: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(dialog_state)
    updated["missing_slots"] = missing_slots(updated)
    return updated


def missing_slots(dialog_state: dict[str, Any]) -> list[str]:
    slots = dict(dialog_state.get("collected_slots") or {})
    return [slot for slot in dialog_state.get("required_slots") or [] if not _slot_has_value(slots.get(slot))]


def increment_attempt(dialog_state: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(dialog_state)
    updated["attempt_count"] = int(updated.get("attempt_count") or 0) + 1
    return updated


def with_phase(dialog_state: dict[str, Any], phase: str) -> dict[str, Any]:
    updated = deepcopy(dialog_state)
    updated["phase"] = phase
    return updated


def public_task_status(dialog_state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not dialog_state:
        return None
    return {
        "active_task": dialog_state.get("active_task", ""),
        "phase": dialog_state.get("phase", ""),
        "missing_slots": list(dialog_state.get("missing_slots") or []),
        "expires_at": dialog_state.get("expires_at", ""),
        "attempt_count": int(dialog_state.get("attempt_count") or 0),
    }


def terminal_task_status(active_task: str, phase: str) -> dict[str, Any]:
    return {
        "active_task": active_task,
        "phase": phase,
        "missing_slots": [],
        "expires_at": "",
        "attempt_count": 0,
    }


def should_persist_dialog_state(dialog_state: dict[str, Any] | None) -> bool:
    return bool(dialog_state and dialog_state.get("phase") not in TERMINAL_PHASES)


def _slot_has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _default_action(active_task: str) -> str:
    if active_task == "refund":
        return "apply_refund"
    if active_task == "logistics":
        return "get_logistics"
    if active_task == "product":
        return "get_product"
    return active_task


def _default_confirmation(active_task: str, action: str, order_id: str) -> dict[str, Any]:
    if active_task == "refund":
        preview = f"将为订单 {order_id} 提交退款申请。" if order_id else "将提交退款申请。"
        return {"required": True, "confirmed": False, "action": action, "preview": preview}
    return {"required": False, "confirmed": False, "action": action, "preview": ""}


def _idempotency_key(action: str, order_id: str) -> str:
    subject = order_id or uuid.uuid4().hex[:12]
    return f"{action}:{subject}"
