from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from agent.dialog_state import refresh_missing_slots
from agent.intent import contains_any, extract_order_id, is_refund_confirmed


RETURN_REQUIRED_SLOTS = ["order_id", "reason", "product_condition"]
REFUND_REQUIRED_SLOTS = ["order_id"]
LOGISTICS_REQUIRED_SLOTS = ["order_id"]

_ORDER_ONLY_RE = re.compile(r"^\s*ORD\d{6,}\s*$", re.IGNORECASE)


def is_return_request(text: str) -> bool:
    return contains_any(text, ("退货", "return"))


def is_cancel_request(text: str) -> bool:
    return contains_any(text, ("取消", "暂不", "不用", "先不", "cancel", "no"))


def is_confirmation_request(text: str) -> bool:
    return is_refund_confirmed(text) or contains_any(text, ("确认退货", "确认退款"))


def is_order_id_only(text: str) -> bool:
    return bool(_ORDER_ONLY_RE.match(text.strip()))


def refund_required_slots(text: str) -> list[str]:
    return list(RETURN_REQUIRED_SLOTS if is_return_request(text) else REFUND_REQUIRED_SLOTS)


def collect_slots(dialog_state: dict[str, Any], message: str) -> dict[str, Any]:
    updated = deepcopy(dialog_state)
    slots = dict(updated.get("collected_slots") or {})
    order_id = extract_order_id(message)
    if order_id:
        slots["order_id"] = order_id

    active_task = str(updated.get("active_task") or "")
    if active_task == "refund":
        reason = _extract_refund_reason(message)
        if reason:
            slots["reason"] = reason
        product_condition = _extract_product_condition(message)
        if product_condition:
            slots["product_condition"] = product_condition
    if active_task == "product":
        selected = _extract_product_selection(message)
        if selected:
            slots["selected_option"] = selected

    updated["collected_slots"] = slots
    return refresh_missing_slots(updated)


def has_required_slots(dialog_state: dict[str, Any]) -> bool:
    return not dialog_state.get("missing_slots")


def _extract_refund_reason(message: str) -> str:
    text = message.strip()
    if not text or is_order_id_only(text) or is_confirmation_request(text) or is_cancel_request(text):
        return ""
    if contains_any(text, ("不想要", "质量", "不好用", "坏", "破损", "买错", "退货原因", "原因", "不合适")):
        return text
    if len(text) >= 8 and not extract_order_id(text):
        return text
    return ""


def _extract_product_condition(message: str) -> str:
    text = message.strip()
    if contains_any(text, ("商品完好", "完好", "未拆封", "未使用", "全新", "包装完整")):
        return "商品完好"
    if contains_any(text, ("破损", "损坏", "坏了", "质量问题")):
        return "商品异常"
    return ""


def _extract_product_selection(message: str) -> str:
    if contains_any(message, ("库存", "有货", "stock")):
        return "check_stock"
    if contains_any(message, ("发货", "时效", "shipping")):
        return "check_shipping"
    return ""
