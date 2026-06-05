from __future__ import annotations

import re
import uuid
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any


DEFAULT_CHOICE_TTL_SECONDS = 15 * 60
CHINESE_INDEXES = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
INDEX_RE = re.compile(r"^\s*(?:选|选择)?\s*第?\s*([1-9]\d*|[一二两三四五六七八九十])\s*(?:个|项|条)?\s*$")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def create_choice_set(
    *,
    scenario: str,
    title: str,
    prompt: str,
    options: list[dict[str, Any]],
    source_tool: str,
    payload: dict[str, Any] | None = None,
    ttl_seconds: int = DEFAULT_CHOICE_TTL_SECONDS,
) -> dict[str, Any]:
    now = utc_now()
    normalized_options = []
    for index, option in enumerate(options, start=1):
        normalized_options.append(
            {
                "id": str(option["id"]),
                "label": str(option["label"]),
                "value": str(option.get("value") or option["id"]),
                "description": str(option.get("description", "")),
                "requires_confirmation": bool(option.get("requires_confirmation", False)),
                "aliases": [str(alias) for alias in option.get("aliases", [])],
                "index": index,
            }
        )
    return {
        "choice_set_id": f"choice_{scenario}_{uuid.uuid4().hex[:12]}",
        "scenario": scenario,
        "title": title,
        "prompt": prompt,
        "options": normalized_options,
        "source_tool": source_tool,
        "payload": dict(payload or {}),
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=ttl_seconds)).isoformat(),
    }


def public_choice_set(choice_set: dict[str, Any] | None) -> dict[str, Any] | None:
    if not choice_set:
        return None
    public_options = []
    for option in choice_set.get("options", []):
        public_options.append(
            {
                "id": option.get("id", ""),
                "label": option.get("label", ""),
                "value": option.get("value", ""),
                "description": option.get("description", ""),
                "requires_confirmation": bool(option.get("requires_confirmation", False)),
            }
        )
    return {
        "choice_set_id": choice_set.get("choice_set_id", ""),
        "scenario": choice_set.get("scenario", ""),
        "title": choice_set.get("title", ""),
        "prompt": choice_set.get("prompt", ""),
        "options": public_options,
        "expires_at": choice_set.get("expires_at", ""),
    }


def is_choice_expired(choice_set: dict[str, Any] | None, *, now: datetime | None = None) -> bool:
    if not choice_set:
        return False
    expires_at = str(choice_set.get("expires_at") or "")
    if not expires_at:
        return False
    try:
        parsed = datetime.fromisoformat(expires_at)
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (now or utc_now()) >= parsed


def resolve_choice(choice_set: dict[str, Any] | None, user_input: str) -> dict[str, Any]:
    if not choice_set:
        return {"matched": False, "expired": False, "option": None}
    if is_choice_expired(choice_set):
        return {"matched": False, "expired": True, "option": None}

    text = user_input.strip()
    normalized = text.casefold()
    index = _parse_choice_index(text)
    if index is not None:
        for option in choice_set.get("options", []):
            if int(option.get("index") or 0) == index:
                return {"matched": True, "expired": False, "option": deepcopy(option)}

    for option in choice_set.get("options", []):
        candidates = [
            str(option.get("id", "")),
            str(option.get("value", "")),
            str(option.get("label", "")),
            *[str(alias) for alias in option.get("aliases", [])],
        ]
        for candidate in candidates:
            candidate = candidate.strip()
            if candidate and candidate.casefold() in normalized:
                return {"matched": True, "expired": False, "option": deepcopy(option)}
    return {"matched": False, "expired": False, "option": None}


def create_refund_choice(order_id: str, refund_payload: dict[str, Any]) -> dict[str, Any]:
    return create_choice_set(
        scenario="refund",
        title="请选择退款处理方式",
        prompt=f"订单 {order_id} 当前可发起退款，请选择下一步。",
        source_tool="apply_refund",
        payload={"order_id": order_id, "refund": refund_payload},
        options=[
            {
                "id": "confirm_refund",
                "label": "确认退款",
                "value": "confirm_refund",
                "description": "提交退款申请并进入人工复核。",
                "requires_confirmation": True,
                "aliases": ["确认", "确认退款", "继续", "同意", "submit", "confirm", "yes"],
            },
            {
                "id": "cancel_refund",
                "label": "暂不退款",
                "value": "cancel_refund",
                "description": "取消本次退款申请。",
                "aliases": ["暂不", "取消", "不用", "先不", "cancel", "no"],
            },
            {
                "id": "human_transfer",
                "label": "转人工处理",
                "value": "human_transfer",
                "description": "让人工客服继续跟进。",
                "aliases": ["人工", "转人工", "客服", "human"],
            },
        ],
    )


def create_logistics_choice(user_memories: list[str]) -> dict[str, Any]:
    memory_text = "；".join(user_memories[:2])
    prompt = "请提供订单号后我可以查询实际承运商。"
    if memory_text:
        prompt = f"我会优先参考你的配送偏好：{memory_text}。请提供订单号确认实际承运商。"
    return create_choice_set(
        scenario="logistics",
        title="请选择物流查询方式",
        prompt=prompt,
        source_tool="delivery_preference",
        payload={"user_memories": list(user_memories[:3])},
        options=[
            {
                "id": "provide_order_id",
                "label": "提供订单号",
                "value": "provide_order_id",
                "description": "发送 ORD 开头的订单号查询物流。",
                "aliases": ["订单号", "提供订单", "查订单", "提供"],
            },
            {
                "id": "show_delivery_preference",
                "label": "查看配送偏好",
                "value": "show_delivery_preference",
                "description": "查看当前可用的配送偏好记忆。",
                "aliases": ["偏好", "配送偏好", "查看偏好", "preference"],
            },
            {
                "id": "human_transfer",
                "label": "转人工处理",
                "value": "human_transfer",
                "description": "让人工客服继续跟进。",
                "aliases": ["人工", "转人工", "客服", "human"],
            },
        ],
    )


def create_product_choice(product: dict[str, Any]) -> dict[str, Any]:
    product_name = str(product.get("name") or "该商品")
    return create_choice_set(
        scenario="product",
        title="请选择商品后续操作",
        prompt=f"{product_name} 的商品信息已查询到，请选择下一步。",
        source_tool="get_product",
        payload={"product": deepcopy(product)},
        options=[
            {
                "id": "check_stock",
                "label": "查看库存",
                "value": "check_stock",
                "description": "查看当前库存状态。",
                "aliases": ["库存", "有货", "stock"],
            },
            {
                "id": "check_shipping",
                "label": "查看发货时效",
                "value": "check_shipping",
                "description": "查看预计发货时间。",
                "aliases": ["发货", "时效", "配送", "shipping"],
            },
            {
                "id": "human_transfer",
                "label": "转人工处理",
                "value": "human_transfer",
                "description": "让人工客服继续跟进。",
                "aliases": ["人工", "转人工", "客服", "human"],
            },
        ],
    )


def _parse_choice_index(text: str) -> int | None:
    match = INDEX_RE.match(text)
    if not match:
        return None
    raw = match.group(1)
    if raw.isdigit():
        return int(raw)
    return CHINESE_INDEXES.get(raw)
