from __future__ import annotations

import re

ORDER_ID_RE = re.compile(r"\bORD\d{6,}\b", re.IGNORECASE)


def extract_order_id(text: str) -> str:
    match = ORDER_ID_RE.search(text)
    return match.group(0).upper() if match else ""


def contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    normalized = text.casefold()
    return any(keyword.casefold() in normalized for keyword in keywords)


def is_human_transfer_request(text: str) -> bool:
    return contains_any(
        text,
        (
            "人工",
            "转人工",
            "投诉",
            "法务",
            "法律",
            "起诉",
            "纠纷",
            "human",
            "complaint",
            "legal",
        ),
    )


def is_refund_request(text: str) -> bool:
    return contains_any(text, ("退款", "退货", "refund", "return"))


def is_refund_confirmed(text: str) -> bool:
    return contains_any(text, ("确认", "同意", "继续", "confirm", "yes", "submit"))


def is_logistics_query(text: str) -> bool:
    return contains_any(text, ("物流", "快递", "配送", "tracking", "where is", "到哪"))


def is_order_query(text: str) -> bool:
    return contains_any(text, ("订单", "order"))


def is_product_query(text: str) -> bool:
    return contains_any(text, ("airbuds", "homehub", "商品", "库存", "有货", "product", "stock"))


def is_faq_query(text: str) -> bool:
    return contains_any(
        text,
        (
            "政策",
            "规则",
            "faq",
            "常见问题",
            "售后",
            "退换货",
            "退货政策",
            "保修",
            "会员",
            "发票",
        ),
    )


def is_memory_recall_query(text: str) -> bool:
    return contains_any(text, ("记得", "偏好", "之前", "历史", "remember", "preference"))


def is_preference_statement(text: str) -> bool:
    return contains_any(text, ("喜欢", "偏好", "优先", "prefer", "preference"))
