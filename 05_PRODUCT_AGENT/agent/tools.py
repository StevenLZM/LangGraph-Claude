from __future__ import annotations

from copy import deepcopy


MOCK_ORDERS: dict[str, dict] = {
    "ORD123456": {
        "order_id": "ORD123456",
        "status": "配送中",
        "product": "AirBuds Pro 2",
        "amount": 1299,
        "currency": "CNY",
        "tracking_no": "SF100200300CN",
        "estimated_delivery": "今天 18:00 前",
    },
    "ORD654321": {
        "order_id": "ORD654321",
        "status": "已签收",
        "product": "HomeHub Mini",
        "amount": 399,
        "currency": "CNY",
        "tracking_no": "YT998877665CN",
        "estimated_delivery": "已于昨天 15:20 签收",
    },
}

MOCK_LOGISTICS: dict[str, dict] = {
    "ORD123456": {
        "order_id": "ORD123456",
        "tracking_no": "SF100200300CN",
        "carrier": "顺丰",
        "latest_location": "上海浦东配送站",
        "latest_status": "快件已到达配送站，正在安排派送",
        "events": [
            "09:10 到达上海浦东配送站",
            "06:40 离开华东分拨中心",
            "昨天 21:30 已揽收",
        ],
    },
    "ORD654321": {
        "order_id": "ORD654321",
        "tracking_no": "YT998877665CN",
        "carrier": "圆通",
        "latest_location": "上海徐汇签收点",
        "latest_status": "已签收",
        "events": [
            "昨天 15:20 用户本人签收",
            "昨天 09:35 快件正在派送",
        ],
    },
}

MOCK_PRODUCTS: dict[str, dict] = {
    "airbuds pro 2": {
        "sku": "SKU-AIRBUDS-PRO-2",
        "name": "AirBuds Pro 2",
        "availability": "有货",
        "price": 1299,
        "currency": "CNY",
        "shipping": "现货订单通常 24 小时内发出",
    },
    "homehub mini": {
        "sku": "SKU-HOMEHUB-MINI",
        "name": "HomeHub Mini",
        "availability": "少量库存",
        "price": 399,
        "currency": "CNY",
        "shipping": "预计 48 小时内发出",
    },
}


def get_order(order_id: str) -> dict:
    order = MOCK_ORDERS.get(order_id.upper())
    if order is None:
        return {"order_id": order_id.upper(), "status": "未找到"}
    return deepcopy(order)


def get_logistics(order_id: str) -> dict:
    logistics = MOCK_LOGISTICS.get(order_id.upper())
    if logistics is None:
        return {
            "order_id": order_id.upper(),
            "tracking_no": "",
            "carrier": "",
            "latest_location": "",
            "latest_status": "未找到物流信息",
            "events": [],
        }
    return deepcopy(logistics)


def get_product(query: str) -> dict:
    normalized = query.casefold()
    for key, product in MOCK_PRODUCTS.items():
        if key in normalized or product["name"].casefold() in normalized:
            return deepcopy(product)
    return {
        "sku": "",
        "name": query.strip(),
        "availability": "未找到",
        "price": 0,
        "currency": "CNY",
        "shipping": "",
    }


def apply_refund(order_id: str, *, confirmed: bool) -> dict:
    order = get_order(order_id)
    if order.get("status") == "未找到":
        return {
            "order_id": order_id.upper(),
            "refund_status": "order_not_found",
            "message": "未找到该订单，暂时无法提交退款申请。",
        }
    if not confirmed:
        return {
            **order,
            "refund_status": "confirmation_required",
            "message": "退款会进入人工复核，请确认是否继续提交退款申请。",
        }
    return {
        **order,
        "refund_status": "submitted",
        "refund_ticket_id": f"RF-{order_id.upper()}",
        "message": "退款申请已提交，预计 1-3 个工作日内完成审核。",
    }


def list_customer_service_tools() -> list[str]:
    return [
        "get_order",
        "get_logistics",
        "get_product",
        "apply_refund",
    ]
