from __future__ import annotations

from agent.tools import (
    apply_refund,
    get_logistics,
    get_order,
    get_product,
    list_customer_service_tools,
)


def test_order_tool_returns_mock_order():
    order = get_order("ORD123456")

    assert order["order_id"] == "ORD123456"
    assert order["status"] == "配送中"
    assert order["tracking_no"] == "SF100200300CN"


def test_logistics_tool_returns_tracking_events():
    logistics = get_logistics("ORD123456")

    assert logistics["carrier"] == "顺丰"
    assert logistics["latest_location"] == "上海浦东配送站"
    assert logistics["events"]


def test_product_tool_returns_mock_product():
    product = get_product("AirBuds Pro 2")

    assert product["name"] == "AirBuds Pro 2"
    assert product["availability"] == "有货"


def test_refund_tool_submits_only_after_confirmation():
    refund = apply_refund("ORD123456", confirmed=True)

    assert refund["order_id"] == "ORD123456"
    assert refund["refund_status"] == "submitted"


def test_tool_registry_lists_m1_tools():
    assert list_customer_service_tools() == [
        "get_order",
        "get_logistics",
        "get_product",
        "apply_refund",
    ]
