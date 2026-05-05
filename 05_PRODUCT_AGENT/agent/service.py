from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.intent import (
    extract_order_id,
    is_human_transfer_request,
    is_logistics_query,
    is_order_query,
    is_product_query,
    is_refund_confirmed,
    is_refund_request,
)
from agent.tools import apply_refund, get_logistics, get_order, get_product


@dataclass(frozen=True)
class CustomerServiceDecision:
    answer: str
    order_context: dict[str, Any] | None
    needs_human_transfer: bool
    transfer_reason: str
    quality_score: int
    tool_name: str


def handle_customer_message(message: str) -> CustomerServiceDecision:
    order_id = extract_order_id(message)

    if is_human_transfer_request(message):
        return CustomerServiceDecision(
            answer="我已为你标记转人工处理，会把当前问题和上下文同步给人工客服继续跟进。",
            order_context=get_order(order_id) if order_id else None,
            needs_human_transfer=True,
            transfer_reason="用户要求人工或涉及投诉/法律问题",
            quality_score=82,
            tool_name="human_transfer",
        )

    if is_refund_request(message):
        if not order_id:
            return CustomerServiceDecision(
                answer="可以处理退款。请提供需要退款的订单号，例如 ORD123456。",
                order_context=None,
                needs_human_transfer=False,
                transfer_reason="",
                quality_score=76,
                tool_name="apply_refund",
            )

        refund = apply_refund(order_id, confirmed=is_refund_confirmed(message))
        if refund["refund_status"] == "submitted":
            answer = (
                f"订单 {order_id} 的退款申请已提交，工单号 {refund['refund_ticket_id']}。"
                "预计 1-3 个工作日内完成审核。"
            )
        elif refund["refund_status"] == "order_not_found":
            answer = f"没有查到订单 {order_id}，请核对订单号后再试。"
        else:
            answer = (
                f"订单 {order_id} 当前可以发起退款。退款会进入人工复核，"
                "请回复“确认退款”后我再提交申请。"
            )
        return CustomerServiceDecision(
            answer=answer,
            order_context=refund,
            needs_human_transfer=False,
            transfer_reason="",
            quality_score=86,
            tool_name="apply_refund",
        )

    if order_id and is_logistics_query(message):
        logistics = get_logistics(order_id)
        order = get_order(order_id)
        context = {**order, **logistics}
        answer = (
            f"订单 {order_id} 的物流由{logistics['carrier']}承运，运单号 {logistics['tracking_no']}。"
            f"最新状态：{logistics['latest_status']}，当前位置：{logistics['latest_location']}。"
        )
        return CustomerServiceDecision(
            answer=answer,
            order_context=context,
            needs_human_transfer=False,
            transfer_reason="",
            quality_score=88,
            tool_name="get_logistics",
        )

    if order_id and is_order_query(message):
        order = get_order(order_id)
        if order["status"] == "未找到":
            answer = f"没有查到订单 {order_id}，请核对订单号后再试。"
        else:
            answer = (
                f"已查询到订单 {order_id}：当前状态为{order['status']}，商品是 {order['product']}，"
                f"预计{order['estimated_delivery']}送达。"
            )
        return CustomerServiceDecision(
            answer=answer,
            order_context=order,
            needs_human_transfer=False,
            transfer_reason="",
            quality_score=88,
            tool_name="get_order",
        )

    if is_product_query(message):
        product = get_product(message)
        if product["availability"] == "未找到":
            answer = "我暂时没有找到对应商品信息，可以换一个商品名再查。"
        else:
            answer = (
                f"{product['name']} 当前{product['availability']}，售价 {product['price']} {product['currency']}，"
                f"{product['shipping']}。"
            )
        return CustomerServiceDecision(
            answer=answer,
            order_context=None,
            needs_human_transfer=False,
            transfer_reason="",
            quality_score=84,
            tool_name="get_product",
        )

    return CustomerServiceDecision(
        answer="我是智能客服，可以帮你查询订单、物流、商品库存，或在你确认后提交退款申请。",
        order_context=None,
        needs_human_transfer=False,
        transfer_reason="",
        quality_score=72,
        tool_name="fallback",
    )
