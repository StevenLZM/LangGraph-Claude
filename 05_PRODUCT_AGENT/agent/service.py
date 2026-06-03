from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.intent import (
    extract_order_id,
    is_human_transfer_request,
    is_faq_query,
    is_logistics_query,
    is_memory_recall_query,
    is_order_query,
    is_preference_statement,
    is_product_query,
    is_refund_confirmed,
    is_refund_request,
)
from agent.choices import (
    create_logistics_choice,
    create_product_choice,
    create_refund_choice,
    public_choice_set,
    resolve_choice,
)
from agent.tools import apply_refund, get_logistics, get_order, get_product
from rag.faq_tool import FAQRAGTool


faq_rag_tool = FAQRAGTool()


@dataclass(frozen=True)
class CustomerServiceDecision:
    answer: str
    order_context: dict[str, Any] | None
    needs_human_transfer: bool
    transfer_reason: str
    quality_score: int
    tool_name: str
    choices: dict[str, Any] | None = None
    pending_choice: dict[str, Any] | None = None


def handle_customer_message(
    message: str,
    user_memories: list[str] | None = None,
    pending_choice: dict[str, Any] | None = None,
) -> CustomerServiceDecision:
    user_memories = list(user_memories or [])
    order_id = extract_order_id(message)

    if is_human_transfer_request(message):
        return CustomerServiceDecision(
            answer="我已为你标记转人工处理，会把当前问题和上下文同步给人工客服继续跟进。",
            order_context=get_order(order_id) if order_id else None,
            needs_human_transfer=True,
            transfer_reason="用户要求人工或涉及投诉/法律问题",
            quality_score=82,
            tool_name="human_transfer",
            pending_choice=None,
        )

    choice_decision = _handle_pending_choice(message, pending_choice, user_memories)
    if choice_decision is not None:
        return choice_decision

    if is_memory_recall_query(message):
        memory_text = "；".join(user_memories[:3])
        if not memory_text:
            return CustomerServiceDecision(
                answer="我暂时没有查到你已保存的相关记忆。",
                order_context=None,
                needs_human_transfer=False,
                transfer_reason="",
                quality_score=72,
                tool_name="load_user_memory",
                pending_choice=None,
            )
        return CustomerServiceDecision(
            answer=f"我记得这些信息：{memory_text}",
            order_context=None,
            needs_human_transfer=False,
            transfer_reason="",
            quality_score=86,
            tool_name="load_user_memory",
            pending_choice=None,
        )

    if is_preference_statement(message):
        return CustomerServiceDecision(
            answer="已记住你的偏好，后续服务会优先参考这条信息。",
            order_context=None,
            needs_human_transfer=False,
            transfer_reason="",
            quality_score=82,
            tool_name="save_user_memory",
            pending_choice=None,
        )

    if is_faq_query(message):
        rag_result = faq_rag_tool.search(message)
        context = {
            "rag_matched": rag_result.matched,
            "rag_sources": rag_result.sources,
            "rag_backend": rag_result.backend,
        }
        if rag_result.error:
            context["rag_error"] = rag_result.error
        return CustomerServiceDecision(
            answer=rag_result.answer,
            order_context=context,
            needs_human_transfer=False,
            transfer_reason="",
            quality_score=86 if rag_result.matched else 72,
            tool_name="faq_rag",
            pending_choice=None,
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
                pending_choice=None,
            )

        refund = apply_refund(order_id, confirmed=is_refund_confirmed(message))
        if refund["refund_status"] == "submitted":
            answer = (
                f"订单 {order_id} 的退款申请已提交，工单号 {refund['refund_ticket_id']}。"
                "预计 1-3 个工作日内完成审核。"
            )
            choices = None
            next_pending_choice = None
        elif refund["refund_status"] == "order_not_found":
            answer = f"没有查到订单 {order_id}，请核对订单号后再试。"
            choices = None
            next_pending_choice = None
        else:
            choice_set = create_refund_choice(order_id, refund)
            answer = (
                f"订单 {order_id} 当前可以发起退款。退款会进入人工复核，"
                "请选择“确认退款”“暂不退款”或“转人工处理”。"
            )
            choices = public_choice_set(choice_set)
            next_pending_choice = choice_set
        return CustomerServiceDecision(
            answer=answer,
            order_context=refund,
            needs_human_transfer=False,
            transfer_reason="",
            quality_score=86,
            tool_name="apply_refund",
            choices=choices,
            pending_choice=next_pending_choice,
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
            pending_choice=None,
        )

    if is_logistics_query(message):
        choice_set = create_logistics_choice(user_memories)
        if user_memories:
            memory_text = "；".join(user_memories[:2])
            answer = (
                f"我会优先参考你的配送偏好：{memory_text}。"
                "不过实际使用哪家快递要以具体订单为准，请提供订单号后我可以查询实际承运商。"
            )
        else:
            answer = "请提供订单号，例如 ORD123456，我可以帮你查询这单实际使用哪家快递。"
        return CustomerServiceDecision(
            answer=answer,
            order_context=None,
            needs_human_transfer=False,
            transfer_reason="",
            quality_score=82,
            tool_name="delivery_preference",
            choices=public_choice_set(choice_set),
            pending_choice=choice_set,
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
            pending_choice=None,
        )

    if is_product_query(message):
        product = get_product(message)
        if product["availability"] == "未找到":
            answer = "我暂时没有找到对应商品信息，可以换一个商品名再查。"
            choices = None
            next_pending_choice = None
        else:
            choice_set = create_product_choice(product)
            answer = (
                f"{product['name']} 当前{product['availability']}，售价 {product['price']} {product['currency']}，"
                f"{product['shipping']}。你也可以继续选择查看库存、发货时效或转人工处理。"
            )
            choices = public_choice_set(choice_set)
            next_pending_choice = choice_set
        return CustomerServiceDecision(
            answer=answer,
            order_context=None,
            needs_human_transfer=False,
            transfer_reason="",
            quality_score=84,
            tool_name="get_product",
            choices=choices,
            pending_choice=next_pending_choice,
        )

    return CustomerServiceDecision(
        answer="我是智能客服，可以帮你查询订单、物流、商品库存，或在你确认后提交退款申请。",
        order_context=None,
        needs_human_transfer=False,
        transfer_reason="",
        quality_score=72,
        tool_name="fallback",
        pending_choice=pending_choice,
    )


def _handle_pending_choice(
    message: str,
    pending_choice: dict[str, Any] | None,
    user_memories: list[str],
) -> CustomerServiceDecision | None:
    if not pending_choice:
        return None
    scenario = str(pending_choice.get("scenario") or "")
    order_id = extract_order_id(message)
    if scenario == "logistics" and order_id:
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
            choices=None,
            pending_choice=None,
        )

    resolved = resolve_choice(pending_choice, message)
    if resolved.get("expired"):
        return CustomerServiceDecision(
            answer="上一次选项已过期，请重新描述你的需求，我会重新处理。",
            order_context=None,
            needs_human_transfer=False,
            transfer_reason="",
            quality_score=72,
            tool_name="choice_expired",
            choices=None,
            pending_choice=None,
        )
    if not resolved.get("matched"):
        return None

    option = resolved["option"]
    option_id = str(option.get("id") or "")
    payload = dict(pending_choice.get("payload") or {})

    if option_id == "human_transfer":
        return CustomerServiceDecision(
            answer="我已为你标记转人工处理，会把当前问题和选项上下文同步给人工客服继续跟进。",
            order_context=payload,
            needs_human_transfer=True,
            transfer_reason="用户在选项中选择转人工处理",
            quality_score=82,
            tool_name="human_transfer",
            choices=None,
            pending_choice=None,
        )

    if scenario == "refund":
        order_id = str(payload.get("order_id") or "")
        if option_id == "confirm_refund":
            refund = apply_refund(order_id, confirmed=True)
            if refund["refund_status"] == "submitted":
                answer = (
                    f"订单 {order_id} 的退款申请已提交，工单号 {refund['refund_ticket_id']}。"
                    "预计 1-3 个工作日内完成审核。"
                )
            else:
                answer = f"没有查到订单 {order_id}，请核对订单号后再试。"
            return CustomerServiceDecision(
                answer=answer,
                order_context=refund,
                needs_human_transfer=False,
                transfer_reason="",
                quality_score=88,
                tool_name="apply_refund",
                choices=None,
                pending_choice=None,
            )
        if option_id == "cancel_refund":
            return CustomerServiceDecision(
                answer=f"已取消订单 {order_id} 的本次退款申请。如需继续处理，可以再次告诉我。",
                order_context={"order_id": order_id, "refund_status": "cancelled"},
                needs_human_transfer=False,
                transfer_reason="",
                quality_score=82,
                tool_name="apply_refund",
                choices=None,
                pending_choice=None,
            )

    if scenario == "logistics":
        if option_id == "show_delivery_preference":
            memory_text = "；".join(user_memories[:3])
            answer = f"当前可参考的配送偏好：{memory_text}" if memory_text else "我暂时没有查到你已保存的配送偏好。"
            return CustomerServiceDecision(
                answer=answer,
                order_context=None,
                needs_human_transfer=False,
                transfer_reason="",
                quality_score=82,
                tool_name="delivery_preference",
                choices=None,
                pending_choice=None,
            )
        if option_id == "provide_order_id":
            return CustomerServiceDecision(
                answer="请直接发送 ORD 开头的订单号，例如 ORD123456，我会继续查询物流状态。",
                order_context=None,
                needs_human_transfer=False,
                transfer_reason="",
                quality_score=78,
                tool_name="delivery_preference",
                choices=public_choice_set(pending_choice),
                pending_choice=pending_choice,
            )

    if scenario == "product":
        product = dict(payload.get("product") or {})
        if option_id == "check_stock":
            answer = f"{product.get('name', '该商品')} 当前{product.get('availability', '库存状态未知')}。"
        elif option_id == "check_shipping":
            answer = f"{product.get('name', '该商品')} {product.get('shipping', '暂时没有发货时效信息')}。"
        else:
            answer = "已收到你的选择。"
        return CustomerServiceDecision(
            answer=answer,
            order_context=None,
            needs_human_transfer=False,
            transfer_reason="",
            quality_score=84,
            tool_name="get_product",
            choices=None,
            pending_choice=None,
        )

    return None
