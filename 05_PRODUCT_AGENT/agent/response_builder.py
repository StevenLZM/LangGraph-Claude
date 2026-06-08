from __future__ import annotations

from copy import deepcopy
from typing import Any

from langchain_core.messages import AIMessage

from agent.choices import create_logistics_choice, create_product_choice, create_refund_choice, public_choice_set
from agent.dialog_state import public_task_status, should_persist_dialog_state, terminal_task_status
from agent.intent import is_memory_recall_query, is_preference_statement
from agent.slot_filling import is_cancel_request
from agent.tools import get_order


def build_response_update(state: dict[str, Any]) -> dict[str, Any]:
    dialog_state = deepcopy(state.get("dialog_state"))
    route = str(state.get("route") or "")
    latest_message = str(state.get("latest_user_message") or "")
    user_memories = list(state.get("user_memories") or [])

    if route == "human_transfer":
        return _response(
            answer="我已为你标记转人工处理，会把当前问题和上下文同步给人工客服继续跟进。",
            needs_human_transfer=True,
            transfer_reason="用户要求人工或涉及投诉/法律问题",
            tool_name="human_transfer",
            task_status=None,
            dialog_state=None,
            quality_score=82,
        )

    if dialog_state:
        active_task = str(dialog_state.get("active_task") or "")
        phase = str(dialog_state.get("phase") or "")
        if phase == "cancelled":
            return _response(
                answer=_cancelled_answer(active_task),
                tool_name=active_task,
                task_status=terminal_task_status(active_task, "cancelled"),
                dialog_state=None,
                quality_score=82,
            )
        if phase == "escalated":
            return _response(
                answer="当前信息仍不够明确，我已为你转人工处理，人工客服会继续跟进。",
                needs_human_transfer=True,
                transfer_reason="多轮补充信息超过系统阈值",
                tool_name="human_transfer",
                task_status=terminal_task_status(active_task, "escalated"),
                dialog_state=None,
                quality_score=78,
            )
        if phase == "collecting_slots":
            return _collecting_response(dialog_state, user_memories)
        if phase == "awaiting_confirmation":
            return _confirmation_response(dialog_state)
        if active_task == "product" and phase == "awaiting_selection":
            return _product_selection_prompt(dialog_state)
        if active_task == "product" and phase == "executing":
            return _product_selected_response(dialog_state)

    tool_results = list(state.get("tool_results") or [])
    if tool_results:
        return _tool_response(state, tool_results)

    if is_memory_recall_query(latest_message):
        return _memory_recall_response(user_memories)
    if is_preference_statement(latest_message):
        return _response(
            answer="已记住你的偏好，后续服务会优先参考这条信息。",
            tool_name="save_user_memory",
            quality_score=82,
        )

    return _response(
        answer="我是智能客服，可以帮你查询订单、物流、商品库存，或在你确认后提交退款申请。",
        tool_name="fallback",
        quality_score=72,
    )


def _collecting_response(dialog_state: dict[str, Any], user_memories: list[str]) -> dict[str, Any]:
    active_task = str(dialog_state.get("active_task") or "")
    missing = list(dialog_state.get("missing_slots") or [])
    if active_task == "logistics":
        choice_set = create_logistics_choice(user_memories)
        memory_text = "；".join(user_memories[:2])
        if memory_text:
            answer = (
                f"我会优先参考你的配送偏好：{memory_text}。"
                "不过实际使用哪家快递要以具体订单为准，请提供订单号后我可以查询实际承运商。"
            )
        else:
            answer = "请提供订单号，例如 ORD123456，我可以继续查询这单物流状态。"
        return _response(
            answer=answer,
            choices=public_choice_set(choice_set),
            dialog_state=dialog_state,
            task_status=public_task_status(dialog_state),
            tool_name="delivery_preference",
            quality_score=82,
        )
    if active_task == "refund":
        if "order_id" in missing:
            answer = "可以处理退货/退款。请提供需要处理的订单号，例如 ORD123456。"
        elif "reason" in missing:
            answer = "已收到订单号。请补充退货原因，例如不想要了、买错了或商品质量问题。"
        elif "product_condition" in missing:
            answer = "请确认商品状态，例如商品完好、未拆封，或说明是否存在破损。"
        else:
            answer = "请继续补充退货/退款信息。"
        return _response(
            answer=answer,
            dialog_state=dialog_state,
            task_status=public_task_status(dialog_state),
            tool_name="apply_refund",
            quality_score=76,
        )
    return _response(
        answer="请继续补充信息，我会接着处理当前任务。",
        dialog_state=dialog_state,
        task_status=public_task_status(dialog_state),
        tool_name=active_task,
        quality_score=72,
    )


def _confirmation_response(dialog_state: dict[str, Any]) -> dict[str, Any]:
    slots = dict(dialog_state.get("collected_slots") or {})
    order_id = str(slots.get("order_id") or "")
    order = get_order(order_id)
    if order.get("status") == "未找到":
        return _response(
            answer=f"没有查到订单 {order_id}，请核对订单号后再试。",
            order_context={"order_id": order_id, "refund_status": "order_not_found"},
            dialog_state=None,
            task_status=terminal_task_status("refund", "cancelled"),
            tool_name="apply_refund",
            quality_score=72,
        )
    refund_context = {
        **order,
        "refund_status": "confirmation_required",
        "message": "退款会进入人工复核，请确认是否继续提交退款申请。",
    }
    choice_set = create_refund_choice(order_id, refund_context)
    return _response(
        answer=(
            f"订单 {order_id} 当前可以发起退款/退货申请。"
            "提交后会进入人工复核，请选择“确认退款”“暂不退款”或“转人工处理”。"
        ),
        order_context=refund_context,
        choices=public_choice_set(choice_set),
        dialog_state=dialog_state,
        task_status=public_task_status(dialog_state),
        tool_name="apply_refund",
        quality_score=86,
    )


def _product_selection_prompt(dialog_state: dict[str, Any]) -> dict[str, Any]:
    product = dict((dialog_state.get("collected_slots") or {}).get("product") or {})
    choice_set = create_product_choice(product)
    return _response(
        answer=(
            f"{product.get('name', '该商品')} 当前{product.get('availability', '库存状态未知')}，"
            f"售价 {product.get('price', 0)} {product.get('currency', 'CNY')}，"
            f"{product.get('shipping', '')}。你也可以继续选择查看库存、发货时效或转人工处理。"
        ),
        choices=public_choice_set(choice_set),
        dialog_state=dialog_state,
        task_status=public_task_status(dialog_state),
        tool_name="get_product",
        quality_score=84,
    )


def _tool_response(state: dict[str, Any], tool_results: list[dict[str, Any]]) -> dict[str, Any]:
    dialog_state = state.get("dialog_state")
    active_task = str((dialog_state or {}).get("active_task") or "")
    latest_by_name = {str(result.get("tool_name") or ""): result for result in tool_results}

    if "apply_refund" in latest_by_name:
        output = dict(latest_by_name["apply_refund"].get("output") or {})
        order_id = str(output.get("order_id") or "")
        if output.get("refund_status") == "submitted":
            answer = (
                f"订单 {order_id} 的退款申请已提交，工单号 {output.get('refund_ticket_id')}。"
                "预计 1-3 个工作日内完成审核。"
            )
            status = terminal_task_status("refund", "completed")
        else:
            answer = f"没有查到订单 {order_id}，请核对订单号后再试。"
            status = terminal_task_status("refund", "cancelled")
        return _response(
            answer=answer,
            order_context=output,
            task_status=status,
            dialog_state=None,
            tool_name="apply_refund",
            quality_score=88,
        )

    if active_task == "product" and dialog_state:
        return _product_selected_response(dialog_state)

    if "get_logistics" in latest_by_name:
        logistics = dict(latest_by_name["get_logistics"].get("output") or {})
        order = dict(latest_by_name.get("get_order", {}).get("output") or {})
        context = {**order, **logistics}
        if "faq_rag" in latest_by_name:
            faq_output = dict(latest_by_name["faq_rag"].get("output") or {})
            context.update(_rag_context(faq_output))
        order_id = str(logistics.get("order_id") or order.get("order_id") or "")
        answer = (
            f"订单 {order_id} 的物流由{logistics.get('carrier', '')}承运，"
            f"运单号 {logistics.get('tracking_no', '')}。"
            f"最新状态：{logistics.get('latest_status', '')}，当前位置：{logistics.get('latest_location', '')}。"
        )
        if "faq_rag" in latest_by_name:
            faq_answer = str((latest_by_name["faq_rag"].get("output") or {}).get("answer") or "")
            if faq_answer:
                answer = f"{answer} 关于退货/退款资格：{faq_answer}"
        task_status = terminal_task_status("logistics", "completed") if active_task == "logistics" else None
        return _response(
            answer=answer,
            order_context=context,
            task_status=task_status,
            dialog_state=None,
            tool_name="get_logistics",
            quality_score=88,
        )

    if "get_order" in latest_by_name:
        order = dict(latest_by_name["get_order"].get("output") or {})
        order_id = str(order.get("order_id") or "")
        if order.get("status") == "未找到":
            answer = f"没有查到订单 {order_id}，请核对订单号后再试。"
        else:
            answer = (
                f"已查询到订单 {order_id}：当前状态为{order.get('status')}，商品是 {order.get('product')}，"
                f"预计{order.get('estimated_delivery')}送达。"
            )
        return _response(answer=answer, order_context=order, tool_name="get_order", quality_score=88)

    if "get_product" in latest_by_name:
        product = dict(latest_by_name["get_product"].get("output") or {})
        if product.get("availability") == "未找到":
            return _response(answer="我暂时没有找到对应商品信息，可以换一个商品名再查。", tool_name="get_product", quality_score=72)
        product_state = {
            "active_task": "product",
            "phase": "awaiting_selection",
            "required_slots": [],
            "collected_slots": {"product": product},
            "missing_slots": [],
            "confirmation": {"required": False, "confirmed": False, "action": "get_product", "preview": ""},
            "attempt_count": 0,
            "expires_at": str((dialog_state or {}).get("expires_at") or ""),
            "idempotency_key": "get_product",
        }
        return _product_selection_prompt(product_state)

    if "faq_rag" in latest_by_name:
        output = dict(latest_by_name["faq_rag"].get("output") or {})
        context = _rag_context(output)
        return _response(answer=str(output.get("answer") or ""), order_context=context, tool_name="faq_rag", quality_score=86)

    return _response(answer="已收到，我会继续帮你处理。", tool_name="fallback", quality_score=72)


def _product_selected_response(dialog_state: dict[str, Any]) -> dict[str, Any]:
    slots = dict(dialog_state.get("collected_slots") or {})
    product = dict(slots.get("product") or {})
    selected = str(slots.get("selected_option") or "")
    if selected == "check_stock":
        answer = f"{product.get('name', '该商品')} 当前{product.get('availability', '库存状态未知')}。"
    elif selected == "check_shipping":
        answer = f"{product.get('name', '该商品')} {product.get('shipping', '暂时没有发货时效信息')}。"
    elif is_cancel_request(selected):
        answer = "已取消商品后续操作。"
    else:
        answer = "已收到你的选择。"
    return _response(
        answer=answer,
        dialog_state=None,
        task_status=terminal_task_status("product", "completed"),
        tool_name="get_product",
        quality_score=84,
    )


def _memory_recall_response(user_memories: list[str]) -> dict[str, Any]:
    memory_text = "；".join(user_memories[:3])
    if not memory_text:
        return _response(answer="我暂时没有查到你已保存的相关记忆。", tool_name="load_user_memory", quality_score=72)
    return _response(answer=f"我记得这些信息：{memory_text}", tool_name="load_user_memory", quality_score=86)


def _rag_context(output: dict[str, Any]) -> dict[str, Any]:
    context = {
        "rag_matched": output.get("matched", False),
        "rag_sources": output.get("sources", []),
        "rag_backend": output.get("backend", ""),
    }
    if output.get("error"):
        context["rag_error"] = output["error"]
    return context


def _cancelled_answer(active_task: str) -> str:
    if active_task == "refund":
        return "已取消本次退货/退款申请。如需继续处理，可以再次告诉我。"
    if active_task == "logistics":
        return "已取消本次物流查询。"
    return "已取消当前任务。"


def _response(
    *,
    answer: str,
    order_context: dict[str, Any] | None = None,
    choices: dict[str, Any] | None = None,
    dialog_state: dict[str, Any] | None = None,
    task_status: dict[str, Any] | None = None,
    needs_human_transfer: bool = False,
    transfer_reason: str = "",
    tool_name: str,
    quality_score: int,
) -> dict[str, Any]:
    persisted_dialog_state = dialog_state if should_persist_dialog_state(dialog_state) else None
    return {
        "messages": [AIMessage(content=answer, additional_kwargs={"tool_name": tool_name})],
        "order_context": order_context,
        "choices": choices,
        "dialog_state": persisted_dialog_state,
        "pending_choice": None,
        "task_status": task_status if task_status is not None else public_task_status(dialog_state),
        "needs_human_transfer": needs_human_transfer,
        "transfer_reason": transfer_reason,
        "quality_score": quality_score,
        "tool_name": tool_name,
        "token_used": max(1, len(answer) // 3),
    }
