from __future__ import annotations

from typing import Any

from agent.tools import apply_refund, get_logistics, get_order, get_product
from agent.tool_planner import advance_plan, current_step
from rag.faq_tool import FAQRAGTool


_faq_rag_tool = FAQRAGTool()


def execute_next_tool(state: dict[str, Any]) -> dict[str, Any]:
    tool_plan = dict(state.get("tool_plan") or {})
    step = current_step(tool_plan)
    if step is None:
        return {"tool_plan": tool_plan, "tool_results": state.get("tool_results", []), "tool_trace": state.get("tool_trace", [])}

    tool_name = str(step.get("tool_name") or "")
    args = dict(step.get("args") or {})
    read_only = bool(step.get("read_only", False))
    try:
        output = _execute(tool_name, args)
        status = "success"
        error = ""
    except Exception as exc:
        output = {}
        status = "error"
        error = str(exc)

    result = {
        "tool_name": tool_name,
        "args": _public_args(args),
        "read_only": read_only,
        "status": status,
        "output": output,
        "error": error,
    }
    return {
        "tool_plan": advance_plan(tool_plan),
        "tool_results": [*list(state.get("tool_results") or []), result],
        "tool_trace": [
            *list(state.get("tool_trace") or []),
            {
                "tool_name": tool_name,
                "read_only": read_only,
                "status": status,
                "error": error,
            },
        ],
    }


def _execute(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "get_order":
        return get_order(str(args.get("order_id") or ""))
    if tool_name == "get_logistics":
        return get_logistics(str(args.get("order_id") or ""))
    if tool_name == "get_product":
        return get_product(str(args.get("query") or ""))
    if tool_name == "apply_refund":
        return apply_refund(
            str(args.get("order_id") or ""),
            confirmed=bool(args.get("confirmed", False)),
            idempotency_key=str(args.get("idempotency_key") or ""),
        )
    if tool_name == "faq_rag":
        faq_result = _faq_rag_tool.search(str(args.get("query") or ""))
        return {
            "answer": faq_result.answer,
            "matched": faq_result.matched,
            "sources": faq_result.sources,
            "backend": faq_result.backend,
            "error": faq_result.error,
        }
    if tool_name == "load_user_memory":
        return {"memories": list(args.get("user_memories") or [])}
    raise ValueError(f"unsupported tool: {tool_name}")


def _public_args(args: dict[str, Any]) -> dict[str, Any]:
    public = dict(args)
    public.pop("idempotency_key", None)
    return public
