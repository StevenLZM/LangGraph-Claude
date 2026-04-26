"""LangGraph astream_events → SSE 事件映射。

设计要点（详见 plans/tidy-hatching-gem.md §1）：
- 只对 langgraph_node == "writer" 的 on_chat_model_stream 发 token，避免 planner/reflector 的中间 token 轰炸前端
- on_chain_end 只挑 evidence_count / next_action / revision_count 等关键字段
- interrupt 检测：astream_events 不会在 output 里漏 __interrupt__，被截断的节点也不发 on_chain_end；
  必须在流结束后用 graph.aget_state(cfg) 从 state.tasks[*].interrupts 取
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Iterable, Optional

from agents.schemas import ResearchPlan

NODE_NAMES = {
    "planner",
    "supervisor",
    "web_researcher",
    "academic_researcher",
    "code_researcher",
    "kb_researcher",
    "reflector",
    "writer",
}


def _summarize_node_output(output: Any) -> dict:
    """从节点输出里挑关键字段返回前端，避免把整个 state dump 出去。"""
    if not isinstance(output, dict):
        return {}
    keep = {}
    for k in (
        "next_action",
        "revision_count",
        "iteration",
        "current_node",
        "plan_confirmed",
        "report_path",
    ):
        if k in output and output[k] is not None:
            keep[k] = output[k]
    if "evidence" in output and output["evidence"]:
        keep["evidence_count"] = len(output["evidence"])
    if "plan" in output and output["plan"]:
        keep["plan_size"] = len(output["plan"])
    if "final_report" in output and output["final_report"]:
        keep["has_report"] = True
    return keep


def _extract_interrupt_payload(output: Any) -> dict | None:
    """（保留兼容）从 dict 里读 __interrupt__；实际 astream_events 不会暴露此字段。"""
    if not isinstance(output, dict):
        return None
    intr = output.get("__interrupt__")
    if not intr:
        return None
    item = intr[0] if isinstance(intr, (list, tuple)) else intr
    val = getattr(item, "value", None)
    if val is None and isinstance(item, dict):
        val = item.get("value")
    return val


async def _detect_pending_interrupt(graph: Any, cfg: dict) -> dict | None:
    """流结束后检查线程是否挂起在 interrupt；这是 LangGraph 1.x 唯一可靠的途径。"""
    if graph is None:
        return None
    try:
        state = await graph.aget_state(cfg)
    except Exception:
        return None
    if not getattr(state, "tasks", None):
        return None
    for task in state.tasks:
        intrs = getattr(task, "interrupts", None) or ()
        for intr in intrs:
            val = getattr(intr, "value", None)
            if val is None and isinstance(intr, dict):
                val = intr.get("value")
            if val is not None:
                return val
    return None


def map_event(ev: dict) -> dict | None:
    """单个 LangGraph 事件 → SSE event dict（{event, data}）。无关事件返回 None。"""
    etype = ev.get("event")
    name = ev.get("name") or ""
    metadata = ev.get("metadata") or {}
    data = ev.get("data") or {}

    if etype == "on_chain_start" and name in NODE_NAMES:
        return {"event": "node_start", "data": {"node": name}}

    if etype == "on_chain_end" and name in NODE_NAMES:
        return {
            "event": "node_end",
            "data": {"node": name, "summary": _summarize_node_output(data.get("output"))},
        }

    if etype == "on_chain_end" and name == "LangGraph":
        # 顶层图 output 在被 interrupt 截断时也不包含 __interrupt__；
        # interrupt 由调用方在流结束后通过 aget_state 检测，这里保留兼容路径但实际不会命中
        intr = _extract_interrupt_payload(data.get("output"))
        if intr is not None:
            return {"event": "interrupt", "data": intr}
        return None

    if etype in {"on_tool_start", "on_tool_end"}:
        node = metadata.get("langgraph_node", "")
        if not node:
            return None
        phase = "start" if etype == "on_tool_start" else "end"
        return {"event": "tool", "data": {"node": node, "tool": name, "phase": phase}}

    if etype == "on_chat_model_stream":
        if metadata.get("langgraph_node") != "writer":
            return None
        chunk = data.get("chunk")
        text = getattr(chunk, "content", "") or ""
        if not text:
            return None
        return {"event": "token", "data": {"text": text}}

    return None


async def stream_events(
    events: AsyncIterator[dict],
    *,
    thread_id: str,
    graph: Optional[Any] = None,
    cfg: Optional[dict] = None,
    final_state_capture: dict | None = None,
) -> AsyncIterator[dict]:
    """包裹 astream_events 异步迭代器，yield SSE event dict 序列。

    流结束后用 graph.aget_state(cfg) 检测挂起的 interrupt；有则发 interrupt 事件并跳过 done。
    """
    yield {"event": "thread", "data": {"thread_id": thread_id}}
    final_output: dict = {}
    try:
        async for ev in events:
            mapped = map_event(ev)
            if ev.get("event") == "on_chain_end" and ev.get("name") == "LangGraph":
                out = (ev.get("data") or {}).get("output")
                if isinstance(out, dict):
                    final_output = out
            if mapped is not None:
                yield mapped
    except Exception as e:
        yield {"event": "error", "data": {"message": str(e), "type": type(e).__name__}}
        return

    if final_state_capture is not None:
        final_state_capture.update(final_output)

    pending = await _detect_pending_interrupt(graph, cfg) if cfg else None
    if pending is not None:
        yield {"event": "interrupt", "data": pending}
        return

    yield {
        "event": "done",
        "data": {
            "final_report": final_output.get("final_report"),
            "report_path": final_output.get("report_path"),
        },
    }


def coerce_plan_payload(raw: Any) -> dict:
    """把 resume_stream 收到的 plan 字符串 / dict 转成 Command(resume=...) 期望的字典。"""
    if raw is None:
        return {}
    if isinstance(raw, ResearchPlan):
        return {"plan": raw.model_dump()}
    if isinstance(raw, str):
        import json
        raw = json.loads(raw)
    if isinstance(raw, dict):
        if "sub_questions" in raw:
            return {"plan": ResearchPlan.model_validate(raw).model_dump()}
        if "plan" in raw:
            inner = raw["plan"]
            if isinstance(inner, dict) and "sub_questions" in inner:
                return {"plan": ResearchPlan.model_validate(inner).model_dump()}
    raise ValueError(f"无法解析 plan payload: {type(raw)}")


def event_node_set() -> Iterable[str]:
    return frozenset(NODE_NAMES)
