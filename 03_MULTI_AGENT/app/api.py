"""FastAPI 入口 —— interrupt/resume/turn 真实实现。

路由：
  POST /research               启动新研究；若命中 interrupt，返回 interrupt payload 与 thread_id
  POST /research/{tid}/resume  Command(resume=...) 恢复被 interrupt 的图
  POST /research/{tid}/turn    已有会话追问（复用历史 evidence）
  GET  /research/{tid}/state   调试：查看当前 state
  GET  /threads                列出已有 thread（从 checkpointer）
  GET  /health                 健康检查
"""
from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from langgraph.types import Command
from sse_starlette.sse import EventSourceResponse

from app import bootstrap, report_store, sse
from app.schemas import ResumeReq, StartReq, StartResp, TurnReq
from app.turn_init import reset_per_turn
from config.settings import settings

logger = logging.getLogger(__name__)


async def _invoke(payload, cfg):
    return await bootstrap.app_state.graph.ainvoke(payload, config=cfg)


async def _aget_state(cfg):
    return await bootstrap.app_state.graph.aget_state(cfg)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await bootstrap.startup()
    yield
    await bootstrap.shutdown()


app = FastAPI(title="InsightLoop", version="0.1.0", lifespan=lifespan)


def _graph():
    g = bootstrap.app_state.graph
    if g is None:
        raise HTTPException(503, "graph 尚未就绪")
    return g


def _config(tid: str, *, query: str | None = None, audience: str | None = None) -> dict:
    """RunnableConfig：thread_id 走 configurable，业务字段走 metadata 让 LangSmith 可筛选。"""
    metadata: dict[str, Any] = {"thread_id": tid, "app": "insightloop"}
    if query is not None:
        metadata["research_query"] = query
    if audience is not None:
        metadata["audience"] = audience
    return {"configurable": {"thread_id": tid}, "metadata": metadata}


def _extract_interrupt(result: dict[str, Any]) -> dict | None:
    """从 ainvoke 结果解析 __interrupt__（LangGraph 0.6 的结构）。"""
    intr = result.get("__interrupt__") if isinstance(result, dict) else None
    if not intr:
        return None
    # 兼容 list[Interrupt] 与单个 Interrupt
    item = intr[0] if isinstance(intr, list) else intr
    return getattr(item, "value", None) or (item.get("value") if isinstance(item, dict) else None)


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "graph_ready": bootstrap.app_state.graph is not None,
        "checkpointer": bootstrap.app_state.checkpointer is not None,
        "registry": repr(bootstrap.app_state.registry),
    }


@app.post("/research", response_model=StartResp)
async def start_research(req: StartReq) -> StartResp:
    tid = uuid.uuid4().hex[:12]
    cfg = _config(tid, query=req.research_query, audience=req.audience)
    payload = {
        "research_query": req.research_query,
        "audience": req.audience,
        "messages": [],
        "evidence": [],
    }
    result = await _invoke(payload, cfg)
    interrupt_val = _extract_interrupt(result)
    return StartResp(
        thread_id=tid,
        interrupt=interrupt_val,
        final_report=result.get("final_report") if interrupt_val is None else None,
        report_path=result.get("report_path") if interrupt_val is None else None,
    )


@app.post("/research/{thread_id}/resume", response_model=StartResp)
async def resume_research(thread_id: str, req: ResumeReq) -> StartResp:
    cfg = _config(thread_id)
    result = await _invoke(Command(resume={"plan": req.plan.model_dump()}), cfg)
    interrupt_val = _extract_interrupt(result)
    return StartResp(
        thread_id=thread_id,
        interrupt=interrupt_val,
        final_report=result.get("final_report") if interrupt_val is None else None,
        report_path=result.get("report_path") if interrupt_val is None else None,
    )


@app.post("/research/{thread_id}/turn", response_model=StartResp)
async def turn_research(thread_id: str, req: TurnReq) -> StartResp:
    """同会话追问：保留历史 evidence / plan，重置易变字段。"""
    cfg = _config(thread_id)
    patch = reset_per_turn({}, req.research_query)
    patch["plan_confirmed"] = False  # 触发重新走 planner（复用历史 evidence 需 Planner 决策）
    result = await _invoke(patch, cfg)
    interrupt_val = _extract_interrupt(result)
    return StartResp(
        thread_id=thread_id,
        interrupt=interrupt_val,
        final_report=result.get("final_report") if interrupt_val is None else None,
        report_path=result.get("report_path") if interrupt_val is None else None,
    )


@app.get("/research/{thread_id}/state")
async def get_state(thread_id: str) -> dict:
    cfg = _config(thread_id)
    snap = await _aget_state(cfg)
    values = snap.values or {}
    # 去掉 messages 等大对象，只返关键字段
    return {
        "thread_id": thread_id,
        "research_query": values.get("research_query"),
        "plan_confirmed": values.get("plan_confirmed"),
        "revision_count": values.get("revision_count"),
        "next_action": values.get("next_action"),
        "evidence_count": len(values.get("evidence") or []),
        "has_report": bool(values.get("final_report")),
        "report_path": values.get("report_path"),
        "next": list(snap.next) if snap.next else [],
    }


@app.get("/threads")
async def list_threads(limit: int = 50) -> dict:
    """列出已有 thread_id（从 checkpointer 扫描）。"""
    ckpt = bootstrap.app_state.checkpointer
    if ckpt is None:
        return {"threads": []}
    seen: dict[str, dict] = {}
    try:
        # SqliteSaver.list 返回 CheckpointTuple 迭代器
        for ct in ckpt.list(None, limit=limit * 4):
            tid = ct.config["configurable"]["thread_id"]
            if tid not in seen:
                values = ct.checkpoint.get("channel_values", {})
                seen[tid] = {
                    "thread_id": tid,
                    "last_query": values.get("research_query", ""),
                    "has_report": bool(values.get("final_report")),
                }
            if len(seen) >= limit:
                break
    except Exception as e:
        logger.warning("[api] list_threads 失败: %s", e)
    return {"threads": list(seen.values())}


@app.get("/reports")
async def list_reports(limit: int = 20) -> dict:
    return {"reports": report_store.list_reports(limit=limit)}


@app.get("/reports/read")
async def read_report(path: str) -> dict:
    try:
        return {"path": path, "content": report_store.read_report(path)}
    except FileNotFoundError:
        raise HTTPException(404, f"报告不存在: {path}")


# ─────────────────────────── SSE 流式端点 ───────────────────────────
# 设计文档：plans/tidy-hatching-gem.md
# 用 graph.astream_events(version="v2") 把 LangGraph 内部事件转成 SSE
# 旧的 /research /resume /turn 全部保留，互不影响


def _sse_response(thread_id: str, payload, cfg) -> EventSourceResponse:
    g = _graph()

    async def gen():
        events = g.astream_events(payload, config=cfg, version="v2")
        async for item in sse.stream_events(events, thread_id=thread_id, graph=g, cfg=cfg):
            yield {"event": item["event"], "data": json.dumps(item["data"], ensure_ascii=False)}

    return EventSourceResponse(gen(), ping=15, send_timeout=settings.sse_retry_ms / 1000)


@app.get("/research/stream")
async def research_stream(
    query: str = Query(..., description="研究问题"),
    audience: str = Query("intermediate"),
):
    """启动新研究并流式推送 LangGraph 事件。命中 interrupt 时发 interrupt 事件后停止；
    前端拿到 thread_id 后调 /research/{tid}/resume_stream 继续。"""
    tid = uuid.uuid4().hex[:12]
    cfg = _config(tid, query=query, audience=audience)
    payload = {
        "research_query": query,
        "audience": audience,
        "messages": [],
        "evidence": [],
    }
    return _sse_response(tid, payload, cfg)


@app.get("/research/{thread_id}/resume_stream")
async def resume_stream(
    thread_id: str,
    plan: str = Query(..., description="JSON 序列化后的 ResearchPlan"),
):
    cfg = _config(thread_id)
    resume_payload = sse.coerce_plan_payload(plan)
    return _sse_response(thread_id, Command(resume=resume_payload), cfg)


@app.get("/research/{thread_id}/turn_stream")
async def turn_stream(
    thread_id: str,
    query: str = Query(..., description="追问问题"),
):
    cfg = _config(thread_id, query=query)
    patch = reset_per_turn({}, query)
    patch["plan_confirmed"] = False
    return _sse_response(thread_id, patch, cfg)
