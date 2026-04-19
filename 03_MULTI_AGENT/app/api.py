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

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from langgraph.types import Command

from app import bootstrap, report_store
from app.schemas import ResumeReq, StartReq, StartResp, TurnReq
from app.turn_init import reset_per_turn

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


def _config(tid: str) -> dict:
    return {"configurable": {"thread_id": tid}}


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
    cfg = _config(tid)
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
