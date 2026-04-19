"""Internal MCP handlers —— 真实实现。

对外暴露 InsightLoop 内部能力：
  - kb_search: 复用 01_RAG 混合检索
  - list_reports: 读取 data/reports/
  - read_report: 按 thread_id 或 path 读
  - list_evidence: 从 SqliteSaver 加载某会话的 evidence
  - trigger_research: 异步触发图执行，返回 thread_id

依赖隔离：不加载 LangGraph 图到本模块的模块级常量；按需延迟构造，避免 stdio server 启动太重。
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from app import report_store
from config.settings import settings
from tools.kb_retriever import KBRetriever

logger = logging.getLogger(__name__)

_kb: KBRetriever | None = None
_graph = None
_ckpt = None


def _get_kb() -> KBRetriever:
    global _kb
    if _kb is None:
        _kb = KBRetriever()
    return _kb


def _get_graph():
    """Internal MCP 独立构建一套图（不共享 FastAPI 进程的 app_state），避免跨进程耦合。"""
    global _graph, _ckpt
    if _graph is not None:
        return _graph
    from langgraph.checkpoint.sqlite import SqliteSaver

    from graph.workflow import build_graph

    db_path = Path(settings.checkpointer_db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    _ckpt = SqliteSaver(conn)
    _graph = build_graph(checkpointer=_ckpt)
    return _graph


# ────────────────────────────────────────────────────────────────────
# 5 个 tool handler


async def kb_search(query: str, top_k: int = 5) -> list[dict]:
    kb = _get_kb()
    results = await kb.search(query, top_k=top_k)
    return [dict(r) for r in results]


async def list_reports(limit: int = 20) -> list[dict]:
    return report_store.list_reports(limit=limit)


async def read_report(thread_id: str) -> dict:
    path = report_store.find_by_thread(thread_id)
    if not path:
        return {"thread_id": thread_id, "found": False, "content": ""}
    return {"thread_id": thread_id, "found": True, "path": path, "content": report_store.read_report(path)}


async def list_evidence(thread_id: str, sub_question_id: str | None = None) -> list[dict]:
    g = _get_graph()
    snap = await g.aget_state({"configurable": {"thread_id": thread_id}})
    values = snap.values or {}
    evidence = values.get("evidence") or []
    out = []
    for ev in evidence:
        d = ev.model_dump() if hasattr(ev, "model_dump") else dict(ev)
        if sub_question_id and d.get("sub_question_id") != sub_question_id:
            continue
        out.append(d)
    return out


async def trigger_research(query: str, audience: str = "intermediate") -> dict:
    """异步触发：立即返回 thread_id；图执行在后台任务中进行。

    注意：后台任务与 stdio server 共享事件循环，图会在 Planner 处 interrupt 等待 resume。
    MCP 客户端可后续通过 list_evidence / read_report 轮询。
    """
    g = _get_graph()
    tid = uuid.uuid4().hex[:12]
    cfg = {"configurable": {"thread_id": tid}}

    async def _run():
        try:
            await g.ainvoke(
                {"research_query": query, "audience": audience, "messages": [], "evidence": []},
                config=cfg,
            )
        except Exception as e:
            logger.warning("[mcp:trigger_research] 后台执行失败 tid=%s: %s", tid, e)

    asyncio.create_task(_run())
    return {"thread_id": tid, "status": "started", "hint": "使用 list_evidence 或 read_report 查询进度"}
