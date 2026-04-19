"""Researcher 通用逻辑 —— 从 registry 降级链调工具，返回 Evidence 列表。

单个 Researcher 节点只处理一个 SubQuestion（由 Send 派发时带入）。
LLM 提炼是可选的 —— 默认直接把工具结果转成 Evidence 以节省 token；若子问题要求 source_type ∈ {academic,code,web} 且结果文本丰富，交 LLM 提炼。
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
from typing import Any

from agents.schemas import Evidence
from tools.base import SearchTool, SourceType, ToolResult

logger = logging.getLogger(__name__)


async def run_research_chain(
    *,
    source_type: SourceType,
    query: str,
    sub_question_id: str,
    registry,
    top_k: int = 5,
) -> list[Evidence]:
    """顺序尝试 registry 中该 source_type 的工具链，第一个非空结果即返回。

    失败工具会被记 WARN 但不中断链路；全部失败则返回 []。
    """
    chain: list[SearchTool] = registry.get_chain(source_type) if registry else []
    if not chain:
        logger.info("[researcher:%s] registry 无可用工具，跳过", source_type)
        return []

    for tool in chain:
        try:
            results = await asyncio.wait_for(tool.search(query, top_k=top_k), timeout=45)
        except Exception as e:
            logger.warning("[researcher:%s] tool=%s 调用失败: %s", source_type, tool.name, e)
            continue
        if results:
            logger.info(
                "[researcher:%s] tool=%s 命中 %d 条，查询=%r",
                source_type,
                tool.name,
                len(results),
                query[:60],
            )
            return _to_evidence(results, source_type=source_type, sub_question_id=sub_question_id)

    return []


def _to_evidence(
    results: list[ToolResult],
    *,
    source_type: SourceType,
    sub_question_id: str,
) -> list[Evidence]:
    now = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    evs: list[Evidence] = []
    for r in results:
        url = r.get("source_url") or ""
        if not url:
            continue
        evs.append(
            Evidence(
                sub_question_id=sub_question_id,
                source_type=source_type,
                source_url=url,
                snippet=(r.get("snippet") or "")[:2000],
                relevance_score=float(r.get("relevance_score") or 0.0),
                fetched_at=now,
            )
        )
    return evs


def extract_sq_and_query(payload: dict) -> tuple[str, str]:
    """从 Send 派发的 payload 解析 sub_question 的 id + 查询字符串。"""
    sq = payload.get("sub_question")
    fallback_q = payload.get("research_query") or ""
    if sq is None:
        return "unknown", fallback_q
    # Pydantic / dict 两种形态兼容
    sq_id = getattr(sq, "id", None) or (sq.get("id") if isinstance(sq, dict) else None) or "unknown"
    question = (
        getattr(sq, "question", None)
        or (sq.get("question") if isinstance(sq, dict) else None)
        or fallback_q
    )
    return sq_id, question
