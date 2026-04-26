"""Reflector —— 真实 LLM 评分 + 决定补查/收敛。"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents._safe import safe_node
from agents.schemas import ReflectionResult
from config.llm import get_llm
from config.tracing import with_tags
from graph.state import ResearchState
from prompts.templates import REFLECTOR_SYSTEM, reflector_user

logger = logging.getLogger(__name__)
MAX_REVISION = 3


@with_tags("reflector")
@safe_node
async def reflector_node(state: ResearchState) -> dict[str, Any]:
    rc = state.get("revision_count", 0) + 1
    plan = state.get("plan") or []
    evidence = state.get("evidence") or []

    # 第 3 轮硬兜底：直接收敛，不再调用 LLM
    if rc >= MAX_REVISION:
        logger.info("[reflector] 达到最大迭代 %d，强制收敛", MAX_REVISION)
        return {
            "next_action": "force_complete",
            "revision_count": rc,
            "current_node": "reflector",
            "missing_aspects": [],
            "messages": [AIMessage(content=f"[reflector] 已达最大迭代 {MAX_REVISION}，进入写作阶段")],
        }

    plan_summary = "\n".join(
        f"- {sq.id}: {sq.question} (sources={sq.recommended_sources})" for sq in plan
    ) or "(无)"

    by_sq: dict[str, list[str]] = {}
    for ev in evidence:
        by_sq.setdefault(ev.sub_question_id, []).append(
            f"  · [{ev.source_type}] {ev.snippet[:200]} ({ev.source_url})"
        )
    evidence_summary = (
        "\n".join(f"[{sid}] (共 {len(items)} 条)\n" + "\n".join(items[:5]) for sid, items in by_sq.items())
        or "(暂无 evidence)"
    )

    llm = get_llm("max", temperature=0.0)
    structured = llm.with_structured_output(ReflectionResult, method="json_mode")
    result: ReflectionResult = await structured.ainvoke(
        [
            SystemMessage(content=REFLECTOR_SYSTEM),
            HumanMessage(content=reflector_user(plan_summary, evidence_summary, rc)),
        ]
    )
    logger.info("[reflector] rc=%d action=%s coverage=%s", rc, result.next_action, result.coverage_by_subq)
    return {
        "coverage_by_subq": result.coverage_by_subq,
        "missing_aspects": result.missing_aspects,
        "next_action": result.next_action,
        "additional_queries": result.additional_queries or [],
        "revision_count": rc,
        "current_node": "reflector",
        "messages": [
            AIMessage(
                content=f"[reflector] 第{rc}轮：{result.next_action}; 缺失={result.missing_aspects[:3]}"
            )
        ],
    }
