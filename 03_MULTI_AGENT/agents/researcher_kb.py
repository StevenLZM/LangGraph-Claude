"""KB Researcher —— 本地知识库混合检索（复用 01_RAG）。

Dogfooding：若 settings.use_internal_mcp_for_kb=True，走 internal MCP；否则直调 KBRetriever。
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from agents._researcher_base import extract_sq_and_query, run_research_chain
from agents._safe import safe_node
from config.settings import settings
from config.tracing import with_tags

logger = logging.getLogger(__name__)


@with_tags("kb_researcher")
@safe_node
async def kb_researcher_node(payload: dict) -> dict[str, Any]:
    from app.bootstrap import app_state

    sq_id, question = extract_sq_and_query(payload)

    if settings.use_internal_mcp_for_kb:
        # Dogfooding 路径：通过 internal MCP 的 kb_search handler
        from tools.internal_mcp import handlers as mcp_h
        try:
            results = await mcp_h.kb_search(query=question, top_k=5)
        except NotImplementedError:
            logger.warning("[kb] internal MCP kb_search 尚未实现，降级到本地 KBRetriever")
            results = None
        if results is not None:
            from agents._researcher_base import _to_evidence
            evidence = _to_evidence(
                [
                    {
                        "snippet": r.get("snippet") or r.get("page_content") or "",
                        "source_url": r.get("source_url") or "kb://local",
                        "relevance_score": float(r.get("relevance_score") or r.get("score") or 0.0),
                    }
                    for r in results
                ],
                source_type="kb",
                sub_question_id=sq_id,
            )
            return {
                "evidence": evidence,
                "messages": [AIMessage(content=f"[kb(mcp):{sq_id}] 收集 {len(evidence)} 条")],
            }

    # 默认路径：registry 本地工具
    evidence = await run_research_chain(
        source_type="kb",
        query=question,
        sub_question_id=sq_id,
        registry=app_state.registry,
        top_k=5,
    )
    return {
        "evidence": evidence,
        "messages": [AIMessage(content=f"[kb:{sq_id}] 收集 {len(evidence)} 条")],
    }
