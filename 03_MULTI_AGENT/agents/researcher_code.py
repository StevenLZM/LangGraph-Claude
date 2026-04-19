"""Code Researcher —— GitHub Search。"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from agents._researcher_base import extract_sq_and_query, run_research_chain
from agents._safe import safe_node
from config.tracing import with_tags


@with_tags("code_researcher")
@safe_node
async def code_researcher_node(payload: dict) -> dict[str, Any]:
    from app.bootstrap import app_state

    sq_id, question = extract_sq_and_query(payload)
    evidence = await run_research_chain(
        source_type="code",
        query=question,
        sub_question_id=sq_id,
        registry=app_state.registry,
        top_k=5,
    )
    return {
        "evidence": evidence,
        "messages": [AIMessage(content=f"[code:{sq_id}] 收集 {len(evidence)} 条")],
    }
