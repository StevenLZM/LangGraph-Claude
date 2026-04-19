"""Writer —— LLM 撰写 + 引用列表 + 落盘归档。

为提升 LLM 引用质量：
  - 把 evidence 编号传给 LLM，要求用 [^N] 引用
  - citations 由后端从 evidence 直接生成（保证编号 ↔ url 一一对应）

注意：此节点用 sync invoke。原因同 graph/workflow.py 顶部说明（Py3.9 异步 contextvars 缺陷）。
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agents.schemas import Citation
from app import report_store
from config.llm import get_llm
from graph.state import ResearchState
from prompts.templates import WRITER_SYSTEM, writer_user

logger = logging.getLogger(__name__)


async def writer_node(state: ResearchState, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
    query = state.get("research_query", "")
    audience = state.get("audience", "intermediate")
    plan = state.get("plan") or []
    evidence = state.get("evidence") or []

    plan_summary = "\n".join(f"- {sq.id}: {sq.question}" for sq in plan) or "(无)"

    numbered = []
    citations: list[Citation] = []
    for i, ev in enumerate(evidence, 1):
        title = (ev.snippet or "")[:80].replace("\n", " ")
        numbered.append(f"[{i}] ({ev.source_type}) {ev.source_url}\n    {ev.snippet[:400]}")
        citations.append(Citation(idx=i, source_url=ev.source_url, title=title))
    numbered_evidence = "\n\n".join(numbered) or "(无 evidence — 请基于通用知识谨慎给出概述并明示)"

    llm = get_llm("max", temperature=0.4)
    resp = await llm.ainvoke(
        [
            SystemMessage(content=WRITER_SYSTEM),
            HumanMessage(content=writer_user(query, audience, plan_summary, numbered_evidence)),
        ]
    )
    report_md = (resp.content if hasattr(resp, "content") else str(resp)).strip()

    if not _has_citation_section(report_md) and citations:
        report_md += "\n\n## 引用\n" + "\n".join(f"[^{c.idx}]: {c.source_url}" for c in citations)

    thread_id = (config or {}).get("configurable", {}).get("thread_id", "unknown")
    try:
        path = report_store.save(query, thread_id, report_md)
        logger.info("[writer] report saved → %s", path)
    except Exception as e:
        path = ""
        logger.warning("[writer] 报告落盘失败: %s", e)

    return {
        "final_report": report_md,
        "citations": citations,
        "report_path": path,
        "current_node": "writer",
        "messages": [AIMessage(content=f"报告已生成（{len(report_md)} 字, {len(citations)} 引用）")],
    }


_CITATION_HEADER = re.compile(r"^##\s*(引用|参考(文献)?|references?)", re.IGNORECASE | re.MULTILINE)


def _has_citation_section(md: str) -> bool:
    return bool(_CITATION_HEADER.search(md))
