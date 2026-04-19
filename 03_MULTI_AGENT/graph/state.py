"""ResearchState —— 主图共享状态。Reducer 规则见 ENGINEERING.md §5.1。

Evidence reducer 升级：
  - 不再用 operator.add（会保留所有重复 URL）
  - 改为 merge_evidence: 按 source_url 去重 + 按 relevance_score 倒序排序
  - 同 URL 多次命中时保留 relevance_score 最高的一条（合并 extra）
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from agents.schemas import Citation, Evidence, SubQuestion


def _to_dict(e: Any) -> dict:
    if hasattr(e, "model_dump"):
        return e.model_dump()
    if isinstance(e, dict):
        return dict(e)
    raise TypeError(f"Evidence-like object expected, got {type(e)}")


def _from_dict(d: dict) -> Evidence:
    if isinstance(d, Evidence):
        return d
    return Evidence(**d)


def merge_evidence(old: list[Any] | None, new: list[Any] | None) -> list[Evidence]:
    """Reducer：按 source_url 去重 + 按 relevance_score 倒序。"""
    pool = list(old or []) + list(new or [])
    by_url: dict[str, dict] = {}
    for raw in pool:
        try:
            d = _to_dict(raw)
        except TypeError:
            continue
        url = d.get("source_url") or ""
        if not url:
            continue
        cur = by_url.get(url)
        if cur is None or d.get("relevance_score", 0.0) > cur.get("relevance_score", 0.0):
            by_url[url] = d
    merged = [_from_dict(d) for d in by_url.values()]
    merged.sort(key=lambda e: -float(e.relevance_score or 0.0))
    return merged


class ResearchState(TypedDict, total=False):
    research_query: str
    audience: str

    plan: list[SubQuestion]
    plan_confirmed: bool

    evidence: Annotated[list[Evidence], merge_evidence]
    revision_count: int

    coverage_by_subq: dict[str, int]
    missing_aspects: list[str]
    next_action: str
    additional_queries: list[str]

    final_report: str
    citations: list[Citation]
    report_path: str

    messages: Annotated[list[BaseMessage], add_messages]

    next_node: str
    current_node: str
    iteration: int


class ResearcherPayload(TypedDict, total=False):
    sub_question: Any
    research_query: str
