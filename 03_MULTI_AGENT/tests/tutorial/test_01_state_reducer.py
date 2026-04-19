"""配套 TUTORIAL.md 第 2 章 —— State 与 Reducer

学什么：merge_evidence 是 LangGraph 在 Annotated[list, reducer] 字段上自动调用的合并函数。
        多个并行节点同时往 evidence 字段写值时，框架不再追加，而是按本函数语义合并。

断言：
1. 同一个 source_url 的两条 evidence，保留 relevance_score 较高的一条
2. 输入混杂 None / dict / Pydantic 实例都能归一
3. 空 source_url 被跳过
4. 输出按 relevance_score 倒序排序
"""
from __future__ import annotations

from agents.schemas import Evidence
from graph.state import merge_evidence


def _ev(url: str, score: float, sq: str = "sq1", snippet: str = "x") -> Evidence:
    return Evidence(
        sub_question_id=sq,
        source_type="web",
        source_url=url,
        snippet=snippet,
        relevance_score=score,
    )


def test_dedup_keeps_highest_score():
    old = [_ev("https://a", 0.6, snippet="old")]
    new = [_ev("https://a", 0.9, snippet="new")]
    merged = merge_evidence(old, new)
    assert len(merged) == 1
    assert merged[0].relevance_score == 0.9
    assert merged[0].snippet == "new"


def test_handles_none_and_dict_mix():
    old = None
    new = [
        _ev("https://b", 0.5),
        {"sub_question_id": "sq1", "source_type": "web",
         "source_url": "https://c", "snippet": "d", "relevance_score": 0.8},
    ]
    merged = merge_evidence(old, new)
    urls = {e.source_url for e in merged}
    assert urls == {"https://b", "https://c"}


def test_skips_empty_url():
    new = [
        _ev("", 0.9),
        _ev("https://x", 0.1),
    ]
    merged = merge_evidence([], new)
    assert len(merged) == 1
    assert merged[0].source_url == "https://x"


def test_sorted_by_score_desc():
    new = [_ev(f"https://u{i}", float(i) / 10) for i in range(5)]
    merged = merge_evidence([], new)
    scores = [e.relevance_score for e in merged]
    assert scores == sorted(scores, reverse=True)
