"""Evidence reducer + 工具 contract 单测（离线，不碰 LLM/外部 HTTP）。"""
from __future__ import annotations

import pytest

from agents.schemas import Evidence
from graph.state import merge_evidence


def _ev(url: str, score: float, sq: str = "sq1") -> Evidence:
    return Evidence(
        sub_question_id=sq,
        source_type="web",
        source_url=url,
        snippet="x",
        relevance_score=score,
    )


def test_merge_evidence_dedupes_by_url_keeping_highest_score():
    merged = merge_evidence(
        [_ev("https://a", 0.3), _ev("https://b", 0.5)],
        [_ev("https://a", 0.9), _ev("https://c", 0.1)],
    )
    urls = [e.source_url for e in merged]
    scores = [e.relevance_score for e in merged]
    assert urls == ["https://a", "https://b", "https://c"]
    assert scores == [0.9, 0.5, 0.1]


def test_merge_evidence_handles_none_and_dicts():
    merged = merge_evidence(
        None,
        [
            {"sub_question_id": "sq1", "source_type": "web", "source_url": "x", "snippet": "s", "relevance_score": 0.1},
        ],
    )
    assert len(merged) == 1
    assert merged[0].source_url == "x"


def test_merge_evidence_skips_empty_urls():
    merged = merge_evidence([], [_ev("", 0.5), _ev("https://a", 0.5)])
    assert [e.source_url for e in merged] == ["https://a"]
