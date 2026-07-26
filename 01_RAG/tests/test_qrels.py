import pytest

from evals.models import EvidenceQrel
from evals.qrels import (
    EvidenceMatch,
    hash_evidence_text,
    match_candidate,
    normalize_evidence_text,
    page_ranges_overlap,
)
from rag.retrieval_trace import CandidateSnapshot


def _candidate(**overrides) -> CandidateSnapshot:
    values = {
        "rank": 1,
        "level": "child",
        "doc_id": "doc-1",
        "parent_id": "parent-1",
        "child_id": "child-1",
        "source": "manual.pdf",
        "page_range": "3",
        "section": "保修政策",
        "score": 1.0,
        "score_type": "bm25_score",
        "page_content": "本产品的保修期为 12 个月，从购买日起计算。",
    }
    values.update(overrides)
    return CandidateSnapshot(**values)


def _qrel(**overrides) -> EvidenceQrel:
    text = overrides.pop("evidence_text", "保修期为 12 个月")
    values = {
        "evidence_id": "warranty",
        "source": "manual.pdf",
        "page_range": "3",
        "section": "保修政策",
        "evidence_text": text,
        "evidence_hash": hash_evidence_text(text),
        "grade": 3,
    }
    values.update(overrides)
    return EvidenceQrel(**values)


def test_normalize_evidence_text_ignores_whitespace_and_common_punctuation():
    assert normalize_evidence_text("产品保修期：12 个月。") == "产品保修期12个月"


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [("3", "3", True), ("2-4", "4-5", True), ("1-2", "3", False)],
)
def test_page_ranges_overlap(left, right, expected):
    assert page_ranges_overlap(left, right) is expected


def test_match_candidate_requires_source_page_and_evidence_text():
    assert match_candidate(_candidate(), [_qrel()]) == (
        EvidenceMatch(evidence_id="warranty", grade=3),
    )


@pytest.mark.parametrize(
    "candidate",
    [
        _candidate(source="other.pdf"),
        _candidate(page_range="7"),
        _candidate(page_content="本页没有相关保修信息"),
    ],
)
def test_match_candidate_rejects_wrong_anchor(candidate):
    assert match_candidate(candidate, [_qrel()]) == ()


def test_match_candidate_uses_source_basename():
    candidate = _candidate(source="/indexed/corpus/manual.pdf")
    qrel = _qrel(source="uploads/manual.pdf")

    assert match_candidate(candidate, [qrel]) == (
        EvidenceMatch(evidence_id="warranty", grade=3),
    )


def test_match_candidate_rejects_corrupted_evidence_hash():
    with pytest.raises(ValueError, match="evidence_hash"):
        match_candidate(_candidate(), [_qrel(evidence_hash="corrupted")])
