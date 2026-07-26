import pytest

from evals.models import EvidenceQrel
from evals.qrels import hash_evidence_text
from evals.retrieval_metrics import compute_stage_metrics
from rag.retrieval_trace import CandidateSnapshot, StageSnapshot


def _qrel(evidence_id: str, text: str, grade: int) -> EvidenceQrel:
    return EvidenceQrel(
        evidence_id=evidence_id,
        source="manual.pdf",
        page_range="3",
        section="保修政策",
        evidence_text=text,
        evidence_hash=hash_evidence_text(text),
        grade=grade,
    )


def _candidate(rank: int, text: str) -> CandidateSnapshot:
    return CandidateSnapshot(
        rank=rank,
        level="child",
        doc_id="doc-1",
        parent_id=f"parent-{rank}",
        child_id=f"child-{rank}",
        source="manual.pdf",
        page_range="3",
        section="保修政策",
        score=1.0 / rank,
        score_type="test_score",
        page_content=text,
    )


def _stage(texts: list[str], *, configured_k: int | None = None) -> StageSnapshot:
    return StageSnapshot(
        name="bm25_child",
        candidate_level="child",
        configured_k=configured_k if configured_k is not None else len(texts),
        latency_ms=1.0,
        candidates=tuple(
            _candidate(rank, text)
            for rank, text in enumerate(texts, start=1)
        ),
    )


def test_stage_metrics_compute_recall_mrr_and_hit():
    qrels = [
        _qrel("e1", "保修期为 12 个月", 3),
        _qrel("e2", "保修从购买日起计算", 2),
    ]
    stage = _stage(["无关内容", "保修期为 12 个月", "其他内容"])

    result = compute_stage_metrics(stage, qrels)

    assert result.recall_at_k == 0.5
    assert result.mrr_at_k == 0.5
    assert result.hit_at_k == 1.0
    assert result.relevant_evidence_count == 2
    assert result.covered_evidence_count == 1


def test_ndcg_counts_repeated_evidence_only_once():
    qrels = [
        _qrel("e1", "核心证据", 3),
        _qrel("e2", "部分证据", 2),
    ]
    stage = _stage(["部分证据", "部分证据", "核心证据"])

    result = compute_stage_metrics(stage, qrels)

    assert result.ndcg_at_k == pytest.approx(0.731, abs=0.001)


@pytest.mark.parametrize("behavior", ["abstain", "deny"])
def test_non_answer_behavior_does_not_report_ir_scores(behavior):
    result = compute_stage_metrics(
        _stage(["核心证据"]),
        [_qrel("e1", "核心证据", 3)],
        expected_behavior=behavior,
    )

    assert result.recall_at_k is None
    assert result.ndcg_at_k is None
    assert result.mrr_at_k is None
    assert result.hit_at_k is None


def test_stage_metrics_never_reads_candidates_beyond_configured_k():
    stage = _stage(["无关内容", "核心证据"], configured_k=1)

    result = compute_stage_metrics(stage, [_qrel("e1", "核心证据", 3)])

    assert result.actual_count == 1
    assert result.recall_at_k == 0.0
    assert result.mrr_at_k == 0.0
    assert result.hit_at_k == 0.0
