from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

from evals.models import EvidenceQrel
from evals.qrels import match_candidate
from rag.retrieval_trace import StageSnapshot


@dataclass(frozen=True)
class StageMetricResult:
    stage_name: str
    configured_k: int
    actual_count: int
    relevant_evidence_count: int
    covered_evidence_count: int
    recall_at_k: float | None
    ndcg_at_k: float | None
    mrr_at_k: float | None
    hit_at_k: float | None


def compute_stage_metrics(
    stage: StageSnapshot,
    qrels: Sequence[EvidenceQrel],
    *,
    expected_behavior: Literal["answer", "abstain", "deny"] = "answer",
) -> StageMetricResult:
    relevant_by_id = {
        qrel.evidence_id: qrel
        for qrel in qrels
        if qrel.grade >= 2
    }
    relevant = tuple(relevant_by_id.values())
    top_candidates = stage.candidates[: max(0, stage.configured_k)]
    first_rank: dict[str, int] = {}

    for candidate in top_candidates:
        for match in match_candidate(candidate, relevant):
            first_rank.setdefault(match.evidence_id, candidate.rank)

    relevant_count = len(relevant)
    covered_count = len(first_rank)
    common_fields = {
        "stage_name": stage.name,
        "configured_k": stage.configured_k,
        "actual_count": len(top_candidates),
        "relevant_evidence_count": relevant_count,
        "covered_evidence_count": covered_count,
    }

    if expected_behavior != "answer":
        return StageMetricResult(
            **common_fields,
            recall_at_k=None,
            ndcg_at_k=None,
            mrr_at_k=None,
            hit_at_k=None,
        )
    if not relevant:
        return StageMetricResult(
            **common_fields,
            recall_at_k=None,
            ndcg_at_k=None,
            mrr_at_k=None,
            hit_at_k=None,
        )

    recall = covered_count / relevant_count
    first_relevant_rank = min(first_rank.values(), default=None)
    mrr = 1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
    hit = 1.0 if first_relevant_rank is not None else 0.0
    ndcg = _evidence_ndcg(
        relevant_by_id=relevant_by_id,
        first_rank=first_rank,
        configured_k=stage.configured_k,
    )
    return StageMetricResult(
        **common_fields,
        recall_at_k=_clamp(recall),
        ndcg_at_k=_clamp(ndcg),
        mrr_at_k=_clamp(mrr),
        hit_at_k=hit,
    )


def _evidence_ndcg(
    *,
    relevant_by_id: dict[str, EvidenceQrel],
    first_rank: dict[str, int],
    configured_k: int,
) -> float:
    dcg = sum(
        _gain(relevant_by_id[evidence_id].grade) / math.log2(rank + 1)
        for evidence_id, rank in first_rank.items()
        if rank <= configured_k
    )
    ideal_grades = sorted(
        (qrel.grade for qrel in relevant_by_id.values()),
        reverse=True,
    )[: max(0, configured_k)]
    idcg = sum(
        _gain(grade) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )
    return dcg / idcg if idcg else 0.0


def _gain(grade: int) -> int:
    return (2**grade) - 1


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))
