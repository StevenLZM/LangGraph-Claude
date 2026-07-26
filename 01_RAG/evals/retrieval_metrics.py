from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from langchain_core.documents import Document

from evals.models import EvidenceQrel
from evals.qrels import match_candidate
from rag.retrieval_trace import StageSnapshot


DEFAULT_RETRIEVAL_K = 5
RETRIEVAL_METRIC_NAMES = [
    f"retrieval_recall_at_{DEFAULT_RETRIEVAL_K}",
    "retrieval_mrr",
    f"retrieval_hit_at_{DEFAULT_RETRIEVAL_K}",
]
RETRIEVAL_METRIC_LABELS = {
    f"retrieval_recall_at_{DEFAULT_RETRIEVAL_K}": f"Recall@{DEFAULT_RETRIEVAL_K}",
    "retrieval_mrr": "MRR",
    f"retrieval_hit_at_{DEFAULT_RETRIEVAL_K}": f"Hit@{DEFAULT_RETRIEVAL_K}",
}


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


def build_retrieval_metric_fields(
    case: dict[str, Any],
    docs: list[Any],
    *,
    k: int = DEFAULT_RETRIEVAL_K,
) -> dict[str, Any]:
    expected_ids, retrieved_ids, target = _select_eval_ids(case, docs)
    if not expected_ids:
        return {
            f"retrieval_recall_at_{k}": None,
            "retrieval_mrr": None,
            f"retrieval_hit_at_{k}": None,
            "retrieval_eval_target": "none",
        }

    return {
        **compute_retrieval_metrics(
            expected_ids=expected_ids,
            retrieved_ids=retrieved_ids,
            k=k,
        ),
        "retrieval_eval_target": target,
    }


def compute_retrieval_metrics(
    *,
    expected_ids: list[Any],
    retrieved_ids: list[Any],
    k: int = DEFAULT_RETRIEVAL_K,
) -> dict[str, float]:
    expected = _normalize_unique(expected_ids)
    retrieved = _normalize_list(retrieved_ids)
    top_k = retrieved[:k]

    if not expected:
        recall = 0.0
        mrr = 0.0
        hit = 0.0
    else:
        expected_set = set(expected)
        hits = expected_set.intersection(top_k)
        recall = len(hits) / len(expected_set)
        mrr = _reciprocal_rank(retrieved, expected_set)
        hit = 1.0 if hits else 0.0

    return {
        f"retrieval_recall_at_{k}": recall,
        "retrieval_mrr": mrr,
        f"retrieval_hit_at_{k}": hit,
    }


def _select_eval_ids(case: dict[str, Any], docs: list[Any]) -> tuple[list[str], list[str], str]:
    for target, expected_key, metadata_key in (
        ("parent_id", "expected_parent_ids", "parent_id"),
        ("source", "expected_sources", "source"),
        ("section", "expected_sections", "section_path"),
    ):
        expected = _normalize_list(case.get(expected_key))
        if expected:
            return expected, _metadata_values(docs, metadata_key), target
    return [], [], "none"


def _metadata_values(docs: list[Any], key: str) -> list[str]:
    values: list[str] = []
    for doc in docs:
        normalized = _ensure_document(doc)
        values.extend(_normalize_list(normalized.metadata.get(key)))
    return values


def _ensure_document(doc: Any) -> Document:
    if isinstance(doc, Document):
        return doc
    page_content = getattr(doc, "page_content", "") or ""
    metadata = getattr(doc, "metadata", {}) or {}
    return Document(page_content=str(page_content), metadata=dict(metadata))


def _normalize_unique(values: Any) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in _normalize_list(values):
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _normalize_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, int, float, bool)):
        raw_values = [values]
    else:
        try:
            raw_values = list(values)
        except TypeError:
            raw_values = [values]
    return [str(value).strip() for value in raw_values if str(value).strip()]


def _reciprocal_rank(retrieved: list[str], expected_set: set[str]) -> float:
    for index, item in enumerate(retrieved, start=1):
        if item in expected_set:
            return 1.0 / index
    return 0.0
