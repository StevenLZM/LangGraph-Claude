from __future__ import annotations

from typing import Any

from langchain_core.documents import Document


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
