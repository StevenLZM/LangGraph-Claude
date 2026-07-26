from __future__ import annotations

from typing import Any, Mapping, Sequence

from langchain_core.documents import Document

from config import rag_config


def fuse_business_features(
    query: str,
    documents: Sequence[Document],
    *,
    time_intent: Mapping[str, Any] | None,
) -> list[Document]:
    """Combine comparable normalized relevance and business features."""
    del query
    if not documents:
        return []

    rerank_values = [_number(doc, "rerank_score") for doc in documents]
    rrf_values = [_number(doc, "rrf_score") for doc in documents]
    authority_values = [
        _clamp(_number(doc, "authority_score", default=0.5))
        for doc in documents
    ]
    version_values = [_number(doc, "version_rank") for doc in documents]
    freshness_values = _freshness_values(documents, time_intent)
    normalized = {
        "rerank": _min_max(rerank_values),
        "rrf": _min_max(rrf_values),
        "authority": authority_values,
        "freshness": freshness_values,
        "version": _min_max(version_values),
    }
    weights = rag_config.BUSINESS_SCORE_WEIGHTS

    scored: list[tuple[float, int, Document]] = []
    for index, document in enumerate(documents):
        features = {
            name: values[index]
            for name, values in normalized.items()
        }
        score = sum(
            float(weights[name]) * features[name]
            for name in weights
        )
        scored.append(
            (
                score,
                index,
                Document(
                    page_content=document.page_content,
                    metadata={
                        **document.metadata,
                        "business_features": features,
                        "business_score": score,
                    },
                ),
            )
        )
    ranked = [
        document
        for _, _, document in sorted(
            scored,
            key=lambda item: (-item[0], item[1]),
        )
    ]
    return ranked[: rag_config.BUSINESS_FUSION_TOP_K]


def _freshness_values(
    documents: Sequence[Document],
    time_intent: Mapping[str, Any] | None,
) -> list[float]:
    if not time_intent or time_intent.get("type") != "latest":
        return [0.0] * len(documents)
    field = time_intent.get("field", "doc_date")
    metadata_key = "upload_date" if field == "upload_date" else "doc_date_max"
    return _min_max([_number(doc, metadata_key) for doc in documents])


def _number(
    document: Document,
    key: str,
    *,
    default: float = 0.0,
) -> float:
    value = document.metadata.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _min_max(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        fill = 1.0 if maximum != 0 else 0.0
        return [fill] * len(values)
    scale = maximum - minimum
    return [_clamp((value - minimum) / scale) for value in values]


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
