from __future__ import annotations

from typing import Any, Callable, Iterable, Protocol

from langchain_core.documents import Document


RAGAS_METRIC_NAMES = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_correctness",
    "answer_relevancy",
    "semantic_similarity",
]


class RagasEvaluator(Protocol):
    def __call__(
        self,
        rows: list[dict[str, Any]],
        metric_names: list[str],
    ) -> list[dict[str, Any]]:
        ...


def build_ragas_row(
    case: dict[str, Any],
    docs: list[Any],
    response: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized_docs = [_ensure_document(doc) for doc in docs]
    return {
        "id": str(case["id"]),
        "category": str(case["category"]),
        "query_type": case.get("query_type", ""),
        "user_input": str(case["question"]),
        "retrieved_contexts": [
            doc.page_content for doc in normalized_docs if doc.page_content
        ],
        "response": _extract_answer(response),
        "reference": str(case["reference"]),
        "retrieved_sources": _metadata_values(normalized_docs, "source"),
        "retrieved_parent_ids": _metadata_values(normalized_docs, "parent_id"),
        "retrieved_sections": _metadata_values(normalized_docs, "section_path"),
    }


def run_ragas_evaluation(
    rows: list[dict[str, Any]],
    *,
    metric_names: Iterable[str] | None = None,
    evaluator: RagasEvaluator | None = None,
) -> list[dict[str, Any]]:
    metrics = list(metric_names or RAGAS_METRIC_NAMES)
    if evaluator is not None:
        return list(evaluator(rows, metrics))
    return _run_real_ragas(rows, metrics)


def _run_real_ragas(
    rows: list[dict[str, Any]],
    metric_names: list[str],
) -> list[dict[str, Any]]:
    try:
        from datasets import Dataset
        from ragas import evaluate
    except ImportError as exc:
        raise ImportError(
            "RAGAS 评估依赖未安装。请先在 langgraph-cc-multiagent 环境执行："
            "pip install -r requirements.txt"
        ) from exc

    dataset = Dataset.from_list(
        [
            {
                "user_input": row["user_input"],
                "retrieved_contexts": row["retrieved_contexts"],
                "response": row["response"],
                "reference": row["reference"],
            }
            for row in rows
        ]
    )
    result = evaluate(
        dataset=dataset,
        metrics=_load_ragas_metrics(metric_names),
        llm=_build_ragas_llm(),
        embeddings=_build_ragas_embeddings(),
    )
    score_rows = _score_rows_to_dicts(result)

    merged: list[dict[str, Any]] = []
    for row, score in zip(rows, score_rows):
        merged.append({**row, **score})
    return merged


def _load_ragas_metrics(metric_names: list[str]) -> list[Any]:
    try:
        from ragas.metrics.collections import (
            AnswerCorrectness,
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
            SemanticSimilarity,
        )
    except ImportError:
        try:
            from ragas.metrics import (
                AnswerCorrectness,
                Faithfulness,
                LLMContextPrecisionWithReference,
                LLMContextRecall,
                ResponseRelevancy,
                SemanticSimilarity,
            )
        except ImportError:
            from ragas.metrics import (
                answer_relevancy,
                answer_correctness,
                context_precision,
                context_recall,
                faithfulness,
            )

            legacy_metrics = {
                "context_precision": context_precision,
                "context_recall": context_recall,
                "faithfulness": faithfulness,
                "answer_correctness": answer_correctness,
                "answer_relevancy": answer_relevancy,
            }
            missing = [name for name in metric_names if name not in legacy_metrics]
            if missing:
                raise ImportError(
                    "当前 RAGAS 版本缺少以下指标类，请升级 ragas："
                    + ", ".join(missing)
                )
            return [legacy_metrics[name] for name in metric_names]

    metric_factories: dict[str, Callable[[], Any]] = {
        "context_precision": LLMContextPrecisionWithReference,
        "context_recall": LLMContextRecall,
        "faithfulness": Faithfulness,
        "answer_correctness": AnswerCorrectness,
        "answer_relevancy": ResponseRelevancy,
        "semantic_similarity": SemanticSimilarity,
    }
    unknown = [name for name in metric_names if name not in metric_factories]
    if unknown:
        raise ValueError(f"unknown RAGAS metric names: {unknown}")
    return [metric_factories[name]() for name in metric_names]


def _build_ragas_llm() -> Any:
    try:
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise ImportError("当前 RAGAS 版本缺少 LangchainLLMWrapper，请升级 ragas。") from exc

    from rag.chain import _get_llm

    return LangchainLLMWrapper(_get_llm())


def _build_ragas_embeddings() -> Any:
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as exc:
        raise ImportError(
            "当前 RAGAS 版本缺少 LangchainEmbeddingsWrapper，请升级 ragas。"
        ) from exc

    from rag.embedder import get_embeddings

    return LangchainEmbeddingsWrapper(get_embeddings())


def _score_rows_to_dicts(result: Any) -> list[dict[str, Any]]:
    if hasattr(result, "to_pandas"):
        records = result.to_pandas().to_dict(orient="records")
        return [_metric_fields(record) for record in records]
    if hasattr(result, "to_list"):
        return [_metric_fields(record) for record in result.to_list()]
    if isinstance(result, list):
        return [_metric_fields(record) for record in result]
    if isinstance(result, dict):
        return _columnar_metric_rows(result)
    raise TypeError(f"unsupported RAGAS result type: {type(result)!r}")


def _columnar_metric_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    metric_columns = {
        _normalize_metric_name(key): value
        for key, value in result.items()
        if _normalize_metric_name(key) in RAGAS_METRIC_NAMES
        or _normalize_metric_name(key).startswith("ragas_")
    }
    if not metric_columns:
        return []
    row_count = max(
        len(value)
        for value in metric_columns.values()
        if isinstance(value, list)
    ) if any(isinstance(value, list) for value in metric_columns.values()) else 1
    rows: list[dict[str, Any]] = []
    for index in range(row_count):
        row: dict[str, Any] = {}
        for key, value in metric_columns.items():
            row[key] = value[index] if isinstance(value, list) else value
        rows.append(row)
    return rows


def _metric_fields(record: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key, value in record.items():
        normalized_key = _normalize_metric_name(key)
        if normalized_key in RAGAS_METRIC_NAMES or normalized_key.startswith("ragas_"):
            fields[normalized_key] = value
    return fields


def _normalize_metric_name(name: str) -> str:
    aliases = {
        "llm_context_precision_with_reference": "context_precision",
        "context_precision_with_reference": "context_precision",
    }
    return aliases.get(name, name)


def _ensure_document(item: Any) -> Document:
    if isinstance(item, Document):
        return item
    if isinstance(item, dict):
        return Document(
            page_content=str(item.get("page_content") or item.get("content") or ""),
            metadata=dict(item.get("metadata") or {}),
        )
    return Document(page_content=str(item), metadata={})


def _extract_answer(response: dict[str, Any] | None) -> str:
    if not response:
        return ""
    return str(response.get("answer") or response.get("response") or "")


def _metadata_values(docs: list[Document], key: str) -> list[str]:
    values: list[str] = []
    for doc in docs:
        value = (doc.metadata or {}).get(key)
        if value is None:
            continue
        text = str(value)
        if text and text not in values:
            values.append(text)
    return values
