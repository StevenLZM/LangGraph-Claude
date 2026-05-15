from __future__ import annotations

import math
from typing import Any, Iterable

from langchain_core.documents import Document


def evaluate_retrieval_case(
    case: dict[str, Any],
    docs: list[Document],
    *,
    k_values: Iterable[int] = (3, 5),
) -> dict[str, Any]:
    relevant_parent_ids = set(case.get("expected_parent_ids") or [])
    relevant_sources = set(case.get("expected_sources") or [])
    relevant_sections = set(case.get("expected_sections") or [])
    expected_keywords = list(case.get("expected_keywords") or [])

    retrieved_parent_ids = [_metadata_value(doc, "parent_id") for doc in docs]
    retrieved_sources = [_metadata_value(doc, "source") for doc in docs]
    retrieved_sections = [_metadata_value(doc, "section_path") for doc in docs]

    relevance = [
        _is_relevant(doc, relevant_parent_ids, relevant_sources, relevant_sections)
        for doc in docs
    ]
    relevant_total = max(1, _expected_relevant_count(case))

    result: dict[str, Any] = {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "retrieved_parent_ids": [item for item in retrieved_parent_ids if item],
        "retrieved_sources": [item for item in retrieved_sources if item],
        "retrieved_sections": [item for item in retrieved_sections if item],
        "parent_hit": bool(relevant_parent_ids and relevant_parent_ids.intersection(retrieved_parent_ids)),
        "source_hit": bool(relevant_sources and relevant_sources.intersection(retrieved_sources)),
        "section_hit": bool(relevant_sections and relevant_sections.intersection(retrieved_sections)),
        "mrr": _mrr(relevance),
        "ndcg@5": _ndcg(relevance, k=5, relevant_total=relevant_total),
        "context_completeness": _keyword_coverage(_joined_content(docs), expected_keywords),
        "time_intent_accuracy": _time_intent_accuracy(case),
        "time_filter_accuracy": _time_filter_accuracy(case, docs),
        "top_result_source": retrieved_sources[0] if retrieved_sources else "",
        "top_result_parent_id": retrieved_parent_ids[0] if retrieved_parent_ids else "",
    }

    for k in k_values:
        result[f"recall@{k}"] = _recall_at_k(relevance, k=k, relevant_total=relevant_total)

    result["retrieval_score"] = _retrieval_score(result)
    result["passed"] = result["retrieval_score"] >= 80
    return result


def evaluate_generation_case(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    answer = str(response.get("answer") or "")
    sources = _extract_response_sources(response)
    answer_keywords = list(case.get("answer_keywords") or case.get("expected_keywords") or [])
    forbidden_keywords = list(case.get("forbidden_keywords") or [])
    expected_sources = set(case.get("expected_sources") or [])

    forbidden_hits = [keyword for keyword in forbidden_keywords if keyword in answer]
    keyword_coverage = _keyword_coverage(answer, answer_keywords)
    source_accuracy = (
        len(expected_sources.intersection(sources)) / len(expected_sources)
        if expected_sources
        else 1.0
    )
    no_answer_ok = _no_answer_behavior_ok(case, answer)

    score = round(
        keyword_coverage * 65
        + source_accuracy * 25
        + (10 if no_answer_ok else 0)
        - min(30, len(forbidden_hits) * 15)
    )
    score = max(0, min(100, score))
    return {
        "answer": answer,
        "answer_keyword_coverage": round(keyword_coverage, 4),
        "citation_source_accuracy": round(source_accuracy, 4),
        "forbidden_keyword_hits": forbidden_hits,
        "no_answer_behavior_ok": no_answer_ok,
        "generation_score": score,
        "generation_passed": score >= 80,
    }


def _is_relevant(
    doc: Document,
    relevant_parent_ids: set[str],
    relevant_sources: set[str],
    relevant_sections: set[str],
) -> bool:
    metadata = doc.metadata or {}
    parent_id = str(metadata.get("parent_id") or "")
    source = str(metadata.get("source") or "")
    section = str(metadata.get("section_path") or "")
    if relevant_parent_ids:
        return parent_id in relevant_parent_ids
    if relevant_sections:
        return section in relevant_sections
    if relevant_sources:
        return source in relevant_sources
    return False


def _expected_relevant_count(case: dict[str, Any]) -> int:
    return max(
        len(case.get("expected_parent_ids") or []),
        len(case.get("expected_sections") or []),
        len(case.get("expected_sources") or []),
        1,
    )


def _recall_at_k(relevance: list[bool], *, k: int, relevant_total: int) -> float:
    hits = sum(1 for item in relevance[:k] if item)
    return round(min(1.0, hits / relevant_total), 4)


def _mrr(relevance: list[bool]) -> float:
    for index, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            return round(1 / index, 4)
    return 0.0


def _ndcg(relevance: list[bool], *, k: int, relevant_total: int) -> float:
    gains = [1.0 if item else 0.0 for item in relevance[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_count = min(k, relevant_total)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
    if idcg == 0:
        return 0.0
    return round(dcg / idcg, 4)


def _keyword_coverage(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    matched = [keyword for keyword in keywords if keyword in text]
    return round(len(matched) / len(keywords), 4)


def _joined_content(docs: list[Document]) -> str:
    return "\n".join(doc.page_content or "" for doc in docs)


def _time_filter_accuracy(case: dict[str, Any], docs: list[Document]) -> float | None:
    expected = case.get("expected_time_range")
    if not expected:
        return None
    if not docs:
        return 0.0
    passed = sum(1 for doc in docs if _doc_passes_time_range(doc, expected))
    return round(passed / len(docs), 4)


def _time_intent_accuracy(case: dict[str, Any]) -> float | None:
    expected = case.get("expected_time_intent")
    if not expected:
        return None
    actual = case.get("actual_time_intent")
    if not actual:
        return 0.0
    keys = ("type", "field", "range", "sort")
    return 1.0 if all(actual.get(key) == expected.get(key) for key in keys) else 0.0


def _doc_passes_time_range(doc: Document, expected: dict[str, Any]) -> bool:
    metadata = doc.metadata or {}
    field = expected.get("field", "doc_date")
    gte = int(expected.get("gte", 0))
    lte = int(expected.get("lte", 99991231))

    if field == "upload_date":
        value = int(metadata.get("upload_date") or 0)
        return gte <= value <= lte

    if not metadata.get("has_doc_date", False):
        return False
    doc_min = int(metadata.get("doc_date_min") or 0)
    doc_max = int(metadata.get("doc_date_max") or 0)
    return doc_min <= lte and doc_max >= gte


def _extract_response_sources(response: dict[str, Any]) -> set[str]:
    source_items = response.get("sources") or []
    sources: set[str] = set()
    for item in source_items:
        if isinstance(item, Document):
            source = item.metadata.get("source")
        elif isinstance(item, dict):
            source = item.get("source") or item.get("metadata", {}).get("source")
        else:
            source = None
        if source:
            sources.add(str(source))
    return sources


def _no_answer_behavior_ok(case: dict[str, Any], answer: str) -> bool:
    if not case.get("expected_no_answer", False):
        return True
    refusal_markers = ["未找到", "没有相关信息", "无法根据当前知识库", "当前知识库"]
    return any(marker in answer for marker in refusal_markers)


def _retrieval_score(result: dict[str, Any]) -> int:
    recall = float(result.get("recall@5", result.get("recall@3", 0.0)) or 0.0)
    mrr = float(result.get("mrr") or 0.0)
    parent_hit = 1.0 if result.get("parent_hit") else 0.0
    completeness = float(result.get("context_completeness") or 0.0)
    time_accuracy = result.get("time_filter_accuracy")
    intent_accuracy = result.get("time_intent_accuracy")
    time_values = [
        float(value)
        for value in (time_accuracy, intent_accuracy)
        if value is not None
    ]
    time_component = sum(time_values) / len(time_values) if time_values else 1.0
    score = recall * 35 + mrr * 25 + parent_hit * 15 + completeness * 15 + time_component * 10
    return round(score)


def _metadata_value(doc: Document, key: str) -> str:
    value = (doc.metadata or {}).get(key)
    return str(value) if value is not None else ""
