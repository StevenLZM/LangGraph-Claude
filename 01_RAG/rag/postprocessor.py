from __future__ import annotations

import hashlib
import math
import unicodedata
from collections import defaultdict
from typing import Any, Sequence

from langchain_core.documents import Document

from config import rag_config
from rag.docstore import ParentDocStore


def diversify_parent_candidates(
    child_documents: Sequence[Document],
    *,
    parent_docstore: ParentDocStore,
    top_k: int = 8,
    max_parents_per_document: int = 2,
    similarity_threshold: float = 0.92,
    mmr_lambda: float = 0.70,
) -> list[Document]:
    parents = _hydrate_and_score_parents(
        child_documents,
        parent_docstore=parent_docstore,
    )
    exact_unique = _remove_exact_duplicates(parents)
    near_unique = _remove_near_duplicates(
        exact_unique,
        similarity_threshold=similarity_threshold,
    )
    quota_limited = _apply_document_quota(
        near_unique,
        max_per_document=max_parents_per_document,
    )
    return _greedy_mmr(
        quota_limited,
        top_k=top_k,
        mmr_lambda=mmr_lambda,
    )


def _hydrate_and_score_parents(
    child_documents: Sequence[Document],
    *,
    parent_docstore: ParentDocStore,
) -> list[Document]:
    grouped: dict[str, list[tuple[int, Document]]] = defaultdict(list)
    for index, child in enumerate(child_documents):
        parent_id = str(child.metadata.get("parent_id") or "").strip()
        if parent_id:
            grouped[parent_id].append((index, child))
    if not grouped:
        return []

    hydrated = parent_docstore.get_parents(grouped)
    scored: list[tuple[float, int, Document]] = []
    for parent_order, parent_id in enumerate(grouped):
        parent = hydrated.get(parent_id)
        if parent is None:
            continue
        children = grouped[parent_id]
        best_index, best_child = max(
            children,
            key=lambda item: (
                _child_score(item[1]),
                -item[0],
            ),
        )
        unique_child_ids = list(
            dict.fromkeys(
                str(child.metadata.get("child_id") or "").strip()
                for _, child in children
                if str(child.metadata.get("child_id") or "").strip()
            )
        )
        bonus = min(
            max(0, len(unique_child_ids) - 1)
            * rag_config.MULTI_EVIDENCE_BONUS_PER_CHILD,
            rag_config.MULTI_EVIDENCE_BONUS_CAP,
        )
        parent_score = _child_score(best_child) + bonus
        representative_vector = best_child.metadata.get(
            "retrieval_embedding"
        )
        metadata = {
            **parent.metadata,
            "matched_child_ids": unique_child_ids,
            "representative_child_id": best_child.metadata.get("child_id"),
            "representative_child_rank": best_index + 1,
            "representative_embedding": representative_vector,
            "best_child_score": _child_score(best_child),
            "multi_evidence_bonus": bonus,
            "parent_score": parent_score,
            "content_hash": _content_hash(parent.page_content),
        }
        scored.append(
            (
                parent_score,
                parent_order,
                Document(
                    page_content=parent.page_content,
                    metadata=metadata,
                ),
            )
        )
    return [
        document
        for _, _, document in sorted(
            scored,
            key=lambda item: (-item[0], item[1]),
        )
    ]


def _remove_exact_duplicates(
    documents: Sequence[Document],
) -> list[Document]:
    unique: list[Document] = []
    seen_hashes: set[str] = set()
    for document in documents:
        content_hash = str(document.metadata.get("content_hash") or "")
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        unique.append(document)
    return unique


def _remove_near_duplicates(
    documents: Sequence[Document],
    *,
    similarity_threshold: float,
) -> list[Document]:
    unique: list[Document] = []
    for document in documents:
        if any(
            _document_similarity(document, selected) >= similarity_threshold
            for selected in unique
        ):
            continue
        unique.append(document)
    return unique


def _apply_document_quota(
    documents: Sequence[Document],
    *,
    max_per_document: int,
) -> list[Document]:
    if max_per_document <= 0:
        return []
    counts: dict[str, int] = defaultdict(int)
    accepted: list[Document] = []
    for document in documents:
        doc_id = str(document.metadata.get("doc_id") or "")
        if counts[doc_id] >= max_per_document:
            continue
        counts[doc_id] += 1
        accepted.append(document)
    return accepted


def _greedy_mmr(
    documents: Sequence[Document],
    *,
    top_k: int,
    mmr_lambda: float,
) -> list[Document]:
    remaining = list(enumerate(documents))
    selected: list[Document] = []
    while remaining and len(selected) < max(0, top_k):
        scored: list[tuple[float, float, int, Document]] = []
        for original_index, candidate in remaining:
            relevance = _clamp(
                float(candidate.metadata.get("parent_score", 0.0))
            )
            redundancy = max(
                (
                    _document_similarity(candidate, selected_document)
                    for selected_document in selected
                ),
                default=0.0,
            )
            mmr_score = (
                mmr_lambda * relevance
                - (1.0 - mmr_lambda) * redundancy
            )
            scored.append(
                (mmr_score, relevance, -original_index, candidate)
            )
        best = max(scored, key=lambda item: item[:3])
        chosen = best[3]
        selected.append(
            Document(
                page_content=chosen.page_content,
                metadata={
                    **chosen.metadata,
                    "mmr_score": best[0],
                },
            )
        )
        remaining = [
            item
            for item in remaining
            if item[1] is not chosen
        ]
    return selected


def _child_score(document: Document) -> float:
    value = document.metadata.get("business_score")
    if value is None:
        value = document.metadata.get("rerank_score", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _document_similarity(left: Document, right: Document) -> float:
    left_vector = _numeric_vector(
        left.metadata.get("representative_embedding")
    )
    right_vector = _numeric_vector(
        right.metadata.get("representative_embedding")
    )
    if left_vector and right_vector and len(left_vector) == len(right_vector):
        return _cosine_similarity(left_vector, right_vector)
    return _shingle_jaccard(left.page_content, right.page_content)


def _numeric_vector(value: Any) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return ()


def _cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return _clamp(dot_product / (left_norm * right_norm))


def _shingle_jaccard(left: str, right: str) -> float:
    left_shingles = _character_shingles(left)
    right_shingles = _character_shingles(right)
    if not left_shingles or not right_shingles:
        return 0.0
    return len(left_shingles & right_shingles) / len(
        left_shingles | right_shingles
    )


def _character_shingles(text: str, width: int = 3) -> set[str]:
    normalized = _normalize_content(text)
    if not normalized:
        return set()
    if len(normalized) <= width:
        return {normalized}
    return {
        normalized[index : index + width]
        for index in range(len(normalized) - width + 1)
    }


def _content_hash(text: str) -> str:
    return hashlib.sha256(
        _normalize_content(text).encode("utf-8")
    ).hexdigest()


def _normalize_content(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))
