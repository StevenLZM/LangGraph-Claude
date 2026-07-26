"""Elasticsearch BM25 + Dense + application RRF with Parent hydration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.documents import Document

from config import rag_config
from rag.docstore import ParentDocStore, get_parent_docstore
from rag.elasticsearch_retrievers import (
    ElasticsearchBM25Retriever,
    ElasticsearchDenseRetriever,
    HybridChildRetriever,
)
from rag.embedder import get_embeddings
from rag.reranker import rerank_documents
from rag.vectorstore import build_time_filter, get_vectorstore


_HARD_TYPES = {"year", "before", "after", "range"}
_SOFT_TYPES = {"latest"}


def _is_hard(time_intent: Optional[dict]) -> bool:
    return bool(time_intent) and time_intent.get("type") in _HARD_TYPES


def _is_soft(time_intent: Optional[dict]) -> bool:
    return bool(time_intent) and time_intent.get("type") in _SOFT_TYPES


@dataclass
class ParentChildHybridRetriever:
    ensemble_retriever: object
    parent_docstore: ParentDocStore
    time_intent: Optional[dict] = None

    def invoke(self, query: str) -> List[Document]:
        try:
            child_hits = self.ensemble_retriever.invoke(
                query,
                config={"callbacks": [RetrievalLoggingHandler()]},
            )
        except TypeError as exc:
            if "config" not in str(exc):
                raise
            child_hits = self.ensemble_retriever.invoke(query)
        child_hits = list(child_hits)[: rag_config.RRF_TOP_K]
        return hydrate_parent_results(
            child_hits,
            parent_docstore=self.parent_docstore,
            limit=rag_config.MAX_HYDRATED_PARENTS,
            time_intent=self.time_intent,
        )


def reset_retriever_cache() -> None:
    """Compatibility no-op: Elasticsearch owns the persistent retrieval index."""


def build_hybrid_retriever(
    all_chunks: List[Document] | None = None,
    parent_docstore: ParentDocStore | None = None,
    time_intent: Optional[dict] = None,
    *,
    vectorstore: object | None = None,
    embeddings: object | None = None,
) -> ParentChildHybridRetriever:
    """
    Build ES BM25 and Dense retrievers using one shared Metadata Filter.

    ``all_chunks`` remains only for source compatibility and is never read.
    """
    del all_chunks
    store = vectorstore or get_vectorstore()
    docstore = parent_docstore or get_parent_docstore()
    metadata_filter = build_time_filter(time_intent) if _is_hard(time_intent) else None
    bm25 = ElasticsearchBM25Retriever(
        store=store,
        top_k=rag_config.BM25_TOP_K,
        metadata_filter=metadata_filter,
    )
    dense = ElasticsearchDenseRetriever(
        store=store,
        embeddings=embeddings or get_embeddings(),
        top_k=rag_config.SEMANTIC_TOP_K,
        metadata_filter=metadata_filter,
    )
    hybrid = HybridChildRetriever(
        bm25=bm25,
        dense=dense,
        bm25_weight=1 - rag_config.SEMANTIC_WEIGHT,
        dense_weight=rag_config.SEMANTIC_WEIGHT,
        rrf_k=rag_config.RRF_K,
        top_k=rag_config.RRF_TOP_K,
    )
    return ParentChildHybridRetriever(
        hybrid,
        docstore,
        time_intent=time_intent,
    )


def get_hybrid_retriever(
    time_intent: Optional[dict] = None,
) -> ParentChildHybridRetriever | None:
    """Return a hybrid retriever without scanning all Child documents."""
    try:
        store = get_vectorstore()
        if store.count_children(status="active") <= 0:
            return None
        return build_hybrid_retriever(
            parent_docstore=get_parent_docstore(),
            time_intent=time_intent,
            vectorstore=store,
        )
    except Exception:
        return None


def hydrate_parent_results(
    child_hits: List[Document],
    parent_docstore: ParentDocStore,
    limit: int | None = None,
    time_intent: Optional[dict] = None,
) -> List[Document]:
    """Aggregate ordered Child hits into hydrated Parent documents."""
    if not child_hits:
        return []

    scores: dict[str, float] = {}
    matched_child_ids: dict[str, list[str]] = {}
    best_meta: dict[str, dict] = {}
    for rank, child in enumerate(child_hits, start=1):
        parent_id = str(child.metadata.get("parent_id") or "")
        child_id = str(child.metadata.get("child_id") or "")
        if not parent_id:
            continue
        score = _child_rank_score(child, rank)
        scores[parent_id] = scores.get(parent_id, 0.0) + score
        matched_child_ids.setdefault(parent_id, [])
        if child_id and child_id not in matched_child_ids[parent_id]:
            matched_child_ids[parent_id].append(child_id)
        current_best = best_meta.get(parent_id)
        if current_best is None or score > current_best["score"]:
            best_meta[parent_id] = {
                "score": score,
                "metadata": dict(child.metadata),
            }

    sort_key = _make_sort_key(
        scores,
        matched_child_ids,
        best_meta,
        time_intent,
    )
    ranked_parent_ids = sorted(scores, key=sort_key)
    if limit is not None:
        ranked_parent_ids = ranked_parent_ids[:limit]

    parents = parent_docstore.get_parents(ranked_parent_ids)
    results: list[Document] = []
    for parent_id in ranked_parent_ids:
        parent_doc = parents.get(parent_id)
        if parent_doc is None:
            continue
        best = best_meta[parent_id]
        metadata = {
            **parent_doc.metadata,
            "matched_child_ids": matched_child_ids.get(parent_id, []),
            "best_child_score": round(best["score"], 6),
            "section_path": parent_doc.metadata.get(
                "section_path",
                best["metadata"].get("section_path", "未命名章节"),
            ),
            "page_range": parent_doc.metadata.get(
                "page_range",
                best["metadata"].get("page_range", "?"),
            ),
        }
        results.append(
            Document(
                page_content=parent_doc.page_content,
                metadata=metadata,
            )
        )
    return results


def _child_rank_score(child: Document, rank: int) -> float:
    for key in ("business_score", "rerank_score", "rrf_score", "similarity_score"):
        value = child.metadata.get(key)
        if value is not None:
            return float(value)
    return 1.0 / (rag_config.RRF_K + rank)


def _make_sort_key(
    scores: dict[str, float],
    matched_child_ids: dict[str, list[str]],
    best_meta: dict[str, dict],
    time_intent: Optional[dict],
) -> Callable[[str], tuple]:
    if not time_intent or time_intent.get("type") == "none":
        return lambda parent_id: (
            -scores[parent_id],
            -len(matched_child_ids.get(parent_id, [])),
        )

    field_name = time_intent.get("field", "doc_date")

    def date_for(parent_id: str) -> int:
        metadata = best_meta[parent_id]["metadata"]
        if field_name == "upload_date":
            return int(metadata.get("upload_date", 0) or 0)
        return int(metadata.get("doc_date_max", 0) or 0)

    if _is_soft(time_intent):
        return lambda parent_id: (
            -date_for(parent_id),
            -scores[parent_id],
            -len(matched_child_ids.get(parent_id, [])),
        )
    return lambda parent_id: (
        -scores[parent_id],
        -len(matched_child_ids.get(parent_id, [])),
        -date_for(parent_id),
    )


def retrieve_with_hybrid(
    query: str,
    top_k: int | None = None,
    ensemble_retriever: Optional[ParentChildHybridRetriever] = None,
    time_intent: Optional[dict] = None,
) -> List[Document]:
    """Retrieve Parent documents and apply the existing phase-A reranker."""
    if ensemble_retriever is None or (
        time_intent is not None
        and getattr(ensemble_retriever, "time_intent", None) != time_intent
    ):
        ensemble_retriever = get_hybrid_retriever(time_intent=time_intent)
        if ensemble_retriever is None:
            return []
    limit = top_k or rag_config.FINAL_TOP_K
    results = ensemble_retriever.invoke(query)
    reranked = rerank_documents(query, results, top_n=limit)
    return reranked[:limit]


class RetrievalLoggingHandler(BaseCallbackHandler):
    """Compact LangChain-compatible retrieval logging callback."""

    def __init__(self) -> None:
        self._name_stack: list[str] = []

    def on_retriever_start(
        self,
        serialized,
        query,
        *,
        run_id=None,
        parent_run_id=None,
        **kwargs,
    ) -> None:
        name = self._extract_name(serialized, kwargs)
        self._name_stack.append(name)
        print(f"\n🔍 [{name}] 开始检索: {query[:80]}...")

    def on_retriever_end(
        self,
        response,
        *,
        run_id=None,
        parent_run_id=None,
        **kwargs,
    ) -> None:
        name = self._name_stack.pop() if self._name_stack else "Unknown"
        print(f"✅ [{name}] 完成，返回 {len(response)} 个文档")

    def _extract_name(self, serialized, kwargs) -> str:
        for tag in kwargs.get("tags") or []:
            if tag.startswith("retriever:"):
                return tag.split(":", 1)[1]
        if serialized and isinstance(serialized, dict):
            return serialized.get("name") or "Unknown"
        return "Unknown"
