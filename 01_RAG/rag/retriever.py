"""Elasticsearch BM25 + Dense + application RRF with Parent hydration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from functools import partial
from time import perf_counter
from typing import Any, Callable, List, Mapping, Optional
from uuid import uuid4

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.documents import Document

from config import rag_config
from rag.business_fusion import fuse_business_features
from rag.context_assembler import assemble_final_context
from rag.docstore import ParentDocStore, get_parent_docstore
from rag.elasticsearch_retrievers import (
    ElasticsearchBM25Retriever,
    ElasticsearchDenseRetriever,
    HybridChildRetriever,
    weighted_rrf,
)
from rag.embedder import get_embeddings
from rag.langsmith_tracing import (
    end_trace_span,
    serialize_documents_for_trace,
    trace_span,
)
from rag.postprocessor import diversify_parent_candidates
from rag.reranker import rerank_documents
from rag.retrieval_trace import EvaluationTrace, StageRecorder
from rag.vectorstore import build_time_filter, get_vectorstore


_HARD_TYPES = {"year", "before", "after", "range"}
_SOFT_TYPES = {"latest"}


@dataclass(frozen=True)
class RetrievalPipelineComponents:
    bm25: Callable[[str, Mapping[str, Any]], list[Document]]
    dense: Callable[[str, Mapping[str, Any]], list[Document]]
    rrf: Callable[[list[Document], list[Document]], list[Document]]
    cross_encoder: Callable[[str, list[Document]], list[Document]]
    business_fusion: Callable[
        [str, list[Document], Mapping[str, Any]],
        list[Document],
    ]
    diversify_parents: Callable[[list[Document]], list[Document]]
    assemble_context: Callable[[list[Document]], list[Document]]


@dataclass(frozen=True)
class RetrievalExecution:
    final_documents: tuple[Document, ...]
    trace: EvaluationTrace


@dataclass
class ProductionHybridRetriever:
    time_intent: Optional[dict] = None
    auth_context: dict[str, Any] | None = None

    def invoke(self, query: str) -> List[Document]:
        execution = retrieve_with_trace(
            query,
            retrieval_context={
                "time_intent": self.time_intent,
                "auth_context": dict(self.auth_context or {}),
            },
        )
        return list(execution.final_documents)


def _is_hard(time_intent: Optional[dict]) -> bool:
    return bool(time_intent) and time_intent.get("type") in _HARD_TYPES


def _is_soft(time_intent: Optional[dict]) -> bool:
    return bool(time_intent) and time_intent.get("type") in _SOFT_TYPES


def build_retrieval_metadata_filter(
    retrieval_context: Mapping[str, Any],
) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    explicit_filter = retrieval_context.get("metadata_filter")
    if isinstance(explicit_filter, Mapping):
        if set(explicit_filter) == {"$and"}:
            clauses.extend(list(explicit_filter["$and"]))
        else:
            clauses.append(dict(explicit_filter))

    auth_context = retrieval_context.get("auth_context")
    if not isinstance(auth_context, Mapping):
        auth_context = retrieval_context
    tenant_id = str(auth_context.get("tenant_id") or "").strip()
    if tenant_id:
        clauses.append({"tenant_id": tenant_id})
    principals = auth_context.get("principals")
    if isinstance(principals, (list, tuple)) and principals:
        clauses.append({"acl_principals": {"$in": list(principals)}})
    visibility = str(auth_context.get("visibility") or "").strip()
    if visibility:
        clauses.append({"visibility": visibility})

    time_filter = build_time_filter(retrieval_context.get("time_intent"))
    if time_filter:
        if set(time_filter) == {"$and"}:
            clauses.extend(time_filter["$and"])
        else:
            clauses.append(time_filter)
    return {"$and": clauses} if clauses else None


def retrieve_with_trace(
    query: str,
    *,
    retrieval_context: Mapping[str, Any],
    components: RetrievalPipelineComponents | None = None,
) -> RetrievalExecution:
    effective_context = dict(retrieval_context)
    effective_context["metadata_filter"] = build_retrieval_metadata_filter(
        retrieval_context
    )
    pipeline = components or _default_pipeline_components()
    recorder = StageRecorder(
        trace_id=str(retrieval_context.get("trace_id") or uuid4().hex),
        original_query=str(
            retrieval_context.get("original_query") or query
        ),
        rewritten_query=query,
        metadata={
            "time_intent": retrieval_context.get("time_intent"),
            "metadata_filter": effective_context["metadata_filter"],
            "auth_filter_applied": bool(
                retrieval_context.get("auth_context")
                or retrieval_context.get("tenant_id")
                or retrieval_context.get("visibility")
            ),
            "degraded_stages": [],
        },
    )

    (
        bm25_documents,
        dense_documents,
        bm25_latency,
        dense_latency,
        branch_errors,
    ) = _parallel_recall(query, effective_context, pipeline)
    recorder.trace.metadata["degraded_stages"] = sorted(branch_errors)
    recorder.trace.metadata["retrieval_errors"] = branch_errors
    recorder.record(
        name="bm25_child",
        documents=bm25_documents,
        configured_k=rag_config.BM25_TOP_K,
        score_type="bm25_score",
        latency_ms=bm25_latency,
    )
    recorder.record(
        name="dense_child",
        documents=dense_documents,
        configured_k=rag_config.SEMANTIC_TOP_K,
        score_type="dense_score",
        latency_ms=dense_latency,
    )

    rrf_documents, latency = _run_traced_stage(
        name="rrf_child",
        function=pipeline.rrf,
        args=(bm25_documents, dense_documents),
        inputs={
            "query": query,
            "metadata_filter": effective_context["metadata_filter"],
            "documents": {
                "bm25_child": serialize_documents_for_trace(
                    bm25_documents
                ),
                "dense_child": serialize_documents_for_trace(
                    dense_documents
                ),
            },
        },
        configured_k=rag_config.RRF_TOP_K,
        score_type="rrf_score",
    )
    recorder.record(
        name="rrf_child",
        documents=rrf_documents,
        configured_k=rag_config.RRF_TOP_K,
        score_type="rrf_score",
        latency_ms=latency,
    )

    reranked_documents, latency = _run_traced_stage(
        name="cross_encoder_child",
        function=pipeline.cross_encoder,
        args=(query, rrf_documents),
        inputs={
            "query": query,
            "metadata_filter": effective_context["metadata_filter"],
            "documents": serialize_documents_for_trace(rrf_documents),
        },
        configured_k=rag_config.RERANK_TOP_K,
        score_type="rerank_score",
    )
    recorder.record(
        name="cross_encoder_child",
        documents=reranked_documents,
        configured_k=rag_config.RERANK_TOP_K,
        score_type="rerank_score",
        latency_ms=latency,
    )

    business_documents, latency = _run_traced_stage(
        name="business_fused_child",
        function=pipeline.business_fusion,
        args=(query, reranked_documents, effective_context),
        inputs={
            "query": query,
            "metadata_filter": effective_context["metadata_filter"],
            "documents": serialize_documents_for_trace(
                reranked_documents
            ),
        },
        configured_k=rag_config.BUSINESS_FUSION_TOP_K,
        score_type="business_score",
    )
    recorder.record(
        name="business_fused_child",
        documents=business_documents,
        configured_k=rag_config.BUSINESS_FUSION_TOP_K,
        score_type="business_score",
        latency_ms=latency,
    )

    parent_documents, latency = _run_traced_stage(
        name="diversified_parent",
        function=pipeline.diversify_parents,
        args=(business_documents,),
        inputs={
            "query": query,
            "metadata_filter": effective_context["metadata_filter"],
            "documents": serialize_documents_for_trace(
                business_documents
            ),
        },
        configured_k=rag_config.DIVERSIFIED_PARENT_TOP_K,
        score_type="mmr_score",
    )
    recorder.record(
        name="diversified_parent",
        documents=parent_documents,
        configured_k=rag_config.DIVERSIFIED_PARENT_TOP_K,
        score_type="mmr_score",
        latency_ms=latency,
    )

    final_documents, latency = _run_traced_stage(
        name="final_context_parent",
        function=pipeline.assemble_context,
        args=(parent_documents,),
        inputs={
            "query": query,
            "metadata_filter": effective_context["metadata_filter"],
            "documents": serialize_documents_for_trace(parent_documents),
        },
        configured_k=rag_config.FINAL_PARENT_TOP_K,
        score_type="context_score",
    )
    recorder.record(
        name="final_context_parent",
        documents=final_documents,
        configured_k=rag_config.FINAL_PARENT_TOP_K,
        score_type="context_score",
        latency_ms=latency,
    )
    recorder.trace.require_complete()
    return RetrievalExecution(
        final_documents=tuple(final_documents),
        trace=recorder.trace,
    )


def _default_pipeline_components() -> RetrievalPipelineComponents:
    store = get_vectorstore()
    embeddings = get_embeddings()
    parent_docstore = get_parent_docstore()

    def bm25(query: str, context: Mapping[str, Any]) -> list[Document]:
        return ElasticsearchBM25Retriever(
            store=store,
            top_k=rag_config.BM25_TOP_K,
            metadata_filter=context.get("metadata_filter"),
        ).invoke(query)

    def dense(query: str, context: Mapping[str, Any]) -> list[Document]:
        return ElasticsearchDenseRetriever(
            store=store,
            embeddings=embeddings,
            top_k=rag_config.SEMANTIC_TOP_K,
            metadata_filter=context.get("metadata_filter"),
        ).invoke(query)

    return RetrievalPipelineComponents(
        bm25=bm25,
        dense=dense,
        rrf=lambda bm25_documents, dense_documents: weighted_rrf(
            bm25_documents=bm25_documents,
            dense_documents=dense_documents,
            bm25_weight=1 - rag_config.SEMANTIC_WEIGHT,
            dense_weight=rag_config.SEMANTIC_WEIGHT,
            rrf_k=rag_config.RRF_K,
            top_k=rag_config.RRF_TOP_K,
            max_children_per_parent=(
                rag_config.MAX_CHILDREN_PER_PARENT_PRE_RRF
            ),
        ),
        cross_encoder=lambda query, documents: rerank_documents(
            query,
            documents,
            enabled=True,
            top_n=rag_config.RERANK_TOP_K,
        ),
        business_fusion=lambda query, documents, context: fuse_business_features(
            query,
            documents,
            time_intent=context.get("time_intent"),
        ),
        diversify_parents=lambda documents: diversify_parent_candidates(
            documents,
            parent_docstore=parent_docstore,
            top_k=rag_config.DIVERSIFIED_PARENT_TOP_K,
            max_parents_per_document=rag_config.MAX_PARENTS_PER_DOCUMENT,
            similarity_threshold=rag_config.SIMILARITY_DEDUP_THRESHOLD,
            mmr_lambda=rag_config.MMR_LAMBDA,
        ),
        assemble_context=lambda documents: assemble_final_context(
            documents,
            max_documents=rag_config.FINAL_PARENT_TOP_K,
            token_budget=rag_config.FINAL_CONTEXT_TOKEN_BUDGET,
            tokenizer_name=rag_config.TOKENIZER_NAME,
        ),
    )


def _parallel_recall(
    query: str,
    context: Mapping[str, Any],
    pipeline: RetrievalPipelineComponents,
) -> tuple[
    list[Document],
    list[Document],
    float,
    float,
    dict[str, str],
]:
    audit_inputs = {
        "query": query,
        "metadata_filter": context.get("metadata_filter"),
        "documents": [],
    }
    bm25_call = partial(
        _run_traced_stage,
        name="bm25_child",
        function=pipeline.bm25,
        args=(query, context),
        inputs=audit_inputs,
        configured_k=rag_config.BM25_TOP_K,
        score_type="bm25_score",
    )
    dense_call = partial(
        _run_traced_stage,
        name="dense_child",
        function=pipeline.dense,
        args=(query, context),
        inputs=audit_inputs,
        configured_k=rag_config.SEMANTIC_TOP_K,
        score_type="dense_score",
    )
    bm25_context = copy_context()
    dense_context = copy_context()
    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="rag-stage",
    ) as executor:
        futures = {
            "bm25_child": executor.submit(
                bm25_context.run,
                bm25_call,
            ),
            "dense_child": executor.submit(
                dense_context.run,
                dense_call,
            ),
        }
        results: dict[str, tuple[list[Document], float]] = {}
        errors: dict[str, str] = {}
        for name, future in futures.items():
            try:
                documents, latency = future.result()
                results[name] = (list(documents), latency)
            except Exception as exc:
                errors[name] = str(exc)
                results[name] = ([], 0.0)
    bm25_documents, bm25_latency = results["bm25_child"]
    dense_documents, dense_latency = results["dense_child"]
    return (
        bm25_documents,
        dense_documents,
        bm25_latency,
        dense_latency,
        errors,
    )


def _run_traced_stage(
    *,
    name: str,
    function: Callable[..., Any],
    args: tuple[Any, ...],
    inputs: Mapping[str, Any],
    configured_k: int,
    score_type: str,
) -> tuple[Any, float]:
    with trace_span(name, inputs=inputs) as span:
        started = perf_counter()
        try:
            result = function(*args)
        except Exception as exc:
            latency = (perf_counter() - started) * 1000
            end_trace_span(
                span,
                outputs={
                    "documents": [],
                    "configured_k": configured_k,
                    "score_type": score_type,
                    "latency_ms": latency,
                    "error": str(exc),
                },
            )
            raise
        latency = (perf_counter() - started) * 1000
        end_trace_span(
            span,
            outputs={
                "documents": serialize_documents_for_trace(result),
                "configured_k": configured_k,
                "score_type": score_type,
                "latency_ms": latency,
            },
        )
    return result, latency


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
        return ProductionHybridRetriever(time_intent=time_intent)
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
    """Compatibility API that uses the complete production path by default."""
    if ensemble_retriever is None or (
        time_intent is not None
        and getattr(ensemble_retriever, "time_intent", None) != time_intent
    ):
        ensemble_retriever = get_hybrid_retriever(time_intent=time_intent)
        if ensemble_retriever is None:
            return []
    limit = top_k or rag_config.FINAL_TOP_K
    if isinstance(ensemble_retriever, ProductionHybridRetriever):
        return ensemble_retriever.invoke(query)[:limit]
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
