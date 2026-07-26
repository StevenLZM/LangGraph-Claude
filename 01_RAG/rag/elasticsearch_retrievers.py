from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Sequence

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag.elasticsearch_store import build_es_filters, hit_to_document


class ElasticsearchBM25Retriever(BaseRetriever):
    store: Any
    top_k: int = 50
    metadata_filter: dict[str, Any] | None = None

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> list[Document]:
        filters = _active_filters(self.metadata_filter)
        response = self.store.client.search(
            index=self.store.config.READ_ALIAS,
            size=self.top_k,
            query={
                "bool": {
                    "must": [{"match": {"content": {"query": query}}}],
                    "filter": filters,
                }
            },
        )
        documents = [
            hit_to_document(hit, retrieval_source="bm25")
            for hit in response.get("hits", {}).get("hits", [])
        ]
        return [
            _copy_with_score(document, "bm25_score")
            for document in documents
        ]


class ElasticsearchDenseRetriever(BaseRetriever):
    store: Any
    embeddings: Any
    top_k: int = 50
    metadata_filter: dict[str, Any] | None = None

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> list[Document]:
        vector = self.embeddings.embed_query(query)
        filters = _active_filters(self.metadata_filter)
        response = self.store.client.search(
            index=self.store.config.READ_ALIAS,
            knn={
                "field": "embedding",
                "query_vector": vector,
                "k": self.top_k,
                "num_candidates": max(
                    self.top_k,
                    int(self.store.config.DENSE_NUM_CANDIDATES),
                ),
                "filter": filters,
            },
            size=self.top_k,
        )
        documents = [
            hit_to_document(hit, retrieval_source="dense")
            for hit in response.get("hits", {}).get("hits", [])
        ]
        return [
            _copy_with_score(document, "dense_score")
            for document in documents
        ]


class HybridChildRetriever:
    def __init__(
        self,
        *,
        bm25: Any,
        dense: Any,
        bm25_weight: float,
        dense_weight: float,
        rrf_k: int,
        top_k: int,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.bm25_weight = float(bm25_weight)
        self.dense_weight = float(dense_weight)
        self.rrf_k = int(rrf_k)
        self.top_k = int(top_k)
        self.last_errors: dict[str, str] = {}
        self.last_bm25_documents: list[Document] = []
        self.last_dense_documents: list[Document] = []

    def invoke(self, query: str, config: Any = None) -> list[Document]:
        bm25_documents, dense_documents = self.retrieve_branches(query)
        return weighted_rrf(
            bm25_documents=bm25_documents,
            dense_documents=dense_documents,
            bm25_weight=self.bm25_weight,
            dense_weight=self.dense_weight,
            rrf_k=self.rrf_k,
            top_k=self.top_k,
        )

    def retrieve_branches(
        self,
        query: str,
    ) -> tuple[list[Document], list[Document]]:
        self.last_errors = {}
        with ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="rag-retrieval",
        ) as executor:
            futures = {
                "bm25": executor.submit(self.bm25.invoke, query),
                "dense": executor.submit(self.dense.invoke, query),
            }
            results: dict[str, list[Document]] = {}
            for name, future in futures.items():
                try:
                    results[name] = list(future.result())
                except Exception as exc:
                    self.last_errors[name] = str(exc)
                    results[name] = []
        self.last_bm25_documents = results["bm25"]
        self.last_dense_documents = results["dense"]
        return self.last_bm25_documents, self.last_dense_documents


def weighted_rrf(
    *,
    bm25_documents: Sequence[Document],
    dense_documents: Sequence[Document],
    bm25_weight: float,
    dense_weight: float,
    rrf_k: int,
    top_k: int,
) -> list[Document]:
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    route_ranks: dict[str, dict[str, int]] = {}
    route_names: dict[str, list[str]] = {}

    for route, route_documents, weight in (
        ("bm25", bm25_documents, bm25_weight),
        ("dense", dense_documents, dense_weight),
    ):
        seen_in_route: set[str] = set()
        for rank, document in enumerate(route_documents, start=1):
            child_id = str(document.metadata.get("child_id") or "").strip()
            if not child_id:
                raise ValueError(f"{route} 候选缺少 child_id，无法执行 RRF")
            if child_id in seen_in_route:
                continue
            seen_in_route.add(child_id)
            scores[child_id] = scores.get(child_id, 0.0) + (
                float(weight) / (rrf_k + rank)
            )
            documents.setdefault(child_id, document)
            route_ranks.setdefault(child_id, {})[route] = rank
            route_names.setdefault(child_id, []).append(route)

    ranked_ids = sorted(
        scores,
        key=lambda child_id: (
            -scores[child_id],
            min(route_ranks[child_id].values()),
            child_id,
        ),
    )[: max(0, top_k)]
    return [
        Document(
            page_content=documents[child_id].page_content,
            metadata={
                **documents[child_id].metadata,
                "rrf_score": scores[child_id],
                "rrf_ranks": route_ranks[child_id],
                "retrieval_sources": route_names[child_id],
            },
        )
        for child_id in ranked_ids
    ]


def _active_filters(
    metadata_filter: Mapping[str, Any] | None,
) -> list[dict]:
    return [
        {"term": {"status": "active"}},
        *build_es_filters(metadata_filter),
    ]


def _copy_with_score(document: Document, score_key: str) -> Document:
    score = document.metadata.get("es_score")
    return Document(
        page_content=document.page_content,
        metadata={
            **document.metadata,
            score_key: float(score) if score is not None else None,
        },
    )
