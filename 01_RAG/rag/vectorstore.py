"""Elasticsearch Child Store facade with SQLite Parent Store coordination."""

from __future__ import annotations

from typing import Any, List, Optional
from uuid import uuid4

from langchain_core.documents import Document

from config import elasticsearch_config, rag_config
from rag.chunker import ChunkingResult
from rag.docstore import ParentDocStore, get_parent_docstore
from rag.elasticsearch_store import ElasticsearchChildStore
from rag.embedder import embed_with_retry, get_embeddings


_vectorstore_instance: Optional[ElasticsearchChildStore] = None


def get_vectorstore(reset: bool = False) -> ElasticsearchChildStore:
    """Return the process-local Elasticsearch Child Store facade."""
    global _vectorstore_instance
    if _vectorstore_instance is not None and not reset:
        return _vectorstore_instance
    _vectorstore_instance = ElasticsearchChildStore()
    return _vectorstore_instance


def add_documents(
    chunks: ChunkingResult | List[Document],
    doc_id: str,
    vectorstore: Optional[Any] = None,
    parent_docstore: Optional[ParentDocStore] = None,
    *,
    ingest_run_id: str | None = None,
) -> int:
    """
    Stage a new Child version, activate it, then retire old Child/Parent versions.

    Before activation, failures remove only the new staging run and its new
    Parent version. Existing active versions remain queryable.
    """
    store = vectorstore or get_vectorstore()
    docstore = parent_docstore or get_parent_docstore()
    if isinstance(chunks, ChunkingResult):
        parents = list(chunks.parents)
        children = list(chunks.children)
    else:
        parents = []
        children = list(chunks)
    if not children:
        return 0

    doc_version = _validate_document_batch(
        doc_id=doc_id,
        parents=parents,
        children=children,
    )
    current_ingest_run = ingest_run_id or uuid4().hex
    embeddings = get_embeddings()
    vectors = embed_with_retry(
        embeddings,
        [child.page_content for child in children],
    )
    activated = False

    try:
        if parents:
            docstore.upsert_parents(parents)
        added = store.bulk_stage_children(
            children,
            vectors,
            current_ingest_run,
        )
        activated_count = store.activate_version(
            doc_id,
            doc_version,
            current_ingest_run,
        )
        if isinstance(activated_count, int) and activated_count != added:
            raise RuntimeError(
                "新版本激活数量与 staging 数量不一致: "
                f"staged={added}, activated={activated_count}"
            )
        activated = True
        store.deactivate_other_versions(doc_id, doc_version)
        docstore.delete_versions_except(doc_id, doc_version)
        return added
    except Exception:
        if activated and hasattr(store, "delete_document_version"):
            store.delete_document_version(doc_id, doc_version)
            if hasattr(store, "reactivate_other_versions"):
                store.reactivate_other_versions(doc_id, doc_version)
        else:
            store.delete_ingest_run(current_ingest_run)
        docstore.delete_document_version(doc_id, doc_version)
        raise


def delete_document(
    doc_id: str,
    vectorstore: Optional[Any] = None,
    parent_docstore: Optional[ParentDocStore] = None,
) -> int:
    """Delete searchable ES Children first, then their SQLite Parents."""
    store = vectorstore or get_vectorstore()
    docstore = parent_docstore or get_parent_docstore()
    deleted_children = int(store.delete_document(doc_id))
    docstore.delete_document(doc_id)
    return deleted_children


def list_documents(
    vectorstore: Optional[Any] = None,
    parent_docstore: Optional[ParentDocStore] = None,
) -> List[dict]:
    """List active documents with Child and Parent counts."""
    store = vectorstore or get_vectorstore()
    docstore = parent_docstore or get_parent_docstore()
    if hasattr(store, "aggregate_documents"):
        child_documents = list(store.aggregate_documents())
    else:
        child_documents = _legacy_document_aggregation(store)

    documents = {
        str(item["doc_id"]): {
            "doc_id": str(item["doc_id"]),
            "source": item.get("source", "未知"),
            "total_pages": int(item.get("total_pages", 0)),
            "total_chunks": int(
                item.get("total_chunks", item.get("child_count", 0))
            ),
            "child_count": int(item.get("child_count", 0)),
            "parent_count": int(item.get("parent_count", 0)),
            "doc_version": item.get("doc_version", ""),
            "pages": list(item.get("pages", [])),
        }
        for item in child_documents
    }
    for parent in docstore.list_documents():
        doc_key = str(parent["doc_id"])
        item = documents.setdefault(
            doc_key,
            {
                "doc_id": doc_key,
                "source": parent.get("source", "未知"),
                "total_pages": 0,
                "total_chunks": 0,
                "child_count": 0,
                "parent_count": 0,
                "doc_version": parent.get("doc_version", ""),
                "pages": [],
            },
        )
        item["parent_count"] = int(parent.get("parent_count", 0))
        item["doc_version"] = parent.get(
            "doc_version",
            item["doc_version"],
        )
        if item["source"] in {"", "未知"}:
            item["source"] = parent.get("source", "未知")
    return list(documents.values())


def get_collection_stats(
    vectorstore: Optional[Any] = None,
    parent_docstore: Optional[ParentDocStore] = None,
) -> dict:
    """Return observable ES Child and SQLite Parent counts."""
    try:
        store = vectorstore or get_vectorstore()
    except Exception as exc:
        return _empty_stats(error=str(exc))
    try:
        if hasattr(store, "count_children"):
            child_count = int(store.count_children())
        elif hasattr(store, "_collection"):
            child_count = int(store._collection.count())
        else:
            child_count = 0
    except Exception as exc:
        return _empty_stats(error=str(exc))

    try:
        docstore = parent_docstore or get_parent_docstore()
        parent_count = int(docstore.count())
    except Exception as exc:
        return {
            **_stats(child_count=child_count, parent_count=0),
            "error": str(exc),
        }
    return _stats(child_count=child_count, parent_count=parent_count)


def build_time_filter(time_intent: Optional[dict]) -> Optional[dict]:
    """Translate query time intent to the shared BM25/Dense metadata filter."""
    if not time_intent:
        return None
    intent_type = time_intent.get("type")
    if intent_type not in {"year", "before", "after", "range"}:
        return None
    date_range = time_intent.get("range") or {}
    gte = date_range.get("gte")
    lte = date_range.get("lte")
    if gte is None or lte is None:
        return None
    if time_intent.get("field", "doc_date") == "upload_date":
        return {"upload_date": {"$gte": gte, "$lte": lte}}
    return {
        "$and": [
            {"has_doc_date": True},
            {"doc_date_min": {"$lte": lte}},
            {"doc_date_max": {"$gte": gte}},
        ]
    }


def similarity_search_with_threshold(
    query: str,
    k: int = 4,
    threshold: float | None = None,
    vectorstore: Optional[Any] = None,
    filter_doc_ids: Optional[List[str]] = None,
    metadata_filter: Optional[dict] = None,
) -> List[Document]:
    """Compatibility wrapper around Elasticsearch Dense retrieval."""
    store = vectorstore or get_vectorstore()
    effective_threshold = (
        rag_config.SIMILARITY_THRESHOLD
        if threshold is None
        else threshold
    )
    if hasattr(store, "dense_search"):
        documents = store.dense_search(
            query,
            k=k,
            metadata_filter=metadata_filter,
        )
    elif hasattr(store, "similarity_search_with_relevance_scores"):
        results = store.similarity_search_with_relevance_scores(
            query=query,
            k=k,
        )
        documents = []
        for document, score in results:
            if score >= effective_threshold:
                document.metadata["similarity_score"] = round(float(score), 4)
                documents.append(document)
    else:
        from rag.elasticsearch_retrievers import ElasticsearchDenseRetriever

        documents = ElasticsearchDenseRetriever(
            store=store,
            embeddings=get_embeddings(),
            top_k=k,
            metadata_filter=metadata_filter,
        ).invoke(query)

    return [
        document
        for document in documents
        if (
            not filter_doc_ids
            or document.metadata.get("doc_id") in filter_doc_ids
        )
        and float(
            document.metadata.get(
                "similarity_score",
                document.metadata.get("es_score", 1.0),
            )
            or 0.0
        )
        >= effective_threshold
    ]


def _validate_document_batch(
    *,
    doc_id: str,
    parents: list[Document],
    children: list[Document],
) -> str:
    versions = {
        str(document.metadata.get("doc_version") or "").strip()
        for document in [*parents, *children]
    }
    document_ids = {
        str(document.metadata.get("doc_id") or "").strip()
        for document in [*parents, *children]
    }
    if versions == {""} or len(versions) != 1:
        raise ValueError("一次摄取必须包含唯一且非空的 doc_version")
    if document_ids != {str(doc_id)}:
        raise ValueError("摄取批次 doc_id 与调用参数不一致")
    for child in children:
        if not str(child.metadata.get("child_id") or "").strip():
            raise ValueError("Child 缺少 child_id")
        if not str(child.metadata.get("parent_id") or "").strip():
            raise ValueError("Child 缺少 parent_id")
    return next(iter(versions))


def _legacy_document_aggregation(vectorstore: Any) -> list[dict]:
    result = vectorstore.get(include=["metadatas"])
    documents: dict[str, dict] = {}
    for metadata in result.get("metadatas", []):
        if not metadata:
            continue
        doc_id = str(metadata.get("doc_id", "unknown"))
        item = documents.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "source": metadata.get("source", "未知"),
                "total_pages": metadata.get("total_pages", 0),
                "total_chunks": 0,
                "child_count": 0,
                "parent_count": 0,
                "doc_version": metadata.get("doc_version", ""),
                "pages": [],
            },
        )
        item["total_chunks"] += 1
        item["child_count"] += 1
        page = metadata.get("page")
        if page and page not in item["pages"]:
            item["pages"].append(page)
    for item in documents.values():
        item["pages"].sort()
    return list(documents.values())


def _stats(*, child_count: int, parent_count: int) -> dict:
    return {
        "total_chunks": child_count,
        "total_children": child_count,
        "total_parents": parent_count,
        "collection_name": elasticsearch_config.READ_ALIAS,
        "persist_dir": elasticsearch_config.URL,
        "backend": "elasticsearch",
    }


def _empty_stats(*, error: str) -> dict:
    return {
        **_stats(child_count=0, parent_count=0),
        "error": error,
    }
