"""
rag/vectorstore.py — child chunk 向量库管理
支持：文档增量添加、删除、查询、持久化，并与 parent docstore 协同工作。
"""
from __future__ import annotations

from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import chroma_config, rag_config
from rag.chunker import ChunkingResult
from rag.docstore import ParentDocStore, get_parent_docstore
from rag.embedder import get_embeddings


_vectorstore_instance: Optional[Chroma] = None


def get_vectorstore(reset: bool = False) -> Chroma:
    """获取（或初始化）ChromaDB 实例（单例）"""
    global _vectorstore_instance

    if _vectorstore_instance is not None and not reset:
        return _vectorstore_instance

    embeddings = get_embeddings()
    _vectorstore_instance = Chroma(
        collection_name=chroma_config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=chroma_config.PERSIST_DIRECTORY,
        collection_metadata={"hnsw:space": "cosine"},
    )
    return _vectorstore_instance


def add_documents(
    chunks: ChunkingResult | List[Document],
    doc_id: str,
    vectorstore: Optional[Chroma] = None,
    parent_docstore: Optional[ParentDocStore] = None,
) -> int:
    """
    增量添加 child chunks 到向量库，并将 parent chunks 写入 docstore。
    失败时执行文档级回滚，避免 child / parent 半成功。
    """
    vs = vectorstore or get_vectorstore()
    docstore = parent_docstore or get_parent_docstore()

    delete_document(doc_id, vs, docstore)

    if isinstance(chunks, ChunkingResult):
        parents = chunks.parents
        children = chunks.children
    else:
        parents = []
        children = chunks

    if not children:
        return 0

    try:
        if parents:
            docstore.upsert_parents(parents)

        ids = [
            child.metadata.get("child_id")
            or child.metadata.get("parent_id")
            or f"{doc_id}_{child.metadata.get('chunk_index', i)}"
            for i, child in enumerate(children)
        ]

        batch_size = 500
        added = 0
        for i in range(0, len(children), batch_size):
            batch_docs = children[i: i + batch_size]
            batch_ids = ids[i: i + batch_size]
            vs.add_documents(documents=batch_docs, ids=batch_ids)
            added += len(batch_docs)
        return added
    except Exception:
        delete_document(doc_id, vs, docstore)
        raise


def delete_document(
    doc_id: str,
    vectorstore: Optional[Chroma] = None,
    parent_docstore: Optional[ParentDocStore] = None,
) -> int:
    """删除指定 doc_id 的 child 向量和 parent 文档。"""
    vs = vectorstore or get_vectorstore()
    docstore = parent_docstore or get_parent_docstore()

    deleted_children = 0
    try:
        existing = vs.get(where={"doc_id": doc_id})
        if existing and existing.get("ids"):
            vs.delete(ids=existing["ids"])
            deleted_children = len(existing["ids"])
    except Exception:
        pass

    try:
        docstore.delete_document(doc_id)
    except Exception:
        pass

    return deleted_children


def list_documents(
    vectorstore: Optional[Chroma] = None,
    parent_docstore: Optional[ParentDocStore] = None,
) -> List[dict]:
    """
    列出索引中的所有文档，按 doc_id 聚合 parent / child 数量。

    Returns:
        [{"doc_id": ..., "source": ..., "parent_count": ..., "child_count": ...}]
    """
    vs = vectorstore or get_vectorstore()
    docstore = parent_docstore or get_parent_docstore()

    docs: dict[str, dict] = {}

    try:
        result = vs.get(include=["metadatas"])
        metadatas = result.get("metadatas", [])
    except Exception:
        metadatas = []

    for meta in metadatas:
        if not meta:
            continue
        doc_id = meta.get("doc_id", "unknown")
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "source": meta.get("source", "未知"),
                "total_pages": meta.get("total_pages", 0),
                "total_chunks": 0,
                "child_count": 0,
                "parent_count": 0,
                "doc_version": meta.get("doc_version", ""),
                "pages": set(),
            }
        docs[doc_id]["total_chunks"] += 1
        docs[doc_id]["child_count"] += 1
        page = meta.get("page")
        if page:
            docs[doc_id]["pages"].add(page)

    try:
        parent_docs = docstore.list_documents()
    except Exception:
        parent_docs = []

    for parent_doc in parent_docs:
        doc_id = parent_doc["doc_id"]
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "source": parent_doc.get("source", "未知"),
                "total_pages": 0,
                "total_chunks": 0,
                "child_count": 0,
                "parent_count": 0,
                "doc_version": parent_doc.get("doc_version", ""),
                "pages": set(),
            }
        docs[doc_id]["parent_count"] = parent_doc.get("parent_count", 0)
        docs[doc_id]["doc_version"] = parent_doc.get("doc_version", docs[doc_id]["doc_version"])
        if not docs[doc_id]["source"] or docs[doc_id]["source"] == "未知":
            docs[doc_id]["source"] = parent_doc.get("source", "未知")

    return [{**doc, "pages": sorted(doc["pages"])} for doc in docs.values()]


def get_collection_stats(
    vectorstore: Optional[Chroma] = None,
    parent_docstore: Optional[ParentDocStore] = None,
) -> dict:
    """返回 child collection 与 parent docstore 统计信息。"""
    vs = vectorstore or get_vectorstore()
    docstore = parent_docstore or get_parent_docstore()

    try:
        child_count = vs._collection.count()
    except Exception:
        child_count = 0

    try:
        parent_count = docstore.count()
    except Exception:
        parent_count = 0

    return {
        "total_chunks": child_count,
        "total_children": child_count,
        "total_parents": parent_count,
        "collection_name": chroma_config.COLLECTION_NAME,
        "persist_dir": chroma_config.PERSIST_DIRECTORY,
    }


def similarity_search_with_threshold(
    query: str,
    k: int = 4,
    threshold: float | None = None,
    vectorstore: Optional[Chroma] = None,
    filter_doc_ids: Optional[List[str]] = None,
) -> List[Document]:
    """
    带相似度阈值过滤的 child 级语义检索。
    """
    vs = vectorstore or get_vectorstore()
    threshold = threshold if threshold is not None else rag_config.SIMILARITY_THRESHOLD

    results_with_scores = vs.similarity_search_with_relevance_scores(
        query=query,
        k=k,
    )

    filtered = []
    for doc, score in results_with_scores:
        if filter_doc_ids and doc.metadata.get("doc_id") not in filter_doc_ids:
            continue
        if score >= threshold:
            doc.metadata["similarity_score"] = round(score, 4)
            filtered.append(doc)

    return filtered
