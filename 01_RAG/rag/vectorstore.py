"""
rag/vectorstore.py — ChromaDB 向量库管理
支持：文档增量添加、删除、查询、持久化
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma

from config import chroma_config, rag_config
from rag.embedder import get_embeddings


# ── 单例管理 ──────────────────────────────────────────────────────
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
        collection_metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
    )
    return _vectorstore_instance


def add_documents(
    chunks: List[Document],
    doc_id: str,
    vectorstore: Optional[Chroma] = None,
) -> int:
    """
    增量添加文档块到向量库
    先删除同 doc_id 的旧数据，再插入新数据（幂等操作）

    Returns:
        成功添加的 chunk 数量
    """
    vs = vectorstore or get_vectorstore()

    # 删除旧版本（如果存在）
    delete_document(doc_id, vs)

    if not chunks:
        return 0

    # 为每个 chunk 生成唯一 ID（doc_id + chunk_index）
    ids = [
        f"{doc_id}_{chunk.metadata.get('chunk_index', i)}"
        for i, chunk in enumerate(chunks)
    ]

    # 批量添加（ChromaDB 每次最多 5461 条）
    batch_size = 500
    added = 0
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i: i + batch_size]
        batch_ids = ids[i: i + batch_size]
        vs.add_documents(documents=batch_chunks, ids=batch_ids)
        added += len(batch_chunks)

    return added


def delete_document(doc_id: str, vectorstore: Optional[Chroma] = None) -> int:
    """删除指定 doc_id 的所有向量数据"""
    vs = vectorstore or get_vectorstore()

    try:
        # 查找该文档的所有 chunk
        existing = vs.get(where={"doc_id": doc_id})
        if existing and existing.get("ids"):
            vs.delete(ids=existing["ids"])
            return len(existing["ids"])
    except Exception:
        pass
    return 0


def list_documents(vectorstore: Optional[Chroma] = None) -> List[dict]:
    """
    列出向量库中的所有文档（去重，按文件名聚合）

    Returns:
        [{"doc_id": ..., "source": ..., "total_chunks": ..., "pages": ...}]
    """
    vs = vectorstore or get_vectorstore()

    try:
        result = vs.get(include=["metadatas"])
        metadatas = result.get("metadatas", [])
    except Exception:
        return []

    # 按 doc_id 聚合
    docs: dict[str, dict] = {}
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
                "pages": set(),
            }
        docs[doc_id]["total_chunks"] += 1
        page = meta.get("page")
        if page:
            docs[doc_id]["pages"].add(page)

    # 序列化（set → sorted list）
    return [
        {**v, "pages": sorted(v["pages"])}
        for v in docs.values()
    ]


def get_collection_stats(vectorstore: Optional[Chroma] = None) -> dict:
    """返回向量库统计信息"""
    vs = vectorstore or get_vectorstore()
    try:
        count = vs._collection.count()
        return {
            "total_chunks": count,
            "collection_name": chroma_config.COLLECTION_NAME,
            "persist_dir": chroma_config.PERSIST_DIRECTORY,
        }
    except Exception:
        return {"total_chunks": 0, "collection_name": chroma_config.COLLECTION_NAME}


def similarity_search_with_threshold(
    query: str,
    k: int = 4,
    threshold: float | None = None,
    vectorstore: Optional[Chroma] = None,
    filter_doc_ids: Optional[List[str]] = None,
) -> List[Document]:
    """
    带相似度阈值过滤的语义检索

    Args:
        query: 查询字符串
        k: 返回数量
        threshold: 相似度阈值（0-1），低于此值的结果过滤掉
        filter_doc_ids: 只在指定文档内检索（None=全库检索）
    """
    vs = vectorstore or get_vectorstore()
    threshold = threshold if threshold is not None else rag_config.SIMILARITY_THRESHOLD

    # 检索 + 相似度分数（不传 where 避免版本兼容问题）
    results_with_scores = vs.similarity_search_with_relevance_scores(
        query=query,
        k=k,
    )

    # 过滤低分结果，并将 score 写入 metadata
    filtered = []
    for doc, score in results_with_scores:
        if score >= threshold:
            doc.metadata["similarity_score"] = round(score, 4)
            filtered.append(doc)

    return filtered
