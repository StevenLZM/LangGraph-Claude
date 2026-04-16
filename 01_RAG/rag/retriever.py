"""
rag/retriever.py — 混合检索器（语义 + BM25 + RRF）
"""
from __future__ import annotations
from typing import List

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from config import rag_config
from rag.vectorstore import get_vectorstore


def build_hybrid_retriever(all_chunks: List[Document]):
    """
    构建混合检索器：
    - 语义检索（Dense）：向量相似度
    - BM25 检索（Sparse）：关键词匹配
    - RRF 融合：Reciprocal Rank Fusion
    """
    vs = get_vectorstore()

    # ── 语义检索器（向量库） ──────────────────────────────────────
    semantic_retriever = vs.as_retriever(
        search_type="similarity",
        # similarity 模式下不能传 score_threshold；
        # 这里先全量召回，让后续融合排序决定最终结果。
        search_kwargs={"k": rag_config.SEMANTIC_TOP_K},
    )

    # ── BM25 检索器（关键词） ────────────────────────────────────
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = rag_config.BM25_TOP_K

    # ── RRF 融合：加权融合两个检索器的排序 ───────────────────────
    # RRF 算法：
    #   score = 1/(k + rank)，其中 k 通常为 60
    #   多个排序器的 RRF score 加权求和
    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[rag_config.SEMANTIC_WEIGHT, 1 - rag_config.SEMANTIC_WEIGHT],
    )

    return ensemble_retriever


def get_hybrid_retriever():
    """获取混合检索器（需要先加载文档到向量库）"""
    vs = get_vectorstore()
    try:
        # 从向量库获取所有 chunks
        result = vs.get(include=["documents", "metadatas", "embeddings"])
        if not result.get("documents"):
            return None

        # 重构 Document 对象
        all_chunks = [
            Document(
                page_content=doc,
                metadata=meta or {}
            )
            for doc, meta in zip(result["documents"], result.get("metadatas", []))
        ]

        return build_hybrid_retriever(all_chunks)
    except Exception:
        return None


def retrieve_with_hybrid(
    query: str,
    top_k: int | None = None,
    ensemble_retriever = None
) -> List[Document]:
    """使用混合检索器检索相关文档"""
    if ensemble_retriever is None:
        ensemble_retriever = get_hybrid_retriever()
        if ensemble_retriever is None:
            return []
    top_k = top_k or rag_config.FINAL_TOP_K
    results = ensemble_retriever.invoke(query)

    return results[:top_k]
