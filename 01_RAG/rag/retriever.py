"""
rag/retriever.py — parent-child 混合检索器
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import rag_config
from rag.docstore import ParentDocStore, get_parent_docstore
from rag.vectorstore import get_vectorstore


@dataclass
class ParentChildHybridRetriever:
    ensemble_retriever: object
    parent_docstore: ParentDocStore

    def invoke(self, query: str) -> List[Document]:
        child_hits = self.ensemble_retriever.invoke(query)
        return hydrate_parent_results(
            child_hits,
            parent_docstore=self.parent_docstore,
            limit=rag_config.MAX_HYDRATED_PARENTS,
        )


def build_hybrid_retriever(
    all_chunks: List[Document],
    parent_docstore: ParentDocStore | None = None,
):
    """
    构建 child 级混合检索器，并在最终阶段回填 parent 文档。
    """
    vs = get_vectorstore()
    docstore = parent_docstore or get_parent_docstore()

    semantic_retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": rag_config.SEMANTIC_TOP_K},
    )

    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = rag_config.BM25_TOP_K

    # ensemble对象
    # retrievers=[VectorStoreRetriever(tags=['Chroma', 'DashScopeEmbeddings'], vectorstore=<langchain_chroma.vectorstores.Chroma object at 0x107956eb0>, search_kwargs={'k': 6}), BM25Retriever(vectorizer=<rank_bm25.BM25Okapi object at 0x115b215e0>, k=6)] weights=[0.6, 0.4]
    ensemble = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[rag_config.SEMANTIC_WEIGHT, 1 - rag_config.SEMANTIC_WEIGHT],
    )
    return ParentChildHybridRetriever(ensemble, docstore)


def get_hybrid_retriever():
    """获取混合检索器（需要先加载 children 到向量库）"""
    vs = get_vectorstore()
    try:
        result = vs.get(include=["documents", "metadatas"])
        if not result.get("documents"):
            return None

        all_chunks = [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(result["documents"], result.get("metadatas", []))
        ]
        return build_hybrid_retriever(all_chunks)
    except Exception:
        return None


def hydrate_parent_results(
    child_hits: List[Document],
    parent_docstore: ParentDocStore,
    limit: int | None = None,
) -> List[Document]:
    """
    将 child 级命中聚合为 parent 级结果。
    """
    if not child_hits:
        return []

    scores: dict[str, float] = {}
    matched_child_ids: dict[str, list[str]] = {}
    best_meta: dict[str, dict] = {}

    for rank, child in enumerate(child_hits):
        parent_id = child.metadata.get("parent_id")
        child_id = child.metadata.get("child_id")
        if not parent_id:
            continue
        
        # 计算每条 child 的分数
        score = child.metadata.get("similarity_score") or 1 / (60 + rank + 1)
        # 累加到 parent 总分
        # 多个 child 命中同一个 parent 时，parent 总分会累加
        scores[parent_id] = scores.get(parent_id, 0.0) + score
        # 收集命中的 child_id
        matched_child_ids.setdefault(parent_id, [])
        if child_id and child_id not in matched_child_ids[parent_id]:
            matched_child_ids[parent_id].append(child_id)
        # 记录“最优 child”的 metadata
        # 每个 parent 下，单条 child 中分数最高的那条
        current_best = best_meta.get(parent_id)
        if current_best is None or score > current_best["score"]:
            best_meta[parent_id] = {
                "score": score,
                "metadata": child.metadata,
            }
    # 对 parent 排序
    """ 
    - sorted(scores, ...)
        遍历的是 scores 这个字典的 key，也就是所有 parent_id
    - key=lambda parent_id: (...)
        排序依据是一个匿名函数返回的元组

    排序规则是：

    1. 先按 -scores[parent_id]
        总分高的排前面
    2. 再按 -len(matched_child_ids[...])
        命中的 child 数更多的排前面

    负号 - 是为了实现降序。
    """
    ranked_parent_ids = sorted(
        scores,
        key=lambda parent_id: (-scores[parent_id], -len(matched_child_ids.get(parent_id, []))),
    )
    # 截断返回数量
    if limit is not None:
        ranked_parent_ids = ranked_parent_ids[:limit]
    
    # 批量回填 parent 原文
    parents = parent_docstore.get_parents(ranked_parent_ids)
    # 组装最终返回结果
    results: list[Document] = []
    for parent_id in ranked_parent_ids:
        parent_doc = parents.get(parent_id)
        if parent_doc is None:
            continue

        best = best_meta[parent_id]
        metadata = {
            # 这是 字典解包。
            # 意思是先把 parent_doc.metadata 全部展开复制进新字典，再补充/覆盖后面的字段
            **parent_doc.metadata,
            "matched_child_ids": matched_child_ids.get(parent_id, []),
            "best_child_score": round(best["score"], 4),
            "section_path": parent_doc.metadata.get(
                "section_path",
                best["metadata"].get("section_path", "未命名章节"),
            ),
            "page_range": parent_doc.metadata.get(
                "page_range",
                best["metadata"].get("page_range", "?"),
            ),
        }
        results.append(Document(page_content=parent_doc.page_content, metadata=metadata))

    return results


def retrieve_with_hybrid(
    query: str,
    top_k: int | None = None,
    ensemble_retriever=None,
) -> List[Document]:
    """使用混合检索器检索相关 parent 文档。"""
    if ensemble_retriever is None:
        ensemble_retriever = get_hybrid_retriever()
        if ensemble_retriever is None:
            return []

    top_k = top_k or rag_config.FINAL_TOP_K
    results = ensemble_retriever.invoke(query)
    return results[:top_k]
