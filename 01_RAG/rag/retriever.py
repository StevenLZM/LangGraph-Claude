"""
rag/retriever.py — parent-child 混合检索器（带时间意图感知）

时间感知策略（按 query_rewriter 输出的 time_intent.type 分派）：
- year/before/after/range  → 方案A：pre-filter（语义路 ChromaDB native filter + BM25 路 post-filter wrapper）
- latest                   → 方案B：召回扩大化 + hydrate 阶段按日期二级排序
- none                     → 原流程，零回归
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import rag_config
from rag.docstore import ParentDocStore, get_parent_docstore
from rag.vectorstore import build_time_filter, get_vectorstore


# ────────────────────────────────────────────────────────────────
# 时间意图工具
# ────────────────────────────────────────────────────────────────
_HARD_TYPES = {"year", "before", "after", "range"}
_SOFT_TYPES = {"latest"}


def _is_hard(time_intent: Optional[dict]) -> bool:
    return bool(time_intent) and time_intent.get("type") in _HARD_TYPES


def _is_soft(time_intent: Optional[dict]) -> bool:
    return bool(time_intent) and time_intent.get("type") in _SOFT_TYPES


def _doc_passes_time_filter(doc: Document, time_intent: dict) -> bool:
    """对单个 child Document 做时间硬过滤（用于 BM25 post-filter）。"""
    rng = time_intent.get("range") or {}
    gte, lte = rng.get("gte"), rng.get("lte")
    if gte is None or lte is None:
        return True

    field_name = time_intent.get("field", "doc_date")
    md = doc.metadata or {}
    if field_name == "upload_date":
        v = md.get("upload_date", 0)
        return gte <= v <= lte

    if not md.get("has_doc_date"):
        return False
    dmin = md.get("doc_date_min", 0)
    dmax = md.get("doc_date_max", 0)
    return dmin <= lte and dmax >= gte


def _doc_sort_date(doc: Document, field_name: str) -> int:
    """取文档用于排序的日期（latest 用 _max，upload 用单值）。"""
    md = doc.metadata or {}
    if field_name == "upload_date":
        return md.get("upload_date", 0) or 0
    return md.get("doc_date_max", 0) or 0


# ────────────────────────────────────────────────────────────────
# BM25 时间过滤包装器
# 因 BM25Retriever 不支持原生 metadata filter，必须召回扩大化后 post-filter
# ────────────────────────────────────────────────────────────────
class TimeFilteredBM25Wrapper(BaseRetriever):
    """
    包装 BM25Retriever：召回 K·M 后按 time_intent 过滤，截断到原 K。
    继承 BaseRetriever 以兼容 EnsembleRetriever 的 invoke 接口。
    """
    base: BM25Retriever
    time_intent: dict
    target_k: int

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        raw = self.base.invoke(query)
        passed = [d for d in raw if _doc_passes_time_filter(d, self.time_intent)]
        return passed[: self.target_k]


# ────────────────────────────────────────────────────────────────
# Hybrid Retriever 数据类
# ────────────────────────────────────────────────────────────────
@dataclass
class ParentChildHybridRetriever:
    ensemble_retriever: object
    parent_docstore: ParentDocStore
    time_intent: Optional[dict] = None

    def invoke(self, query: str) -> List[Document]:
        child_hits = self.ensemble_retriever.invoke(
            query, config={"callbacks": [RetrievalLoggingHandler()]}
        )
        return hydrate_parent_results(
            child_hits,
            parent_docstore=self.parent_docstore,
            limit=rag_config.MAX_HYDRATED_PARENTS,
            time_intent=self.time_intent,
        )


# ────────────────────────────────────────────────────────────────
# 缓存底层 BM25 索引：避免每次 query 都从 all_chunks 重建（开销大）
# ────────────────────────────────────────────────────────────────
@dataclass
class _RetrieverBase:
    """缓存与 query 无关的重资源：all_chunks 引用 + BM25 base + parent_docstore。"""
    all_chunks: List[Document]
    bm25_base: BM25Retriever
    parent_docstore: ParentDocStore


_base_cache: Optional[_RetrieverBase] = None


def _get_or_build_base(
    all_chunks: List[Document],
    parent_docstore: ParentDocStore | None = None,
) -> _RetrieverBase:
    """构建或复用 BM25 base。当 chunks 数量变化时自动重建。"""
    global _base_cache
    docstore = parent_docstore or get_parent_docstore()
    if _base_cache is None or len(_base_cache.all_chunks) != len(all_chunks):
        bm25 = BM25Retriever.from_documents(all_chunks)
        _base_cache = _RetrieverBase(all_chunks=all_chunks, bm25_base=bm25, parent_docstore=docstore)
    return _base_cache


def reset_retriever_cache() -> None:
    """重新 ingest 后调用，强制重建 BM25 索引。"""
    global _base_cache
    _base_cache = None


def build_hybrid_retriever(
    all_chunks: List[Document],
    parent_docstore: ParentDocStore | None = None,
    time_intent: Optional[dict] = None,
) -> ParentChildHybridRetriever:
    """
    构建 child 级混合检索器，并按 time_intent 决定 filter 策略。

    硬意图（year/before/after/range）：
      - 语义路：注入 ChromaDB native filter，k *= HARD_FILTER_K_MULTIPLIER
      - BM25 路：用 TimeFilteredBM25Wrapper 包装，召回扩大化后 post-filter
    软意图（latest）：
      - 不做 filter，仅 hydrate 阶段按日期排序
    none：原流程
    """
    vs = get_vectorstore()
    base = _get_or_build_base(all_chunks, parent_docstore)
    docstore = base.parent_docstore

    is_hard = _is_hard(time_intent)
    semantic_k = rag_config.SEMANTIC_TOP_K
    bm25_k = rag_config.BM25_TOP_K
    semantic_kwargs: dict[str, Any] = {"k": semantic_k}
    if is_hard:
        where = build_time_filter(time_intent)
        if where:
            semantic_kwargs["filter"] = where
            semantic_kwargs["k"] = semantic_k * rag_config.HARD_FILTER_K_MULTIPLIER

    semantic_retriever = vs.as_retriever(search_type="similarity", search_kwargs=semantic_kwargs)
    semantic_retriever.tags = ["retriever:SemanticRetriever向量检索"]

    if is_hard:
        # BM25 召回扩大化 + post-filter
        base.bm25_base.k = bm25_k * rag_config.BM25_FILTER_K_MULTIPLIER
        bm25_retriever: BaseRetriever = TimeFilteredBM25Wrapper(
            base=base.bm25_base,
            time_intent=time_intent,
            target_k=bm25_k,
        )
        bm25_retriever.tags = ["retriever:BM25Retriever关键词检索(时间过滤)"]
    else:
        base.bm25_base.k = bm25_k
        bm25_retriever = base.bm25_base
        bm25_retriever.tags = ["retriever:BM25Retriever关键词检索"]

    ensemble = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[rag_config.SEMANTIC_WEIGHT, 1 - rag_config.SEMANTIC_WEIGHT],
    )
    ensemble.tags = ["retriever:Ensemble混合结果"]
    return ParentChildHybridRetriever(ensemble, docstore, time_intent=time_intent)


def get_hybrid_retriever(time_intent: Optional[dict] = None):
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
        return build_hybrid_retriever(all_chunks, time_intent=time_intent)
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────
# Parent 聚合 + 二级排序
# ────────────────────────────────────────────────────────────────
def hydrate_parent_results(
    child_hits: List[Document],
    parent_docstore: ParentDocStore,
    limit: int | None = None,
    time_intent: Optional[dict] = None,
) -> List[Document]:
    """
    将 child 级命中聚合为 parent 级结果。

    排序 key 按 time_intent.type 切换：
      - latest:                    (-date, -score, -child_count)  时间优先
      - year/range/before/after:   (-score, -child_count, -date)  相关性优先（时间已过滤）
      - none:                      (-score, -child_count)         原 key
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

        score = child.metadata.get("similarity_score") or 1 / (60 + rank + 1)
        scores[parent_id] = scores.get(parent_id, 0.0) + score
        matched_child_ids.setdefault(parent_id, [])
        if child_id and child_id not in matched_child_ids[parent_id]:
            matched_child_ids[parent_id].append(child_id)

        current_best = best_meta.get(parent_id)
        if current_best is None or score > current_best["score"]:
            best_meta[parent_id] = {"score": score, "metadata": child.metadata}

    sort_key = _make_sort_key(scores, matched_child_ids, best_meta, time_intent)
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
            "best_child_score": round(best["score"], 4),
            "section_path": parent_doc.metadata.get(
                "section_path", best["metadata"].get("section_path", "未命名章节"),
            ),
            "page_range": parent_doc.metadata.get(
                "page_range", best["metadata"].get("page_range", "?"),
            ),
        }
        results.append(Document(page_content=parent_doc.page_content, metadata=metadata))

    return results


def _make_sort_key(
    scores: dict[str, float],
    matched_child_ids: dict[str, list[str]],
    best_meta: dict[str, dict],
    time_intent: Optional[dict],
) -> Callable[[str], tuple]:
    """根据 time_intent 类型返回排序 key 函数。"""
    if not time_intent or time_intent.get("type") == "none":
        return lambda pid: (-scores[pid], -len(matched_child_ids.get(pid, [])))

    field_name = time_intent.get("field", "doc_date")

    def _date_for(pid: str) -> int:
        md = best_meta[pid]["metadata"]
        if field_name == "upload_date":
            return md.get("upload_date", 0) or 0
        return md.get("doc_date_max", 0) or 0

    if _is_soft(time_intent):  # latest：时间优先
        return lambda pid: (-_date_for(pid), -scores[pid], -len(matched_child_ids.get(pid, [])))

    # 硬意图：时间已过滤过，相关性优先，同分时偏新
    return lambda pid: (-scores[pid], -len(matched_child_ids.get(pid, [])), -_date_for(pid))


# ────────────────────────────────────────────────────────────────
# 对外检索 API
# ────────────────────────────────────────────────────────────────
def retrieve_with_hybrid(
    query: str,
    top_k: int | None = None,
    ensemble_retriever: Optional[ParentChildHybridRetriever] = None,
    time_intent: Optional[dict] = None,
) -> List[Document]:
    """
    使用混合检索器检索相关 parent 文档。

    若传入 time_intent，会按其类型决定 filter / 排序策略。
    """
    if ensemble_retriever is None or (
        time_intent is not None
        and getattr(ensemble_retriever, "time_intent", None) != time_intent
    ):
        # time_intent 与现有 retriever 不一致时，重建（仅复用 BM25 base）
        ensemble_retriever = get_hybrid_retriever(time_intent=time_intent)
        if ensemble_retriever is None:
            return []

    top_k = top_k or rag_config.FINAL_TOP_K
    results = ensemble_retriever.invoke(query)
    return results[:top_k]


# ────────────────────────────────────────────────────────────────
# 检索过程日志回调
# ────────────────────────────────────────────────────────────────
class RetrievalLoggingHandler(BaseCallbackHandler):
    """兼容 LangChain 回调签名的检索日志处理器"""

    def __init__(self):
        self._current_query: str = ""
        self._current_name: str = "Unknown"
        self._name_stack: list[str] = []

    def on_retriever_start(self, serialized, query, *, run_id=None, parent_run_id=None, **kwargs):
        name = self._extract_name(serialized, kwargs)
        self._current_query = query
        self._current_name = name

        indent = "  " * len(self._name_stack)
        self._name_stack.append(name)
        print(f"\n{indent}🔍 [{name}] 开始检索: {query[:80]}...")

    def on_retriever_end(self, response, *, run_id=None, parent_run_id=None, **kwargs):
        name = self._name_stack.pop() if self._name_stack else "Unknown"
        documents = response
        indent = "  " * len(self._name_stack)

        print(f"{indent}✅ [{name}] 完成，返回 {len(documents)} 个文档")

        for i, doc in enumerate(documents[:3], 1):
            preview = doc.page_content[:80].replace('\n', ' ')
            print(f"{indent}   {i}. {preview}...")
        print()

    def _extract_name(self, serialized, kwargs) -> str:
        tags = kwargs.get("tags") or []
        for tag in tags:
            if tag.startswith("retriever:"):
                return tag.split(":", 1)[1]

        if serialized and isinstance(serialized, dict):
            cls_name = serialized.get("name")
            if cls_name:
                return cls_name

        return "Unknown"
