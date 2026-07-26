"""
rag/reranker.py — optional cross-encoder reranking for retrieved parent docs.
"""
from __future__ import annotations

from typing import Callable, Sequence

from langchain_core.documents import Document

from config import rerank_config


ScoreFunction = Callable[[str, Sequence[Document]], Sequence[float]]

_cross_encoder_instance = None


def get_cross_encoder():
    """Lazily load the configured sentence-transformers CrossEncoder."""
    global _cross_encoder_instance
    if _cross_encoder_instance is not None:
        return _cross_encoder_instance

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "启用 cross-encoder rerank 需要安装 sentence-transformers。"
        ) from exc

    kwargs = {}
    if rerank_config.DEVICE:
        kwargs["device"] = rerank_config.DEVICE
    _cross_encoder_instance = CrossEncoder(rerank_config.MODEL, **kwargs)
    return _cross_encoder_instance


def _score_with_cross_encoder(query: str, docs: Sequence[Document]) -> Sequence[float]:
    model = get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in docs]
    return model.predict(pairs, batch_size=rerank_config.BATCH_SIZE)


def rerank_documents(
    query: str,
    docs: list[Document],
    *,
    enabled: bool | None = None,
    top_n: int | None = None,
    score_threshold: float | None = None,
    scorer: ScoreFunction | None = None,
) -> list[Document]:
    """
    Rerank retrieved docs with a cross-encoder score.

    Disabled mode returns the original list unchanged so baseline retrieval remains
    byte-for-byte predictable unless RERANK_ENABLED=true.
    """
    active = rerank_config.ENABLED if enabled is None else enabled
    if not active or not docs:
        return docs

    limit = top_n if top_n is not None else rerank_config.TOP_N
    threshold = (
        rerank_config.SCORE_THRESHOLD
        if score_threshold is None
        else float(score_threshold)
    )
    score_fn = scorer or _score_with_cross_encoder
    scores = list(score_fn(query, docs))
    if len(scores) != len(docs):
        raise ValueError("rerank scorer 返回的分数数量必须与文档数量一致")

    scored_docs: list[tuple[float, int, Document]] = []
    for index, (doc, score) in enumerate(zip(docs, scores)):
        numeric_score = float(score)
        if numeric_score < threshold:
            continue
        metadata = {
            **(doc.metadata or {}),
            "rerank_score": round(numeric_score, 6),
        }
        scored_docs.append((
            numeric_score,
            index,
            Document(page_content=doc.page_content, metadata=metadata),
        ))

    reranked = [doc for _, _, doc in sorted(scored_docs, key=lambda item: (-item[0], item[1]))]
    if limit and limit > 0:
        return reranked[:limit]
    return reranked
