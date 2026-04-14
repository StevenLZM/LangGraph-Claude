"""
rag/chunker.py — 文本分块策略
提供两种策略：
  A. RecursiveCharacterTextSplitter（默认，速度快）
  B. 基于句子边界的精细分块（语义感知）
"""
from __future__ import annotations
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import rag_config


def chunk_documents(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    strategy: str = "recursive",
) -> List[Document]:
    """
    将文档列表分块

    Args:
        documents: load_pdf 返回的文档列表（按页）
        chunk_size: 每块字符数，None 时使用配置文件值
        chunk_overlap: 块间重叠字符数
        strategy: "recursive"（默认）| "sentence"

    Returns:
        分块后的 Document 列表，每个 chunk 携带完整 metadata 及 chunk_index
    """
    chunk_size = chunk_size or rag_config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or rag_config.CHUNK_OVERLAP

    if strategy == "recursive":
        return _recursive_chunk(documents, chunk_size, chunk_overlap)
    elif strategy == "sentence":
        return _sentence_aware_chunk(documents, chunk_size, chunk_overlap)
    else:
        raise ValueError(f"未知的分块策略: {strategy}")


def _recursive_chunk(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    RecursiveCharacterTextSplitter：
    - 优先按段落（\\n\\n）分割
    - 其次按句子（。！？）分割
    - 最后按字符分割
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",          # 段落
            "\n",            # 换行
            "。", "！", "？", # 中文句号
            ".", "!", "?",   # 英文句号
            "；", ";",       # 分号
            "，", ",",       # 逗号（最后考虑）
            " ",             # 空格
            "",              # 字符级
        ],
        length_function=len,
        is_separator_regex=False,
        keep_separator=True,
    )

    chunks = splitter.split_documents(documents)

    # 补充 chunk_index metadata（每个文档内从0开始计数）
    doc_chunk_counts: dict[str, int] = {}
    enriched = []
    for chunk in chunks:
        doc_id = chunk.metadata.get("doc_id", "unknown")
        idx = doc_chunk_counts.get(doc_id, 0)
        doc_chunk_counts[doc_id] = idx + 1

        # 确保 metadata 完整
        new_metadata = {**chunk.metadata, "chunk_index": idx}
        enriched.append(Document(
            page_content=chunk.page_content,
            metadata=new_metadata,
        ))

    return enriched


def _sentence_aware_chunk(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    句子感知分块：在 RecursiveCharacterTextSplitter 基础上，
    确保分块边界落在句子结束处（而非词语中间）
    """
    import re

    chunks = _recursive_chunk(documents, chunk_size, chunk_overlap)
    refined = []

    SENTENCE_END = re.compile(r"([。！？.!?])")

    for chunk in chunks:
        text = chunk.page_content

        # 如果 chunk 末尾不是句子结束符，尝试截断到最后一个句子结束处
        if len(text) > 50 and not text.rstrip()[-1:] in {"。", "！", "？", ".", "!", "?"}:
            # 找最后一个句子结束符
            matches = list(SENTENCE_END.finditer(text))
            if matches and matches[-1].end() > len(text) * 0.7:
                # 只在丢失不超过30%内容时截断
                text = text[:matches[-1].end()]

        refined.append(Document(
            page_content=text,
            metadata=chunk.metadata,
        ))

    return refined


def get_chunk_stats(chunks: List[Document]) -> dict:
    """返回分块统计信息，用于 UI 展示"""
    if not chunks:
        return {"total": 0, "avg_len": 0, "min_len": 0, "max_len": 0}

    lengths = [len(c.page_content) for c in chunks]
    return {
        "total": len(chunks),
        "avg_len": round(sum(lengths) / len(lengths)),
        "min_len": min(lengths),
        "max_len": max(lengths),
    }
