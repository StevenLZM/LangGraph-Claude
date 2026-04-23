"""
rag/chunker.py — 分层分块策略

V2 目标：
1. 先按结构粗切 section
2. 再按 token 约束生成 parent / child chunks
3. 为索引层提供稳定内容 ID 和聚合 metadata
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import time
from typing import Iterable, Iterator, List, Sequence

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import rag_config


DEFAULT_PARENT_TARGET_TOKENS = 900
DEFAULT_PARENT_MAX_TOKENS = 1200
DEFAULT_PARENT_OVERLAP_TOKENS = 100
DEFAULT_CHILD_TARGET_TOKENS = 280
DEFAULT_CHILD_MAX_TOKENS = 360
DEFAULT_CHILD_OVERLAP_TOKENS = 60
DEFAULT_TOKENIZER_NAME = "cl100k_base"


@dataclass(frozen=True)
class Section:
    text: str
    metadata: dict


@dataclass
class ChunkingResult(Sequence[Document]):
    parents: List[Document]
    children: List[Document]
    stats: dict

    def __len__(self) -> int:
        return len(self.children)

    def __iter__(self) -> Iterator[Document]:
        return iter(self.children)

    def __getitem__(self, index):
        return self.children[index]

    def __eq__(self, other) -> bool:
        if isinstance(other, list):
            return self.children == other
        return super().__eq__(other)


def chunk_documents(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    strategy: str = "hierarchical_v2",
) -> ChunkingResult:
    """
    将文档分为 parent / child 两层 chunk。

    兼容旧调用方式：保留 chunk_size / chunk_overlap 参数，但优先使用 v2 配置。
    """
    if not documents:
        return ChunkingResult(parents=[], children=[], stats=_empty_stats())

    if strategy not in {"hierarchical_v2", "recursive", "sentence"}:
        raise ValueError(f"未知的分块策略: {strategy}")

    tokenizer_name = getattr(rag_config, "TOKENIZER_NAME", DEFAULT_TOKENIZER_NAME)
    parent_target = getattr(rag_config, "PARENT_TARGET_TOKENS", DEFAULT_PARENT_TARGET_TOKENS)
    parent_max = getattr(rag_config, "PARENT_MAX_TOKENS", DEFAULT_PARENT_MAX_TOKENS)
    parent_overlap = getattr(rag_config, "PARENT_OVERLAP_TOKENS", DEFAULT_PARENT_OVERLAP_TOKENS)
    child_target = chunk_size or getattr(rag_config, "CHILD_TARGET_TOKENS", DEFAULT_CHILD_TARGET_TOKENS)
    child_max = max(child_target, getattr(rag_config, "CHILD_MAX_TOKENS", DEFAULT_CHILD_MAX_TOKENS))
    child_overlap = chunk_overlap or getattr(
        rag_config, "CHILD_OVERLAP_TOKENS", DEFAULT_CHILD_OVERLAP_TOKENS
    )

    token_length = _build_token_length(tokenizer_name)
    parent_splitter = _build_splitter(
        tokenizer_name=tokenizer_name,
        chunk_size=max(parent_target, parent_max),
        chunk_overlap=parent_overlap,
    )
    child_splitter = _build_splitter(
        tokenizer_name=tokenizer_name,
        chunk_size=child_max,
        chunk_overlap=child_overlap,
    )

    all_parents: list[Document] = []
    all_children: list[Document] = []

    for doc_group in _group_documents_by_id(documents):
        all_text = "\n\n".join(_normalize_text(doc.page_content) for doc in doc_group if doc.page_content.strip())
        doc_version = _stable_hash(all_text, length=12)
        sections = _build_sections(doc_group)

        for section_index, section in enumerate(sections):
            parent_texts = parent_splitter.split_text(section.text) or [section.text]

            for parent_index, parent_text in enumerate(parent_texts):
                normalized_parent = _normalize_text(parent_text)
                parent_hash = _stable_hash(normalized_parent, length=12)
                parent_id = f"{section.metadata['doc_id']}:p:{parent_hash}"
                parent_metadata = {
                    # 将 section.metadata 字典中的所有键值对展开，合并到新字典中
                    **section.metadata,
                    "doc_version": doc_version,
                    "chunk_role": "parent",
                    "parent_id": parent_id,
                    "parent_index": len(all_parents),
                    "section_index": section_index,
                    "chunk_index": len(all_parents),
                    "token_count": token_length(normalized_parent),
                    "page_range": _format_page_range(
                        section.metadata.get("page_start"),
                        section.metadata.get("page_end"),
                    ),
                }
                parent_doc = Document(page_content=normalized_parent, metadata=parent_metadata)
                all_parents.append(parent_doc)

                child_texts = child_splitter.split_text(normalized_parent) or [normalized_parent]
                for child_text in child_texts:
                    normalized_child = _normalize_text(child_text)
                    child_hash = _stable_hash(normalized_child, length=12)
                    child_id = f"{section.metadata['doc_id']}:c:{child_hash}"
                    child_metadata = {
                        **section.metadata,
                        "doc_version": doc_version,
                        "chunk_role": "child",
                        "parent_id": parent_id,
                        "child_id": child_id,
                        "child_index": len(all_children),
                        "chunk_index": len(all_children),
                        "section_index": section_index,
                        "section_path": section.metadata.get("section_path", "未命名章节"),
                        "token_count": token_length(normalized_child),
                        "page_range": _format_page_range(
                            section.metadata.get("page_start"),
                            section.metadata.get("page_end"),
                        ),
                        # 当前年月，int类型 -> 202604
                        "upload_time": int(time.strftime("%Y%m"))
                    }
                    all_children.append(Document(page_content=normalized_child, metadata=child_metadata))

    stats = get_chunk_stats(ChunkingResult(parents=all_parents, children=all_children, stats={}))
    return ChunkingResult(parents=all_parents, children=all_children, stats=stats)


def _group_documents_by_id(documents: Iterable[Document]) -> list[list[Document]]:
    grouped: dict[str, list[Document]] = {}
    ordered: list[str] = []

    for doc in documents:
        doc_id = doc.metadata.get("doc_id", "unknown")
        if doc_id not in grouped:
            grouped[doc_id] = []
            ordered.append(doc_id)
        grouped[doc_id].append(doc)

    return [grouped[doc_id] for doc_id in ordered]


def _build_sections(documents: List[Document]) -> list[Section]:
    sections: list[Section] = []
    current_heading = "导言"
    current_blocks: list[str] = []
    current_meta: dict | None = None

    def flush():
        nonlocal current_blocks, current_meta
        if not current_blocks or current_meta is None:
            current_blocks = []
            return

        section_text = _normalize_text("\n\n".join(current_blocks))
        if section_text:
            sections.append(
                Section(
                    text=section_text,
                    metadata={
                        **current_meta,
                        "section_path": current_heading,
                    },
                )
            )
        current_blocks = []
        current_meta = None

    for doc in documents:
        page = doc.metadata.get("page", 1)
        for block in _split_blocks(doc.page_content):
            if _is_heading(block):
                flush()
                current_heading = block.strip()
                current_meta = _build_section_metadata(doc.metadata, page_start=page, page_end=page)
                current_blocks = [current_heading]
                continue

            if current_meta is None:
                current_meta = _build_section_metadata(doc.metadata, page_start=page, page_end=page)
            else:
                current_meta["page_end"] = page

            current_blocks.append(block)

    flush()
    return sections


def _build_section_metadata(base: dict, page_start: int, page_end: int) -> dict:
    return {
        **base,
        "page_start": page_start,
        "page_end": page_end,
    }


def _split_blocks(text: str) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    raw_blocks = re.split(r"\n{2,}", normalized)
    return [block.strip() for block in raw_blocks if block.strip()]


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ").replace("\t", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = [line.strip() for line in text.split("\n")]
    merged: list[str] = []
    buffer = ""
    for line in lines:
        if not line:
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            continue

        if not buffer:
            buffer = line
            continue

        if _should_merge_lines(buffer, line):
            buffer = f"{buffer} {line}"
        else:
            merged.append(buffer.strip())
            buffer = line

    if buffer:
        merged.append(buffer.strip())

    return "\n\n".join(part for part in merged if part)


def _should_merge_lines(previous: str, current: str) -> bool:
    prev_tail = previous.rstrip()[-1:] if previous.strip() else ""
    if prev_tail in {"。", "！", "？", ".", "!", "?", ":", "："}:
        return False
    if _is_heading(current):
        return False
    if re.match(r"^[-*•]\s+", current):
        return False
    return True


def _is_heading(block: str) -> bool:
    text = block.strip()
    if not text or len(text) > 80 or "\n" in text:
        return False

    heading_patterns = [
        r"^第[一二三四五六七八九十百千\d]+[章节篇部分]",
        r"^[0-9]+[.)、．]",
        r"^[一二三四五六七八九十]+、",
        r"^[A-Z][A-Za-z0-9 _-]{1,40}$",
    ]
    if any(re.match(pattern, text) for pattern in heading_patterns):
        return True

    if text.endswith(("：", ":")) and len(text) <= 40:
        return True

    return False


def _build_splitter(
    tokenizer_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=tokenizer_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            "。", "！", "？",
            ".", "!", "?",
            "；", ";",
            "，", ",",
            " ",
            "",
        ],
        keep_separator=True,
    )


def _build_token_length(tokenizer_name: str):
    encoder = tiktoken.get_encoding(tokenizer_name)

    def _token_length(text: str) -> int:
        if not text:
            return 0
        return len(encoder.encode(text))

    return _token_length


def _stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _format_page_range(page_start: int | None, page_end: int | None) -> str:
    if page_start is None and page_end is None:
        return "?"
    if page_start == page_end or page_end is None:
        return str(page_start)
    return f"{page_start}-{page_end}"


def _empty_stats() -> dict:
    return {
        "total_parents": 0,
        "total_children": 0,
        "avg_parent_tokens": 0,
        "avg_child_tokens": 0,
        "min_child_tokens": 0,
        "max_child_tokens": 0,
    }


def get_chunk_stats(chunks: ChunkingResult | List[Document]) -> dict:
    """返回分块统计信息，用于 UI 展示。"""
    if isinstance(chunks, ChunkingResult):
        parents = chunks.parents
        children = chunks.children
    else:
        parents = []
        children = chunks

    if not parents and not children:
        return _empty_stats()

    parent_lengths = [doc.metadata.get("token_count", 0) for doc in parents]
    child_lengths = [doc.metadata.get("token_count", 0) or len(doc.page_content) for doc in children]
    return {
        "total_parents": len(parents),
        "total_children": len(children),
        "avg_parent_tokens": round(sum(parent_lengths) / len(parent_lengths)) if parent_lengths else 0,
        "avg_child_tokens": round(sum(child_lengths) / len(child_lengths)) if child_lengths else 0,
        "min_child_tokens": min(child_lengths) if child_lengths else 0,
        "max_child_tokens": max(child_lengths) if child_lengths else 0,
        # 兼容旧 UI / 测试字段，默认映射到 child 层统计
        "total": len(children),
        "avg_len": round(sum(child_lengths) / len(child_lengths)) if child_lengths else 0,
        "min_len": min(child_lengths) if child_lengths else 0,
        "max_len": max(child_lengths) if child_lengths else 0,
    }
