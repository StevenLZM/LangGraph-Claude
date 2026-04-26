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
from rag.date_extractor import DateExtractionResult, extract_dates


DEFAULT_PARENT_TARGET_TOKENS = 900
DEFAULT_PARENT_MAX_TOKENS = 1200
DEFAULT_PARENT_OVERLAP_TOKENS = 100
DEFAULT_CHILD_TARGET_TOKENS = 280
DEFAULT_CHILD_MAX_TOKENS = 360
DEFAULT_CHILD_OVERLAP_TOKENS = 60
DEFAULT_TOKENIZER_NAME = "cl100k_base"

ATOMIC_PLACEHOLDER_PREFIX = "\x00ATOM"
ATOMIC_PLACEHOLDER_SUFFIX = "\x00"


def _protect_atomics(text: str, atomics: Sequence[str]) -> tuple[str, dict[str, str]]:
    """把不可切原子块（表格/代码）替换为占位符，避免被 splitter 切散或被 _normalize_text 折行。"""
    placeholder_map: dict[str, str] = {}
    if not atomics:
        return text, placeholder_map
    work = text
    for i, atomic in enumerate(atomics):
        if not atomic:
            continue
        idx = work.find(atomic)
        if idx == -1:
            continue
        ph = f"{ATOMIC_PLACEHOLDER_PREFIX}{i}{ATOMIC_PLACEHOLDER_SUFFIX}"
        work = work[:idx] + f"\n\n{ph}\n\n" + work[idx + len(atomic):]
        placeholder_map[ph] = atomic
    return work, placeholder_map


def _restore_atomics(text: str, placeholder_map: dict[str, str]) -> str:
    if not placeholder_map:
        return text
    out = text
    for ph, atomic in placeholder_map.items():
        if ph in out:
            out = out.replace(ph, atomic)
    return out


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
        if _has_structured(doc_group):
            sections = _build_sections_structured(doc_group)
        else:
            sections = _build_sections(doc_group)

        for section_index, section in enumerate(sections):
            atomics = section.metadata.get("atomic_texts", ()) or ()
            heading_level = section.metadata.get("heading_level", 0)
            work_text, placeholder_map = _protect_atomics(section.text, atomics)
            parent_works = parent_splitter.split_text(work_text) or [work_text]

            for parent_index, parent_work in enumerate(parent_works):
                normalized_parent = _restore_atomics(_normalize_text(parent_work), placeholder_map)
                parent_hash = _stable_hash(normalized_parent, length=12)
                parent_id = f"{section.metadata['doc_id']}:p:{parent_hash}"

                # 日期抽取：parent 级一次，child 继承，避免重复调用 LLM
                doc_id = section.metadata.get("doc_id", "anonymous")
                dates = extract_dates(normalized_parent, doc_id=doc_id)
                upload_date = int(time.strftime("%Y%m%d"))
                date_fields = {
                    "upload_date": upload_date,
                    "doc_date_min": dates.min,
                    "doc_date_max": dates.max,
                    "has_doc_date": dates.found,
                }

                section_meta_clean = {k: v for k, v in section.metadata.items() if k != "atomic_texts"}
                parent_metadata = {
                    **section_meta_clean,
                    "doc_version": doc_version,
                    "chunk_role": "parent",
                    "parent_id": parent_id,
                    "parent_index": len(all_parents),
                    "section_index": section_index,
                    "chunk_index": len(all_parents),
                    "heading_level": heading_level,
                    "token_count": token_length(normalized_parent),
                    "page_range": _format_page_range(
                        section.metadata.get("page_start"),
                        section.metadata.get("page_end"),
                    ),
                    **date_fields,
                }
                parent_doc = Document(page_content=normalized_parent, metadata=parent_metadata)
                all_parents.append(parent_doc)

                child_works = child_splitter.split_text(parent_work) or [parent_work]
                for child_work in child_works:
                    is_pure_atomic = (
                        bool(placeholder_map)
                        and child_work.strip() in placeholder_map
                    )
                    if is_pure_atomic:
                        normalized_child = placeholder_map[child_work.strip()].strip()
                    else:
                        normalized_child = _restore_atomics(_normalize_text(child_work), placeholder_map)
                    if not normalized_child:
                        continue
                    is_atomic_child = is_pure_atomic
                    child_hash = _stable_hash(normalized_child, length=12)
                    child_id = f"{section.metadata['doc_id']}:c:{child_hash}"
                    child_metadata = {
                        **section_meta_clean,
                        "doc_version": doc_version,
                        "chunk_role": "child",
                        "parent_id": parent_id,
                        "child_id": child_id,
                        "child_index": len(all_children),
                        "chunk_index": len(all_children),
                        "section_index": section_index,
                        "section_path": section.metadata.get("section_path", "未命名章节"),
                        "heading_level": heading_level,
                        "is_atomic": is_atomic_child,
                        "token_count": token_length(normalized_child),
                        "page_range": _format_page_range(
                            section.metadata.get("page_start"),
                            section.metadata.get("page_end"),
                        ),
                        **date_fields,
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


def _has_structured(documents: Sequence[Document]) -> bool:
    return any(doc.metadata.get("structured_blocks") for doc in documents)


def _build_sections_structured(documents: List[Document]) -> list[Section]:
    """基于 loader 提供的 structured_blocks 构建 section。表格/代码记入 atomic_texts。"""
    sections: list[Section] = []
    current_heading = "导言"
    current_level = 0
    current_parts: list[tuple[str, str]] = []  # (kind, text); kind in {"text","atomic"}
    current_atomics: list[str] = []
    current_meta: dict | None = None

    base_meta = {k: v for k, v in documents[0].metadata.items() if k != "structured_blocks"}

    def flush():
        nonlocal current_parts, current_atomics, current_meta
        if not current_parts or current_meta is None:
            current_parts = []
            current_atomics = []
            return
        rendered: list[str] = []
        for kind, text in current_parts:
            if kind == "atomic":
                rendered.append(text)
            else:
                norm = _normalize_text(text)
                if norm:
                    rendered.append(norm)
        section_text = "\n\n".join(rendered)
        if section_text:
            sections.append(Section(
                text=section_text,
                metadata={
                    **current_meta,
                    "section_path": current_heading,
                    "heading_level": current_level,
                    "atomic_texts": tuple(current_atomics),
                },
            ))
        current_parts = []
        current_atomics = []
        current_meta = None

    for doc in documents:
        blocks = doc.metadata.get("structured_blocks") or []
        default_page = doc.metadata.get("page", 1)
        for blk in blocks:
            btype = blk.get("type")
            page = blk.get("page", default_page)
            text = (blk.get("text") or "").strip()
            if not text:
                continue
            if btype == "heading":
                flush()
                current_heading = text
                current_level = int(blk.get("level", 0) or 0)
                current_meta = _build_section_metadata(base_meta, page_start=page, page_end=page)
                current_parts = [("text", current_heading)]
            else:
                if current_meta is None:
                    current_meta = _build_section_metadata(base_meta, page_start=page, page_end=page)
                else:
                    current_meta["page_end"] = max(current_meta.get("page_end", page), page)
                if btype in {"table", "code"}:
                    current_parts.append(("atomic", text))
                    current_atomics.append(text)
                else:
                    current_parts.append(("text", text))
    flush()
    return sections


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
    if ATOMIC_PLACEHOLDER_PREFIX in previous or ATOMIC_PLACEHOLDER_PREFIX in current:
        return False
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
