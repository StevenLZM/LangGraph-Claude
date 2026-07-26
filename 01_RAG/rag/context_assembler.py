from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

from langchain_core.documents import Document


def assemble_final_context(
    documents: Sequence[Document],
    *,
    max_documents: int = 6,
    token_budget: int = 6000,
    tokenizer_name: str = "cl100k_base",
) -> list[Document]:
    if max_documents <= 0 or token_budget <= 0:
        return []

    encoding = _get_encoding(tokenizer_name)
    accepted: list[Document] = []
    seen_parent_ids: set[str] = set()
    used_tokens = 0

    for document in documents:
        if len(accepted) >= max_documents:
            break
        parent_id = str(document.metadata.get("parent_id") or "").strip()
        if not parent_id or parent_id in seen_parent_ids:
            continue

        evidence_id = f"S{len(accepted) + 1}"
        metadata = {
            **document.metadata,
            "evidence_id": evidence_id,
            "context_truncated": False,
        }
        candidate = Document(
            page_content=document.page_content,
            metadata=metadata,
        )
        full_count = _count_tokens(render_evidence_block(candidate), encoding)
        if used_tokens + full_count <= token_budget:
            candidate.metadata["context_token_count"] = full_count
            accepted.append(candidate)
            seen_parent_ids.add(parent_id)
            used_tokens += full_count
            continue

        header_count = _count_tokens(_evidence_header(candidate), encoding)
        content_budget = token_budget - used_tokens - header_count
        if content_budget <= 0:
            break
        truncated_content = _truncate_at_boundary(
            document.page_content,
            max_tokens=content_budget,
            encoding=encoding,
        )
        if not truncated_content:
            break
        truncated = Document(
            page_content=truncated_content,
            metadata={
                **metadata,
                "context_truncated": True,
            },
        )
        truncated_count = _count_tokens(
            render_evidence_block(truncated),
            encoding,
        )
        while truncated_count > token_budget - used_tokens and truncated.page_content:
            reduced_budget = max(
                0,
                _count_tokens(truncated.page_content, encoding) - 1,
            )
            reduced_content = _truncate_at_boundary(
                truncated.page_content,
                max_tokens=reduced_budget,
                encoding=encoding,
            )
            if reduced_content == truncated.page_content:
                break
            truncated = Document(
                page_content=reduced_content,
                metadata=dict(truncated.metadata),
            )
            truncated_count = _count_tokens(
                render_evidence_block(truncated),
                encoding,
            )
        if not truncated.page_content or truncated_count > token_budget - used_tokens:
            break
        truncated.metadata["context_token_count"] = truncated_count
        accepted.append(truncated)
        seen_parent_ids.add(parent_id)
        break
    return accepted


def render_evidence_block(document: Document) -> str:
    return _evidence_header(document) + document.page_content


def _evidence_header(document: Document) -> str:
    metadata = document.metadata
    evidence_id = metadata.get("evidence_id", "S?")
    source = metadata.get("source", "未知")
    page = metadata.get("page_range") or metadata.get("page") or "?"
    parent_id = metadata.get("parent_id", "")
    version = metadata.get("version") or metadata.get("doc_version") or ""
    version_text = f" | 版本: {version}" if version else ""
    section = metadata.get("section_path")
    section_text = f" | 章节: {section}" if section else ""
    return (
        f"[{evidence_id}] 来源: {source} | 第{page}页"
        f"{section_text} | parent_id: {parent_id}{version_text}\n"
    )


def _truncate_at_boundary(
    text: str,
    *,
    max_tokens: int,
    encoding: Any,
) -> str:
    if max_tokens <= 0:
        return ""
    tokens = encoding.encode(str(text))
    if len(tokens) <= max_tokens:
        return str(text).strip()
    decoded = encoding.decode(tokens[:max_tokens]).rstrip()
    if not decoded:
        return ""
    boundary_index = max(
        decoded.rfind(boundary)
        for boundary in ("\n", "。", "！", "？", ".", "!", "?", "；", ";")
    )
    if boundary_index >= 0:
        return decoded[: boundary_index + 1].rstrip()
    return decoded


def _count_tokens(text: str, encoding: Any) -> int:
    return len(encoding.encode(text))


@lru_cache(maxsize=8)
def _get_encoding(tokenizer_name: str) -> Any:
    import tiktoken

    return tiktoken.get_encoding(tokenizer_name)
