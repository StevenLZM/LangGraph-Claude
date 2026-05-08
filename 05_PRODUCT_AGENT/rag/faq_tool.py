from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class FAQRAGResult:
    matched: bool
    answer: str
    sources: list[str]
    backend: str
    error: str = ""


class FAQRAGTool:
    def __init__(self, backend_loader: Callable[[], Any] | None = None) -> None:
        self.backend_loader = backend_loader or load_01_rag_backend

    def search(self, query: str) -> FAQRAGResult:
        try:
            backend = self.backend_loader()
        except Exception as exc:
            return _unavailable_result(str(exc))
        if backend is None:
            return _unavailable_result("knowledge base backend is not configured")

        try:
            docs = list(backend.invoke(query) or [])
        except Exception as exc:
            return _unavailable_result(str(exc))
        if not docs:
            return FAQRAGResult(
                matched=False,
                answer="根据当前知识库，未找到该问题的相关信息。",
                sources=[],
                backend="01_RAG",
            )

        return FAQRAGResult(
            matched=True,
            answer=_format_answer(docs),
            sources=_format_sources(docs),
            backend="01_RAG",
        )


def load_01_rag_backend() -> Any | None:
    repo_root = Path(__file__).resolve().parents[2]
    rag_root = repo_root.parent / "01_RAG"
    if not rag_root.exists():
        return None
    rag_root_str = str(rag_root)
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "config" or name == "rag" or name.startswith("rag.")
    }
    for name in list(saved_modules):
        sys.modules.pop(name, None)
    if rag_root_str not in sys.path:
        sys.path.insert(0, rag_root_str)
    try:
        retriever_module = import_module("rag.retriever")
        return retriever_module.get_hybrid_retriever()
    finally:
        for name in [name for name in sys.modules if name == "config" or name == "rag" or name.startswith("rag.")]:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


def _unavailable_result(error: str) -> FAQRAGResult:
    return FAQRAGResult(
        matched=False,
        answer="根据当前知识库，未找到该问题的相关信息。",
        sources=[],
        backend="unavailable",
        error=error,
    )


def _format_answer(docs: list[Any]) -> str:
    snippets = []
    for doc in docs[:3]:
        content = " ".join(str(getattr(doc, "page_content", "")).split())
        if content:
            snippets.append(content[:180])
    if not snippets:
        return "根据当前知识库，未找到该问题的相关信息。"
    return "根据知识库：" + "；".join(snippets)


def _format_sources(docs: list[Any]) -> list[str]:
    sources: list[str] = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        source = str(metadata.get("source") or "unknown")
        page = metadata.get("page_range") or metadata.get("page")
        source_id = f"{source}#{page}" if page is not None else source
        if source_id not in sources:
            sources.append(source_id)
    return sources
