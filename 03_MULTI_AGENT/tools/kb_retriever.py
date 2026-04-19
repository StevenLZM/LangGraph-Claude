"""KB Retriever —— 适配 01_RAG 的 ParentChildHybridRetriever。

挑战：01_RAG 同时有 `mcp/` 和 `config/` 包，会与本项目（03）的同名模块冲突。
   - `mcp/` 与 pip 装的官方 MCP SDK 同名
   - `config/` 与本项目的 settings 模块同名
   - 不能直接 sys.path.insert + 全局 from rag... 否则破坏其他模块的 import 解析

策略：surgical sys.path/sys.modules 隔离
   1. 备份 sys.modules 中的 `config` / `mcp`（本项目的版本）
   2. 临时把 01_RAG 插到 sys.path 最前
   3. 加载 rag.retriever（其内部 `from config import rag_config` 解析到 01_RAG/config）
   4. 恢复 sys.path 和 sys.modules：本项目的 config/mcp 完整不受影响
   5. retriever 对象内部已 bind 到 01_RAG/config，可独立运行

这是工程务实选择：避免修改 01_RAG，避免拷贝 01_RAG 的 rag/ 到本项目。
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from pathlib import Path

from tools.base import SearchTool, ToolResult

logger = logging.getLogger(__name__)

_RAG_PATH = Path(__file__).resolve().parent.parent.parent / "01_RAG"


def _load_01rag_get_hybrid_retriever():
    """surgical import：把 rag.retriever 加载进来又不污染本项目的 sys.modules。"""
    if not _RAG_PATH.is_dir():
        return None

    rag_path_str = str(_RAG_PATH)

    # 备份本项目的 config / mcp（这两个是会冲突的高危模块名）
    saved: dict[str, object] = {}
    for name in list(sys.modules.keys()):
        if name == "config" or name.startswith("config.") or name == "mcp" or name.startswith("mcp."):
            saved[name] = sys.modules[name]

    # 临时把 01_RAG 插到 sys.path 最前
    sys.path.insert(0, rag_path_str)

    # 把 config / mcp 从 sys.modules 移除，强制重新解析到 01_RAG 下
    for name in list(saved.keys()):
        if name == "config" or name.startswith("config."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()

    try:
        from rag.retriever import get_hybrid_retriever  # type: ignore
        return get_hybrid_retriever
    finally:
        # 移除 01_RAG 路径
        try:
            sys.path.remove(rag_path_str)
        except ValueError:
            pass
        # 清掉 01_RAG 注入的 config / rag 模块（它们的 __file__ 在 01_RAG 下）
        for name in list(sys.modules.keys()):
            mod = sys.modules.get(name)
            f = getattr(mod, "__file__", "") or ""
            if name in {"config", "rag"} or name.startswith("config.") or name.startswith("rag."):
                if rag_path_str in f:
                    del sys.modules[name]
        # 恢复本项目的 config / mcp
        for name, mod in saved.items():
            sys.modules[name] = mod
        importlib.invalidate_caches()


_GET_HYBRID = None


def _ensure_loader():
    global _GET_HYBRID
    if _GET_HYBRID is None:
        _GET_HYBRID = _load_01rag_get_hybrid_retriever()
    return _GET_HYBRID


class KBRetriever:
    name = "kb_hybrid"
    source_type = "kb"

    def __init__(self) -> None:
        self._impl = None
        try:
            getter = _ensure_loader()
            if getter is None:
                logger.warning("[kb_retriever] 01_RAG 目录不存在，跳过")
                return
            self._impl = getter()
            if self._impl is None:
                logger.warning("[kb_retriever] 01_RAG 索引为空，search() 将返回 []")
        except Exception as e:
            logger.warning("[kb_retriever] 加载 01_RAG 失败: %s", e)

    async def search(self, query: str, *, top_k: int = 5) -> list[ToolResult]:
        if self._impl is None:
            return []
        try:
            docs = await asyncio.to_thread(self._impl.invoke, query)
        except Exception as e:
            logger.warning("[kb_retriever] 检索失败: %s", e)
            return []
        return [_to_tool_result(d) for d in docs[:top_k]]

    async def close(self) -> None:
        return None


def _to_tool_result(doc) -> ToolResult:
    md = getattr(doc, "metadata", {}) or {}
    return ToolResult(
        snippet=getattr(doc, "page_content", "")[:1000],
        source_url=md.get("source") or md.get("source_url") or "kb://local",
        relevance_score=float(md.get("best_child_score", 0.0)),
        extra={k: v for k, v in md.items() if k not in {"source"}},
    )


_: SearchTool = KBRetriever.__new__(KBRetriever)  # type: ignore[assignment]
