"""启动顺序 —— ENGINEERING.md §7.6 真实实现。

生命周期：
  startup:
    1. 工具初始化（Tavily/外部 MCP/DashScope内置/ArXiv/GitHub/KB）
    2. AsyncSqliteSaver checkpointer（异步 SQLite，需 aiosqlite）
    3. build_graph(checkpointer=...) 编译
  shutdown:
    关闭 registry（HTTP client / MCP session）→ 关 checkpointer 上下文
"""
from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.settings import settings
from graph.workflow import build_graph
from tools.arxiv_tool import ArxivTool
from tools.dashscope_search_tool import DashScopeSearchTool
from tools.github_tool import GitHubTool
from tools.kb_retriever import KBRetriever
from tools.mcp_loader import load_external_mcp
from tools.registry import ToolRegistry
from tools.tavily_tool import TavilyTool

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    registry: ToolRegistry = field(default_factory=ToolRegistry)
    graph: Any = None
    checkpointer: Any = None
    _exit_stack: AsyncExitStack | None = None


app_state = AppState()


async def startup() -> None:
    logger.info("[bootstrap] 开始启动 InsightLoop")

    # 1. 工具注册（顺序即 web 降级链：Tavily 主 → Brave MCP → DashScope 内置搜索）
    registry = ToolRegistry()
    try:
        registry.register(TavilyTool(api_key=settings.tavily_api_key))
    except Exception as e:
        logger.warning("[bootstrap] Tavily 初始化失败: %s", e)

    try:
        external = await load_external_mcp(settings.mcp_config_path)
        for tool in external:
            registry.register(tool)
    except Exception as e:
        logger.warning("[bootstrap] 外部 MCP 加载失败: %s", e)

    try:
        registry.register(DashScopeSearchTool())
    except Exception as e:
        logger.warning("[bootstrap] DashScopeSearch 初始化失败: %s", e)

    try:
        registry.register(ArxivTool())
    except Exception as e:
        logger.warning("[bootstrap] ArXiv 初始化失败: %s", e)
    try:
        registry.register(GitHubTool(token=settings.github_token))
    except Exception as e:
        logger.warning("[bootstrap] GitHub 初始化失败: %s", e)
    try:
        registry.register(KBRetriever())
    except Exception as e:
        logger.warning("[bootstrap] KBRetriever 初始化失败: %s", e)

    app_state.registry = registry
    logger.info("[bootstrap] 工具注册完成: %s", registry)

    # 2. 异步 Checkpointer（AsyncSqliteSaver 是 async context manager）
    app_state._exit_stack = AsyncExitStack()
    checkpointer = None
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        db_path = Path(settings.checkpointer_db)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        cm = AsyncSqliteSaver.from_conn_string(str(db_path))
        checkpointer = await app_state._exit_stack.enter_async_context(cm)
        logger.info("[bootstrap] AsyncSqliteSaver 就绪 → %s", db_path)
    except Exception as e:
        logger.warning("[bootstrap] Checkpointer 初始化失败（图将无持久化）: %s", e)

    # 3. 编译图
    app_state.checkpointer = checkpointer
    app_state.graph = build_graph(checkpointer=checkpointer)
    logger.info("[bootstrap] 图编译完成")


async def shutdown() -> None:
    try:
        await app_state.registry.close_all()
    except Exception as e:
        logger.warning("[bootstrap] registry 关闭异常: %s", e)
    try:
        if app_state._exit_stack is not None:
            await app_state._exit_stack.aclose()
    except Exception as e:
        logger.warning("[bootstrap] exit_stack 关闭异常: %s", e)
    logger.info("[bootstrap] shutdown 完成")
