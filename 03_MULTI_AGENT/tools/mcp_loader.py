"""外部 MCP 加载器 —— 解析 .mcp.json 并把每个 server 包装成 SearchTool。

当前真实实现：
  - brave-search → MCPBraveSearchTool（官方 mcp SDK）
  - 其他 server（如 filesystem）当前不进 web 降级链，仅打印发现日志

API key 解析：优先从 pydantic-settings 读，不依赖 os.environ。
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from config.settings import settings
from tools.base import SearchTool

logger = logging.getLogger(__name__)

_ENV_REF = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")

# .env 中的变量 → 实际值；mcp_loader 解析 ${VAR} 时优先查这里
_KEY_ALIASES = {
    "BRAVE_API_KEY": lambda: settings.brave_api_key,
    "TAVILY_API_KEY": lambda: settings.tavily_api_key,
    "GITHUB_TOKEN": lambda: settings.github_token,
    "DASHSCOPE_API_KEY": lambda: settings.dashscope_api_key,
}


def _resolve_env(value: str | None) -> str:
    if not value:
        return ""

    def _sub(m: re.Match) -> str:
        name = m.group(1)
        if name in _KEY_ALIASES:
            v = _KEY_ALIASES[name]()
            if v:
                return v
        return os.getenv(name, "")

    return _ENV_REF.sub(_sub, value)


async def load_external_mcp(config_path: str | Path = ".mcp.json") -> list[SearchTool]:
    p = Path(config_path)
    if not p.exists():
        logger.info("[mcp_loader] %s 不存在，跳过外部 MCP 加载", p)
        return []
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("[mcp_loader] 解析 %s 失败: %s", p, e)
        return []

    servers = cfg.get("mcpServers") or {}
    tools: list[SearchTool] = []

    for name, spec in servers.items():
        if name.startswith("_"):
            continue
        try:
            tool = _build_tool(name, spec)
        except Exception as e:
            logger.warning("[mcp_loader] 构建 %s 失败: %s", name, e)
            continue
        if tool is None:
            continue
        tools.append(tool)
        logger.info("[mcp_loader] ✓ 注册外部 MCP server: %s (source_type=%s)", name, tool.source_type)

    return tools


def _build_tool(name: str, spec: dict) -> SearchTool | None:
    if name == "brave-search":
        env_cfg = spec.get("env") or {}
        api_key = _resolve_env(env_cfg.get("BRAVE_API_KEY", ""))
        if not api_key:
            logger.info("[mcp_loader] brave-search 跳过：BRAVE_API_KEY 未设置")
            return None
        from tools.mcp_brave_tool import MCPBraveSearchTool

        cmd = [spec["command"], *spec.get("args", [])]
        return MCPBraveSearchTool(api_key=api_key, command=cmd, proxy=settings.https_proxy)

    # filesystem / 其他非 search 类型仅记录，不进 registry
    logger.info("[mcp_loader] 发现 %s（source_type=%s），暂不接入 registry", name, spec.get("source_type"))
    return None
