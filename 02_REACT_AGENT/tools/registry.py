"""Unified tool registry for built-in and MCP-provided tools."""
from __future__ import annotations

from typing import Any

from tools.builtin import get_builtin_tools
from tools.mcp_loader import load_mcp_tools


def get_tools() -> list[Any]:
    builtin_tools = get_builtin_tools()
    builtin_names = {tool.name for tool in builtin_tools}
    return [*builtin_tools, *load_mcp_tools(existing_names=builtin_names)]
