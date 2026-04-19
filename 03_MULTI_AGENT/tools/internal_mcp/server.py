"""Internal MCP Server 入口 —— 详见 ENGINEERING.md §7.4。

运行：python -m tools.internal_mcp.server  （stdio 模式，供 Claude Desktop/Cursor 配置）
"""
from __future__ import annotations

import asyncio
import json

from tools.internal_mcp import handlers, schemas

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool
except Exception:  # pragma: no cover  —— 允许未安装 mcp 时仍能 import 上层
    Server = None  # type: ignore
    stdio_server = None  # type: ignore
    TextContent = Tool = None  # type: ignore


def build_app():
    if Server is None:
        raise RuntimeError("mcp SDK 未安装，请 `pip install mcp`")

    app = Server("insightloop-internal")

    @app.list_tools()
    async def _list() -> list[Tool]:  # type: ignore[valid-type]
        return [
            Tool(name="kb_search", description="本地知识库混合检索", inputSchema=schemas.KB_SEARCH),
            Tool(name="list_reports", description="列出历史研究报告", inputSchema=schemas.LIST_REPORTS),
            Tool(name="read_report", description="读取某次研究报告", inputSchema=schemas.READ_REPORT),
            Tool(name="list_evidence", description="查询某会话已收集证据", inputSchema=schemas.LIST_EVIDENCE),
            Tool(name="trigger_research", description="异步触发新研究", inputSchema=schemas.TRIGGER_RESEARCH),
        ]

    @app.call_tool()
    async def _call(name: str, arguments: dict) -> list[TextContent]:  # type: ignore[valid-type]
        match name:
            case "kb_search":
                data = await handlers.kb_search(**arguments)
            case "list_reports":
                data = await handlers.list_reports(**arguments)
            case "read_report":
                data = await handlers.read_report(**arguments)
            case "list_evidence":
                data = await handlers.list_evidence(**arguments)
            case "trigger_research":
                data = await handlers.trigger_research(**arguments)
            case _:
                raise ValueError(f"未知 tool: {name}")
        return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False))]

    return app


async def main() -> None:
    app = build_app()
    async with stdio_server() as (read, write):  # type: ignore[misc]
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
