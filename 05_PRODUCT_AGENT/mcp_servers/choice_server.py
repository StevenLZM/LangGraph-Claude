"""Choice MCP server for customer-service option flows."""
from __future__ import annotations

import asyncio
import json

from agent.choices import create_choice_set, resolve_choice

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool
except Exception:  # pragma: no cover - optional runtime dependency
    Server = None  # type: ignore[assignment]
    stdio_server = None  # type: ignore[assignment]
    TextContent = Tool = None  # type: ignore[assignment]


CREATE_CHOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "scenario": {"type": "string"},
        "title": {"type": "string"},
        "prompt": {"type": "string"},
        "source_tool": {"type": "string"},
        "options": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "value": {"type": "string"},
                    "description": {"type": "string"},
                    "requires_confirmation": {"type": "boolean"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "label"],
            },
        },
        "payload": {"type": "object"},
        "ttl_seconds": {"type": "integer"},
    },
    "required": ["scenario", "title", "prompt", "source_tool", "options"],
}

RESOLVE_CHOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "choice_set": {"type": "object"},
        "user_input": {"type": "string"},
    },
    "required": ["choice_set", "user_input"],
}

CLEAR_CHOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "choice_set_id": {"type": "string"},
        "reason": {"type": "string"},
    },
}


def build_app():
    if Server is None:
        raise RuntimeError("mcp SDK 未安装，请安装 05_PRODUCT_AGENT requirements.txt 中的 mcp")

    app = Server("customer-service-choice")

    @app.list_tools()
    async def _list_tools() -> list[Tool]:  # type: ignore[valid-type]
        return [
            Tool(
                name="create_choice_set",
                description="为客服对话创建一组可追踪、可确认的用户选项。",
                inputSchema=CREATE_CHOICE_SCHEMA,
            ),
            Tool(
                name="resolve_choice",
                description="解析用户对待确认选项的回复，例如选第2个、确认退款、转人工。",
                inputSchema=RESOLVE_CHOICE_SCHEMA,
            ),
            Tool(
                name="clear_choice_set",
                description="清理已完成、已过期或转人工的待确认选项。",
                inputSchema=CLEAR_CHOICE_SCHEMA,
            ),
        ]

    @app.call_tool()
    async def _call_tool(name: str, arguments: dict) -> list[TextContent]:  # type: ignore[valid-type]
        if name == "create_choice_set":
            payload = create_choice_set(
                scenario=str(arguments["scenario"]),
                title=str(arguments["title"]),
                prompt=str(arguments["prompt"]),
                source_tool=str(arguments["source_tool"]),
                options=list(arguments["options"]),
                payload=dict(arguments.get("payload") or {}),
                ttl_seconds=int(arguments.get("ttl_seconds") or 900),
            )
        elif name == "resolve_choice":
            payload = resolve_choice(
                dict(arguments["choice_set"]),
                str(arguments["user_input"]),
            )
        elif name == "clear_choice_set":
            payload = {
                "cleared": True,
                "choice_set_id": str(arguments.get("choice_set_id", "")),
                "reason": str(arguments.get("reason", "")),
            }
        else:
            raise ValueError(f"未知 tool: {name}")
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    return app


async def main() -> None:
    app = build_app()
    async with stdio_server() as (read, write):  # type: ignore[misc]
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
