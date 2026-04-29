"""Internal Weather MCP server."""
from __future__ import annotations

import asyncio
import json

from mcp_servers import weather_data

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool
except Exception:  # pragma: no cover
    Server = None  # type: ignore[assignment]
    stdio_server = None  # type: ignore[assignment]
    TextContent = Tool = None  # type: ignore[assignment]

WEATHER_QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "description": "城市名称，支持中文，如北京、上海"},
        "units": {
            "type": "string",
            "enum": ["metric", "imperial"],
            "default": "metric",
            "description": "温度单位，metric=摄氏度，imperial=华氏度",
        },
    },
    "required": ["city"],
}


def build_app():
    if Server is None:
        raise RuntimeError("mcp SDK 未安装，请安装 requirements.txt 中的 mcp")

    app = Server("react-agent-weather")

    @app.list_tools()
    async def _list_tools() -> list[Tool]:  # type: ignore[valid-type]
        return [
            Tool(
                name="weather_query",
                description="查询内部天气 MCP 数据，适合回答城市天气和户外活动建议。",
                inputSchema=WEATHER_QUERY_SCHEMA,
            )
        ]

    @app.call_tool()
    async def _call_tool(name: str, arguments: dict) -> list[TextContent]:  # type: ignore[valid-type]
        if name != "weather_query":
            raise ValueError(f"未知 tool: {name}")
        data = await weather_data.get_weather(
            city=str(arguments["city"]),
            units=str(arguments.get("units", "metric")),
        )
        payload = {"data": data, "text": weather_data.format_weather(data)}
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    return app


async def main() -> None:
    app = build_app()
    async with stdio_server() as (read, write):  # type: ignore[misc]
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
