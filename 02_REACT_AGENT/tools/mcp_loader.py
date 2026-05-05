"""Load LangChain tools from MCP server definitions in .mcp.json."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, Tool

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    command: str
    args: list[str]
    cwd: Path
    env: dict[str, str]
    description: str


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("同步 MCP 工具调用不能运行在已有 asyncio event loop 中，请使用异步调用路径。")


def load_mcp_server_configs(config_path: str | Path | None = None) -> list[MCPServerConfig]:
    path = Path(config_path or settings.mcp_config_path).expanduser()
    if not path.exists():
        return []
    path = path.resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    servers = raw.get("mcpServers", {})
    if not isinstance(servers, dict):
        raise ValueError(".mcp.json 中的 mcpServers 必须是对象")

    configs: list[MCPServerConfig] = []
    for name, server in servers.items():
        if not isinstance(server, dict):
            raise ValueError(f"MCP server {name} 配置必须是对象")
        command = server.get("command")
        if not isinstance(command, str) or not command:
            raise ValueError(f"MCP server {name} 缺少 command")
        args = server.get("args", [])
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            raise ValueError(f"MCP server {name} 的 args 必须是字符串列表")

        cwd_value = server.get("cwd", str(path.parent))
        if not isinstance(cwd_value, str):
            raise ValueError(f"MCP server {name} 的 cwd 必须是字符串")
        cwd = Path(cwd_value).expanduser()
        if not cwd.is_absolute():
            cwd = path.parent / cwd

        env_value = server.get("env", {})
        if not isinstance(env_value, dict):
            raise ValueError(f"MCP server {name} 的 env 必须是对象")

        configs.append(
            MCPServerConfig(
                name=str(name),
                command=command,
                args=args,
                cwd=cwd.resolve(),
                env={str(key): str(value) for key, value in env_value.items()},
                description=str(server.get("description", "")),
            )
        )
    return configs


def _server_params(config: MCPServerConfig) -> StdioServerParameters:
    env = {**os.environ, **config.env} if config.env else None
    return StdioServerParameters(command=config.command, args=config.args, cwd=config.cwd, env=env)


async def _list_tools(config: MCPServerConfig) -> list[Tool]:
    async with stdio_client(_server_params(config)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return list(result.tools)


async def _call_tool(config: MCPServerConfig, tool_name: str, arguments: dict[str, Any]) -> str:
    async with stdio_client(_server_params(config)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return _tool_result_to_text(result)


def _tool_result_to_text(result: CallToolResult) -> str:
    chunks: list[str] = []
    for item in result.content:
        if getattr(item, "type", None) == "text":
            text = str(getattr(item, "text", ""))
            chunks.append(_extract_text_payload(text))
        elif hasattr(item, "model_dump_json"):
            chunks.append(item.model_dump_json())
        else:
            chunks.append(str(item))

    if not chunks and result.structuredContent:
        chunks.append(json.dumps(result.structuredContent, ensure_ascii=False))

    output = "\n".join(chunk for chunk in chunks if chunk)
    if result.isError:
        return f"MCP 工具执行失败: {output}"
    return output


def _extract_text_payload(text: str) -> str:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(payload, dict) and isinstance(payload.get("text"), str):
        return payload["text"]
    return text


def _exposed_tool_name(server_name: str, tool_name: str, used_names: set[str]) -> str:
    if tool_name not in used_names:
        return tool_name
    return f"{server_name}__{tool_name}"


def _to_langchain_tool(config: MCPServerConfig, tool_def: Tool, exposed_name: str) -> StructuredTool:
    async def _ainvoke(**kwargs: Any) -> str:
        return await _call_tool(config, tool_def.name, kwargs)

    def _invoke(**kwargs: Any) -> str:
        return _run_async(_call_tool(config, tool_def.name, kwargs))

    description = tool_def.description or config.description or f"MCP tool {tool_def.name}"
    if exposed_name != tool_def.name:
        description = f"{description}\nMCP 原始工具名: {tool_def.name}"

    return StructuredTool.from_function(
        func=_invoke,
        coroutine=_ainvoke,
        name=exposed_name,
        description=description,
        args_schema=tool_def.inputSchema,
        metadata={"mcp_server": config.name, "mcp_tool": tool_def.name},
    )


def load_mcp_tools(
    config_path: str | Path | None = None,
    *,
    existing_names: set[str] | None = None,
    raise_on_error: bool = False,
) -> list[StructuredTool]:
    tools: list[StructuredTool] = []
    used_names = set(existing_names or set())
    for config in load_mcp_server_configs(config_path):
        try:
            server_tools = _run_async(_list_tools(config))
        except Exception as exc:
            if raise_on_error:
                raise RuntimeError(f"加载 MCP server {config.name} 失败: {exc}") from exc
            logger.warning("加载 MCP server %s 失败: %s", config.name, exc)
            continue

        for tool_def in server_tools:
            exposed_name = _exposed_tool_name(config.name, tool_def.name, used_names)
            used_names.add(exposed_name)
            tools.append(_to_langchain_tool(config, tool_def, exposed_name))
    return tools
