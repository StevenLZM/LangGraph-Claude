from __future__ import annotations

from pathlib import Path

from agent.prompts import build_plan_system_prompt, build_react_system_prompt
from tools.mcp_loader import load_mcp_tools
from tools.registry import get_tools

ROOT = Path(__file__).resolve().parent.parent


def _tool_names(tools) -> set[str]:
    return {tool.name for tool in tools}


def test_load_mcp_tools_reads_config_and_invokes_weather_tool():
    tools = load_mcp_tools(ROOT / ".mcp.json")

    names = _tool_names(tools)
    assert "weather_query" in names

    weather = next(tool for tool in tools if tool.name == "weather_query")
    result = weather.invoke({"city": "北京"})

    assert "北京" in result
    assert "温度" in result
    assert "数据源: internal-mcp" in result


def test_load_mcp_tools_returns_empty_list_for_missing_config(tmp_path):
    tools = load_mcp_tools(tmp_path / ".mcp.json")

    assert tools == []


def test_default_registry_combines_builtin_and_dynamic_mcp_tools():
    tools = get_tools()

    names = _tool_names(tools)
    assert "calculator" in names
    assert "weather_query" in names


def test_agent_prompts_render_current_tool_registry():
    tools = get_tools()

    react_prompt = build_react_system_prompt(tools)
    plan_prompt = build_plan_system_prompt(tools)

    assert "- calculator:" in react_prompt
    assert "- weather_query:" in react_prompt
    assert "当前可用工具名：" in plan_prompt
    assert "weather_query" in plan_prompt
