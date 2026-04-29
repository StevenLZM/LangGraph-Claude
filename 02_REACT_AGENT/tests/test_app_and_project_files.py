from __future__ import annotations

import json
from pathlib import Path

from agent.events import AgentEvent
from app import format_event_for_display

ROOT = Path(__file__).resolve().parent.parent


def test_format_event_for_display_handles_tool_call():
    event = AgentEvent(type="tool_call", title="调用工具: calculator", tool="calculator", tool_input={"expression": "1+1"})

    display = format_event_for_display(event)

    assert display["label"] == "调用工具: calculator"
    assert display["language"] == "json"
    assert "expression" in display["body"]


def test_project_files_document_deepseek_and_internal_weather_mcp():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    mcp_config = json.loads((ROOT / ".mcp.json").read_text(encoding="utf-8"))

    assert "langchain-openai" in requirements
    assert "mcp" in requirements
    assert "DEEPSEEK_API_KEY" in env_example
    assert mcp_config["mcpServers"]["weather"]["args"] == ["-m", "mcp_servers.weather_server"]
