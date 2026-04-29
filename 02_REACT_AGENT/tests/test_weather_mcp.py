from __future__ import annotations

import pytest

from mcp_servers import weather_data
from mcp_servers.weather_server import build_app


@pytest.mark.asyncio
async def test_internal_weather_handler_returns_mock_weather():
    data = await weather_data.get_weather("上海")

    assert data["city"] == "上海"
    assert data["source"] == "internal-mcp"
    assert "condition" in data


def test_weather_mcp_server_can_be_built():
    app = build_app()

    assert app.name == "react-agent-weather"
