"""Internal weather data used by the local MCP server.

This intentionally avoids external API calls so the teaching demo and tests are
stable without weather credentials or network access.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

_WEATHER_BY_CITY: dict[str, dict[str, Any]] = {
    "北京": {
        "city": "北京",
        "temp_c": 18,
        "condition": "晴",
        "humidity": 45,
        "wind_speed": 2.3,
        "advice": "空气清爽，适合跑步，建议避开正午强光。",
    },
    "上海": {
        "city": "上海",
        "temp_c": 22,
        "condition": "多云",
        "humidity": 63,
        "wind_speed": 3.1,
        "advice": "温度舒适，可以户外活动，注意湿度略高。",
    },
    "深圳": {
        "city": "深圳",
        "temp_c": 27,
        "condition": "阵雨",
        "humidity": 78,
        "wind_speed": 4.0,
        "advice": "可能有雨，户外运动建议携带雨具或改在室内。",
    },
    "广州": {
        "city": "广州",
        "temp_c": 26,
        "condition": "阴",
        "humidity": 72,
        "wind_speed": 2.8,
        "advice": "湿度较高，运动强度宜适中。",
    },
}


async def get_weather(city: str, units: str = "metric") -> dict[str, Any]:
    normalized = city.strip()
    data = deepcopy(_WEATHER_BY_CITY.get(normalized))
    if data is None:
        data = {
            "city": normalized or "未知城市",
            "temp_c": 20,
            "condition": "晴",
            "humidity": 50,
            "wind_speed": 2.0,
            "advice": "内部演示数据未覆盖该城市，返回默认温和天气。",
        }
    if units == "imperial":
        data["temp_f"] = round(data["temp_c"] * 9 / 5 + 32, 1)
    data["units"] = units
    data["source"] = "internal-mcp"
    return data


def format_weather(data: dict[str, Any]) -> str:
    temp = f"{data['temp_c']}°C"
    if "temp_f" in data:
        temp = f"{temp} / {data['temp_f']}°F"
    return (
        f"城市: {data['city']}\n"
        f"温度: {temp}\n"
        f"天气: {data['condition']}\n"
        f"湿度: {data['humidity']}%\n"
        f"风速: {data['wind_speed']}m/s\n"
        f"建议: {data['advice']}\n"
        f"数据源: {data['source']}"
    )
