from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def test_root_serves_customer_service_workspace():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "智能客服工作台" in response.text
    assert "/chat" in response.text
    assert "订单上下文" in response.text
    assert "用户记忆" in response.text
    assert "降级状态" in response.text
    assert "LLM 状态" in response.text
    assert "工具路径" in response.text
    assert "处理摘要" in response.text
    assert "清除记忆" in response.text
