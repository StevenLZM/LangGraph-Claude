from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def _client() -> TestClient:
    return TestClient(app)


def test_admin_sessions_lists_quality_token_and_transfer_fields():
    client = _client()
    session_id = "admin_session_transfer_001"

    chat_response = client.post(
        "/chat",
        json={
            "user_id": "admin_user_001",
            "session_id": session_id,
            "message": "我要投诉，给我转人工",
        },
    )
    assert chat_response.status_code == 200

    response = client.get("/admin/sessions")

    assert response.status_code == 200
    sessions = response.json()["sessions"]
    matching = [item for item in sessions if item["session_id"] == session_id]
    assert matching
    assert matching[0]["user_id"] == "admin_user_001"
    assert matching[0]["needs_human_transfer"] is True
    assert matching[0]["transfer_reason"] == "用户要求人工或涉及投诉/法律问题"
    assert isinstance(matching[0]["quality_score"], int)
    assert isinstance(matching[0]["token_used"], int)


def test_admin_user_memories_returns_saved_memory_rows():
    client = _client()
    user_id = "admin_memory_user_001"

    client.post(
        "/chat",
        json={
            "user_id": user_id,
            "session_id": "admin_memory_session_001",
            "message": "我喜欢顺丰配送，以后发货优先顺丰",
        },
    )
    response = client.get(f"/admin/users/{user_id}/memories")

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == user_id
    assert payload["memories"]
    assert payload["memories"][0]["category"] == "delivery_preference"
    assert "顺丰" in payload["memories"][0]["content"]


def test_admin_transfer_stats_summarizes_sessions():
    client = _client()
    client.post(
        "/chat",
        json={
            "user_id": "admin_stats_user_001",
            "session_id": "admin_stats_session_001",
            "message": "我要投诉，给我转人工",
        },
    )

    response = client.get("/admin/stats/transfers")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_sessions"] >= 1
    assert payload["human_transfer_count"] >= 1
    assert payload["transfer_reasons"]["用户要求人工或涉及投诉/法律问题"] >= 1
    assert payload["token_total"] >= 1
    assert payload["average_quality_score"] > 0
