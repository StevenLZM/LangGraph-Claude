from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def test_same_request_id_replays_first_result_and_skips_second_llm_call():
    import api.main as main_module

    client = TestClient(app)
    request = {
        "user_id": "idempotency_user_001",
        "session_id": "idempotency_session_001",
        "request_id": "req_001",
        "message": "我的订单 ORD123456 到哪了？",
    }

    first = client.post("/chat", json=request)
    second = client.post("/chat", json=request)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["answer"] == second.json()["answer"]
    assert first.json()["request_id"] == "req_001"
    assert second.json()["request_id"] == "req_001"
    assert first.json()["request_status"] == "processed"
    assert second.json()["request_status"] == "replayed"
    assert main_module.customer_service_llm.calls == 1


def test_chat_request_status_endpoint_returns_stored_response():
    client = TestClient(app)
    request = {
        "user_id": "status_user_001",
        "session_id": "status_session_001",
        "request_id": "req_status_001",
        "message": "我的订单 ORD123456 到哪了？",
    }

    response = client.post("/chat", json=request)
    assert response.status_code == 200

    status = client.get(
        f"/chat/requests/{request['request_id']}",
        params={
            "user_id": request["user_id"],
            "session_id": request["session_id"],
        },
    )

    assert status.status_code == 200
    payload = status.json()
    assert payload["request_id"] == request["request_id"]
    assert payload["status"] == "succeeded"
    assert payload["response"]["answer"] == response.json()["answer"]
    assert payload["response"]["request_status"] == "processed"


def test_reusing_request_id_for_different_message_is_rejected():
    client = TestClient(app)
    request_id = "req_conflict_001"

    first = client.post(
        "/chat",
        json={
            "user_id": "conflict_user_001",
            "session_id": "conflict_session_001",
            "request_id": request_id,
            "message": "我的订单 ORD123456 到哪了？",
        },
    )
    second = client.post(
        "/chat",
        json={
            "user_id": "conflict_user_001",
            "session_id": "conflict_session_001",
            "request_id": request_id,
            "message": "我要退款",
        },
    )

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["detail"]["error"] == "idempotency_key_conflict"
