from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import api.main as main
from memory.long_term import UserMemoryManager
from memory.session_store import SessionStore
from messaging.events import (
    build_chat_completed_event,
    build_human_transfer_reminder_event,
    build_postprocess_requested_event,
)
from messaging.handlers import PostprocessEventHandler
from messaging.outbox import MessageOutboxStore
from messaging.publisher import RocketMQPublisher


class RecordingProducer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.messages = []

    def send(self, message):
        if self.fail:
            raise RuntimeError("rocketmq unavailable")
        self.messages.append(message)
        return SimpleNamespace(message_id=f"msg-{len(self.messages)}")


class RecordingPublisher:
    def __init__(self) -> None:
        self.published_batches = []

    def publish_many(self, events):
        self.published_batches.append(list(events))


def test_chat_completed_event_uses_versioned_topic_tag_and_session_key():
    event = build_chat_completed_event(
        session_id="session_001",
        user_id="user_001",
        question="我的订单 ORD123456 到哪了？",
        answer="订单 ORD123456 正在配送中。",
        token_used=12,
        response_time_ms=34,
        quality_score=88,
        needs_human_transfer=False,
        transfer_reason="",
        order_context={"order_id": "ORD123456"},
        trace_id="trace-001",
    )

    assert event.topic == "agent-customer-service-normal-v1"
    assert event.tag == "ChatCompleted"
    assert event.event_type == "customer_service.chat_completed"
    assert event.event_version == "v1"
    assert event.aggregate_id == "session_001"
    assert event.keys == ["session_001", "user_001"]
    assert event.payload["answer"] == "订单 ORD123456 正在配送中。"
    assert event.payload["order_context"] == {"order_id": "ORD123456"}


def test_postprocess_requested_event_uses_fifo_topic_and_message_group():
    event = build_postprocess_requested_event(
        session_id="session_001",
        user_id="user_001",
        question="我偏好顺丰配送",
        answer="已帮你记录偏好。",
        trace_id="trace-001",
    )

    assert event.topic == "agent-customer-service-fifo-v1"
    assert event.tag == "PostprocessRequested"
    assert event.event_type == "customer_service.postprocess_requested"
    assert event.message_group == "session_001"
    assert event.aggregate_id == "session_001"


def test_human_transfer_reminder_event_uses_delay_topic_and_delay_level():
    event = build_human_transfer_reminder_event(
        session_id="session_001",
        user_id="user_001",
        transfer_reason="用户要求人工或涉及投诉/法律问题",
        trace_id="trace-001",
    )

    assert event.topic == "agent-customer-service-delay-v1"
    assert event.tag == "HumanTransferReminderRequested"
    assert event.event_type == "customer_service.human_transfer_reminder_requested"
    assert event.delay_level == 3
    assert event.payload["transfer_reason"] == "用户要求人工或涉及投诉/法律问题"


def test_publisher_marks_outbox_published_or_enqueue_failed(tmp_path: Path):
    store = MessageOutboxStore(str(tmp_path / "outbox.db"))
    event = build_postprocess_requested_event(
        session_id="session_001",
        user_id="user_001",
        question="我偏好顺丰配送",
        answer="已帮你记录偏好。",
        trace_id="trace-001",
    )

    publisher = RocketMQPublisher(outbox_store=store, producer=RecordingProducer())
    publisher.publish(event)

    row = store.get_event(event.event_id)
    assert row is not None
    assert row["status"] == "published"
    assert row["message_id"] == "msg-1"

    failed_event = build_postprocess_requested_event(
        session_id="session_002",
        user_id="user_001",
        question="我偏好京东物流",
        answer="已帮你记录偏好。",
        trace_id="trace-002",
    )
    failing_publisher = RocketMQPublisher(outbox_store=store, producer=RecordingProducer(fail=True))

    failing_publisher.publish(failed_event)

    failed_row = store.get_event(failed_event.event_id)
    assert failed_row is not None
    assert failed_row["status"] == "enqueue_failed"
    assert "rocketmq unavailable" in failed_row["last_error"]


def test_chat_publishes_chat_completed_and_postprocess_events(monkeypatch):
    publisher = RecordingPublisher()
    monkeypatch.setattr(main, "message_publisher", publisher)

    client = TestClient(main.app)
    response = client.post(
        "/chat",
        json={
            "user_id": "mq_user",
            "session_id": "mq_session",
            "request_id": uuid.uuid4().hex,
            "message": "我的订单 ORD123456 到哪了？",
        },
    )

    assert response.status_code == 200
    assert publisher.published_batches
    event_types = [event.event_type for event in publisher.published_batches[-1]]
    assert event_types == [
        "customer_service.chat_completed",
        "customer_service.postprocess_requested",
    ]
    chat_event = publisher.published_batches[-1][0]
    assert chat_event.payload["session_id"] == "mq_session"
    assert chat_event.payload["answer"] == response.json()["answer"]


def test_postprocess_handler_updates_session_metadata_and_is_idempotent(tmp_path: Path):
    session_store = SessionStore(str(tmp_path / "sessions.db"))
    memory_manager = UserMemoryManager(str(tmp_path / "memory.db"))
    handler = PostprocessEventHandler(
        session_store=session_store,
        user_memory_manager=memory_manager,
    )
    event = build_postprocess_requested_event(
        session_id="session_001",
        user_id="user_001",
        question="我喜欢顺丰配送",
        answer="好的，后续我会优先参考顺丰配送偏好。",
        trace_id="trace-001",
    )

    first = handler.handle(event)
    second = handler.handle(event)

    session = session_store.get_public_session("session_001")
    memories = memory_manager.list_memories("user_001")
    assert first["status"] == "processed"
    assert second["status"] == "already_processed"
    assert session is not None
    assert session["quality_score"] is not None
    assert len(memories) == 1
    assert memories[0]["category"] == "delivery_preference"
