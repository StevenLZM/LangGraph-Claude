from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

NORMAL_TOPIC = "agent-customer-service-normal-v1"
FIFO_TOPIC = "agent-customer-service-fifo-v1"
DELAY_TOPIC = "agent-customer-service-delay-v1"
TRANSACTION_TOPIC = "agent-customer-service-tx-v1"
PRODUCER_NAME = "05_PRODUCT_AGENT"


@dataclass(frozen=True)
class RocketMQEvent:
    event_id: str
    event_type: str
    event_version: str
    producer: str
    occurred_at: str
    trace_id: str
    aggregate_id: str
    topic: str
    tag: str
    payload: dict[str, Any]
    keys: list[str] = field(default_factory=list)
    message_group: str = ""
    delay_level: int | None = None
    properties: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    def body(self) -> bytes:
        envelope = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "event_version": self.event_version,
            "producer": self.producer,
            "occurred_at": self.occurred_at,
            "trace_id": self.trace_id,
            "aggregate_id": self.aggregate_id,
            "payload": self.payload,
        }
        return json.dumps(envelope, ensure_ascii=False, sort_keys=True).encode("utf-8")

    @classmethod
    def from_json(cls, raw: str) -> "RocketMQEvent":
        payload = json.loads(raw)
        return cls(**payload)


@dataclass(frozen=True)
class RocketMQMessage:
    topic: str
    tag: str
    body: bytes
    keys: list[str]
    message_group: str = ""
    delay_level: int | None = None
    properties: dict[str, str] = field(default_factory=dict)


def build_chat_completed_event(
    *,
    session_id: str,
    user_id: str,
    question: str,
    answer: str,
    token_used: int,
    response_time_ms: int,
    quality_score: int | None,
    needs_human_transfer: bool,
    transfer_reason: str,
    order_context: dict[str, Any] | None,
    trace_id: str,
) -> RocketMQEvent:
    return RocketMQEvent(
        event_id=_new_event_id(),
        event_type="customer_service.chat_completed",
        event_version="v1",
        producer=PRODUCER_NAME,
        occurred_at=_now_iso(),
        trace_id=trace_id,
        aggregate_id=session_id,
        topic=NORMAL_TOPIC,
        tag="ChatCompleted",
        keys=[session_id, user_id],
        payload={
            "session_id": session_id,
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "token_used": token_used,
            "response_time_ms": response_time_ms,
            "quality_score": quality_score,
            "needs_human_transfer": needs_human_transfer,
            "transfer_reason": transfer_reason,
            "order_context": order_context,
        },
    )


def build_postprocess_requested_event(
    *,
    session_id: str,
    user_id: str,
    question: str,
    answer: str,
    trace_id: str,
) -> RocketMQEvent:
    return RocketMQEvent(
        event_id=_new_event_id(),
        event_type="customer_service.postprocess_requested",
        event_version="v1",
        producer=PRODUCER_NAME,
        occurred_at=_now_iso(),
        trace_id=trace_id,
        aggregate_id=session_id,
        topic=FIFO_TOPIC,
        tag="PostprocessRequested",
        keys=[session_id, user_id],
        message_group=session_id,
        payload={
            "session_id": session_id,
            "user_id": user_id,
            "question": question,
            "answer": answer,
        },
        properties={"message_group": session_id},
    )


def build_human_transfer_reminder_event(
    *,
    session_id: str,
    user_id: str,
    transfer_reason: str,
    trace_id: str,
    delay_level: int = 3,
) -> RocketMQEvent:
    return RocketMQEvent(
        event_id=_new_event_id(),
        event_type="customer_service.human_transfer_reminder_requested",
        event_version="v1",
        producer=PRODUCER_NAME,
        occurred_at=_now_iso(),
        trace_id=trace_id,
        aggregate_id=session_id,
        topic=DELAY_TOPIC,
        tag="HumanTransferReminderRequested",
        keys=[session_id, user_id],
        delay_level=delay_level,
        payload={
            "session_id": session_id,
            "user_id": user_id,
            "transfer_reason": transfer_reason,
            "delay_level": delay_level,
        },
    )


def event_to_message(event: RocketMQEvent) -> RocketMQMessage:
    properties = dict(event.properties)
    properties.update(
        {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "event_version": event.event_version,
            "producer": event.producer,
            "trace_id": event.trace_id,
        }
    )
    if event.message_group:
        properties["message_group"] = event.message_group
    return RocketMQMessage(
        topic=event.topic,
        tag=event.tag,
        body=event.body(),
        keys=list(event.keys),
        message_group=event.message_group,
        delay_level=event.delay_level,
        properties=properties,
    )


def _new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
