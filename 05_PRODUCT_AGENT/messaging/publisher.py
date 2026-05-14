from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from messaging.events import RocketMQEvent, RocketMQMessage, event_to_message
from messaging.outbox import MessageOutboxStore, PostgresMessageOutboxStore, build_message_outbox_store


class MessageProducer(Protocol):
    def send(self, message: RocketMQMessage) -> Any:
        ...


class NoopMessagePublisher:
    enabled = False

    def publish(self, event: RocketMQEvent) -> None:
        del event

    def publish_many(self, events: list[RocketMQEvent]) -> None:
        del events


class RocketMQPublisher:
    enabled = True

    def __init__(
        self,
        *,
        outbox_store: MessageOutboxStore | PostgresMessageOutboxStore,
        producer: MessageProducer,
    ) -> None:
        self.outbox_store = outbox_store
        self.producer = producer

    def publish_many(self, events: list[RocketMQEvent]) -> None:
        for event in events:
            self.publish(event)

    def publish(self, event: RocketMQEvent) -> None:
        self.outbox_store.enqueue(event)
        try:
            result = self.producer.send(event_to_message(event))
        except Exception as exc:
            self.outbox_store.mark_failed(event.event_id, error=str(exc))
            return
        self.outbox_store.mark_published(event.event_id, message_id=_extract_message_id(result))


@dataclass
class RocketMQSDKProducer:
    endpoint: str
    producer_group: str
    topics: tuple[str, ...]
    access_key: str = ""
    secret_key: str = ""
    _producer: Any | None = field(default=None, init=False)
    _message_cls: Any | None = field(default=None, init=False)

    def send(self, message: RocketMQMessage) -> Any:
        self._ensure_started()
        sdk_message = self._message_cls()
        sdk_message.topic = message.topic
        sdk_message.body = message.body
        sdk_message.tag = message.tag
        sdk_message.keys = ";".join(message.keys)
        if message.message_group:
            sdk_message.message_group = message.message_group
        for key, value in message.properties.items():
            sdk_message.add_property(key, value)
        if message.delay_level is not None:
            sdk_message.delivery_timestamp = int(time.time() * 1000) + message.delay_level * 60_000
            sdk_message.add_property("delay_level", str(message.delay_level))
        return self._producer.send(sdk_message)

    def _ensure_started(self) -> None:
        if self._producer is not None and self._message_cls is not None:
            return
        try:
            from rocketmq.v5.client import ClientConfiguration, Credentials
            from rocketmq.v5.model import Message
            from rocketmq.v5.producer import Producer
        except ImportError as exc:  # pragma: no cover - depends on optional SDK install
            raise RuntimeError("rocketmq-python-client is required when ROCKETMQ_ENABLED=true") from exc
        credentials = Credentials(self.access_key, self.secret_key) if self.access_key or self.secret_key else Credentials()
        self._message_cls = Message
        self._producer = Producer(ClientConfiguration(self.endpoint, credentials), self.topics)
        self._producer.startup()


def build_message_publisher(
    settings: Any,
    *,
    outbox_store: MessageOutboxStore | PostgresMessageOutboxStore | None = None,
) -> RocketMQPublisher | NoopMessagePublisher:
    if not getattr(settings, "rocketmq_enabled", False):
        return NoopMessagePublisher()
    outbox = outbox_store or build_message_outbox_store(settings)
    producer = RocketMQSDKProducer(
        endpoint=getattr(settings, "rocketmq_endpoint", "localhost:9876"),
        producer_group=getattr(settings, "rocketmq_producer_group", "PID_05_PRODUCT_AGENT"),
        topics=(
            getattr(settings, "rocketmq_normal_topic", "agent-customer-service-normal-v1"),
            getattr(settings, "rocketmq_fifo_topic", "agent-customer-service-fifo-v1"),
            getattr(settings, "rocketmq_delay_topic", "agent-customer-service-delay-v1"),
            getattr(settings, "rocketmq_transaction_topic", "agent-customer-service-tx-v1"),
        ),
        access_key=getattr(settings, "rocketmq_access_key", ""),
        secret_key=getattr(settings, "rocketmq_secret_key", ""),
    )
    return RocketMQPublisher(outbox_store=outbox, producer=producer)


def _extract_message_id(result: Any) -> str:
    for attr in ("message_id", "msg_id", "id"):
        value = getattr(result, attr, "")
        if value:
            return str(value)
    return str(result or "")
