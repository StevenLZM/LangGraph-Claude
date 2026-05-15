from __future__ import annotations

import itertools
import random
from uuid import uuid4

from locust import HttpUser, between, task


SCENARIOS = [
    "我的订单 ORD123456 到哪了？",
    "帮我查一下物流 ORD123456",
    "AirBuds Pro 2 还有库存吗？",
    "我要给订单 ORD123456 退款",
    "我确认退款 ORD123456",
    "我要投诉，给我转人工",
]


class CustomerServiceUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        self.user_id = f"load_user_{uuid4().hex[:8]}"
        self.session_counter = itertools.count()

    @task(8)
    def chat_core_scenarios(self) -> None:
        index = next(self.session_counter)
        message = random.choice(SCENARIOS)
        self.client.post(
            "/chat",
            name="POST /chat",
            json={
                "user_id": self.user_id,
                "session_id": f"{self.user_id}_session_{index}",
                "request_id": f"{self.user_id}_req_{index}",
                "message": message,
            },
        )

    @task(1)
    def health_check(self) -> None:
        self.client.get("/health", name="GET /health")

    @task(1)
    def scrape_metrics(self) -> None:
        self.client.get("/metrics", name="GET /metrics")
