from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMResult:
    content: str
    model_used: str
    fallback_used: bool
    attempts: int
    circuit_state: str


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_time_seconds: int = 60,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_time_seconds = recovery_time_seconds
        self.last_failure_time: float | None = None
        self.state = "closed"
        self._clock = clock

    def is_open(self) -> bool:
        if self.state != "open":
            return False
        if self.last_failure_time is None:
            return True
        if self._clock() - self.last_failure_time >= self.recovery_time_seconds:
            self.state = "half-open"
            return False
        return True

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = self._clock()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

    def record_success(self) -> None:
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"


class OfflineStubLLM:
    async def ainvoke(self, messages: list[object]) -> str:
        return "offline_stub"


class ResilientLLM:
    def __init__(
        self,
        *,
        primary_client: Any | None = None,
        fallback_client: Any | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        max_attempts: int = 3,
        base_backoff_seconds: float = 0.1,
        max_backoff_seconds: float = 2.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.primary_client = primary_client or OfflineStubLLM()
        self.fallback_client = fallback_client or OfflineStubLLM()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.max_attempts = max(1, max_attempts)
        self.base_backoff_seconds = base_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self._sleep = sleep

    async def ainvoke(self, messages: list[object]) -> str:
        result = await self.ainvoke_with_metadata(messages)
        return result.content

    async def ainvoke_with_metadata(self, messages: list[object]) -> LLMResult:
        if self.circuit_breaker.is_open():
            fallback_content = await self._call_client(self.fallback_client, messages)
            return LLMResult(
                content=fallback_content,
                model_used="fallback",
                fallback_used=True,
                attempts=0,
                circuit_state=self.circuit_breaker.state,
            )

        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                primary_content = await self._call_client(self.primary_client, messages)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_attempts:
                    await self._sleep(self._backoff_for_attempt(attempt))
                    continue
                self.circuit_breaker.record_failure()
                break
            else:
                self.circuit_breaker.record_success()
                return LLMResult(
                    content=primary_content,
                    model_used="primary",
                    fallback_used=False,
                    attempts=attempt,
                    circuit_state=self.circuit_breaker.state,
                )

        try:
            fallback_content = await self._call_client(self.fallback_client, messages)
        except Exception:
            if last_error is not None:
                raise last_error
            raise
        return LLMResult(
            content=fallback_content,
            model_used="fallback",
            fallback_used=True,
            attempts=self.max_attempts,
            circuit_state=self.circuit_breaker.state,
        )

    async def _call_client(self, client: Any, messages: list[object]) -> str:
        response = await client.ainvoke(messages)
        return self._extract_content(response)

    def _backoff_for_attempt(self, attempt: int) -> float:
        delay = self.base_backoff_seconds * (2 ** (attempt - 1))
        return min(delay, self.max_backoff_seconds)

    @staticmethod
    def _extract_content(response: object) -> str:
        content = getattr(response, "content", response)
        return str(content)
