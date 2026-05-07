from __future__ import annotations

import asyncio
import importlib

import pytest


def _llm_classes():
    try:
        module = importlib.import_module("llm.resilient_llm")
        return module.CircuitBreaker, module.ResilientLLM
    except AttributeError as exc:
        pytest.fail(f"ResilientLLM dependencies are not implemented: {exc}")


class FakeLLM:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    async def ainvoke(self, messages: list[object]) -> object:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


async def _no_sleep(_: float) -> None:
    return None


def test_primary_failure_uses_fallback_model():
    _, ResilientLLM = _llm_classes()
    primary = FakeLLM([RuntimeError("primary down")])
    fallback = FakeLLM(["fallback answer"])
    llm = ResilientLLM(
        primary_client=primary,
        fallback_client=fallback,
        max_attempts=1,
        sleep=_no_sleep,
    )

    result = asyncio.run(llm.ainvoke_with_metadata(["hello"]))

    assert result.content == "fallback answer"
    assert result.model_used == "fallback"
    assert result.fallback_used is True
    assert primary.calls == 1
    assert fallback.calls == 1


def test_primary_retries_with_exponential_backoff_before_fallback():
    _, ResilientLLM = _llm_classes()
    primary = FakeLLM([RuntimeError("temporary"), "primary answer"])
    fallback = FakeLLM(["fallback answer"])
    llm = ResilientLLM(
        primary_client=primary,
        fallback_client=fallback,
        max_attempts=3,
        sleep=_no_sleep,
    )

    result = asyncio.run(llm.ainvoke_with_metadata(["hello"]))

    assert result.content == "primary answer"
    assert result.model_used == "primary"
    assert result.fallback_used is False
    assert result.attempts == 2
    assert primary.calls == 2
    assert fallback.calls == 0


def test_circuit_breaker_routes_to_fallback_after_consecutive_failures():
    CircuitBreaker, ResilientLLM = _llm_classes()
    primary = FakeLLM([RuntimeError("first"), RuntimeError("second"), "primary should not run"])
    fallback = FakeLLM(["fallback one", "fallback two", "fallback three"])
    breaker = CircuitBreaker(failure_threshold=2, recovery_time_seconds=60)
    llm = ResilientLLM(
        primary_client=primary,
        fallback_client=fallback,
        circuit_breaker=breaker,
        max_attempts=1,
        sleep=_no_sleep,
    )

    first = asyncio.run(llm.ainvoke_with_metadata(["hello"]))
    second = asyncio.run(llm.ainvoke_with_metadata(["hello"]))
    third = asyncio.run(llm.ainvoke_with_metadata(["hello"]))

    assert first.content == "fallback one"
    assert second.content == "fallback two"
    assert third.content == "fallback three"
    assert breaker.state == "open"
    assert primary.calls == 2
    assert fallback.calls == 3
