from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from llm.factory import build_customer_service_llm


class FakeChatOpenAI:
    instances: list["FakeChatOpenAI"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.calls = 0
        FakeChatOpenAI.instances.append(self)

    async def ainvoke(self, messages: list[object]) -> str:
        self.calls += 1
        return "openai answer"


def _settings(**overrides):
    values = {
        "llm_mode": "offline_stub",
        "deepseek_api_key": "",
        "deepseek_base_url": "https://api.deepseek.com",
        "deepseek_max_model": "deepseek-v4-pro",
        "deepseek_light_model": "deepseek-v4-flash",
        "anthropic_api_key": "",
        "openai_api_key": "",
        "openai_base_url": "",
        "primary_model": "gpt-4o-mini",
        "fallback_model": "gpt-4o-mini",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_offline_mode_is_rejected_because_runtime_requires_real_llm():
    setup = build_customer_service_llm(_settings())

    assert "offline_stub" in setup.startup_error
    with pytest.raises(RuntimeError, match="real LLM"):
        asyncio.run(setup.llm.ainvoke_with_metadata(["hello"]))


def test_hybrid_mode_without_matching_key_returns_explainable_startup_error():
    setup = build_customer_service_llm(_settings(llm_mode="hybrid"))

    assert "OPENAI_API_KEY" in setup.startup_error
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        asyncio.run(setup.llm.ainvoke_with_metadata(["hello"]))


def test_hybrid_mode_with_openai_key_builds_openai_primary_client(monkeypatch):
    import llm.factory as factory

    FakeChatOpenAI.instances = []
    monkeypatch.setattr(factory, "_load_chat_openai", lambda: FakeChatOpenAI)
    setup = build_customer_service_llm(
        _settings(
            llm_mode="hybrid",
            openai_api_key="sk-test",
            openai_base_url="https://api.example.com/v1",
            primary_model="gpt-4o-mini",
            fallback_model="gpt-4o-mini",
        )
    )

    result = asyncio.run(setup.llm.ainvoke_with_metadata(["hello"]))

    assert setup.startup_error == ""
    assert result.content == "openai answer"
    assert result.model_used == "primary"
    assert FakeChatOpenAI.instances[0].kwargs["model"] == "gpt-4o-mini"
    assert FakeChatOpenAI.instances[0].kwargs["api_key"] == "sk-test"
    assert FakeChatOpenAI.instances[0].kwargs["base_url"] == "https://api.example.com/v1"


def test_deepseek_mode_builds_openai_compatible_primary_and_light_fallback(monkeypatch):
    import llm.factory as factory

    FakeChatOpenAI.instances = []
    monkeypatch.setattr(factory, "_load_chat_openai", lambda: FakeChatOpenAI)

    setup = build_customer_service_llm(
        _settings(
            llm_mode="deepseek",
            deepseek_api_key="sk-deepseek-test",
            deepseek_base_url="https://api.deepseek.com",
            deepseek_max_model="deepseek-v4-pro",
            deepseek_light_model="deepseek-v4-flash",
        )
    )

    result = asyncio.run(setup.llm.ainvoke_with_metadata(["hello"]))

    assert setup.startup_error == ""
    assert result.content == "openai answer"
    assert result.model_used == "primary"
    assert FakeChatOpenAI.instances[0].kwargs == {
        "model": "deepseek-v4-pro",
        "api_key": "sk-deepseek-test",
        "base_url": "https://api.deepseek.com",
        "temperature": 0.2,
        "timeout": 240,
        "max_retries": 2,
    }
    assert FakeChatOpenAI.instances[1].kwargs["model"] == "deepseek-v4-flash"


def test_deepseek_mode_requires_deepseek_key():
    setup = build_customer_service_llm(_settings(llm_mode="deepseek"))

    assert "DEEPSEEK_API_KEY" in setup.startup_error
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        asyncio.run(setup.llm.ainvoke_with_metadata(["hello"]))
