from __future__ import annotations

import asyncio
from types import SimpleNamespace

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
        "anthropic_api_key": "",
        "openai_api_key": "",
        "openai_base_url": "",
        "primary_model": "gpt-4o-mini",
        "fallback_model": "gpt-4o-mini",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_offline_mode_builds_stub_llm_without_startup_error():
    setup = build_customer_service_llm(_settings())

    result = asyncio.run(setup.llm.ainvoke_with_metadata(["hello"]))

    assert setup.startup_error == ""
    assert result.content == "offline_stub"


def test_hybrid_mode_without_matching_key_returns_explainable_startup_error():
    setup = build_customer_service_llm(_settings(llm_mode="hybrid"))

    result = asyncio.run(setup.llm.ainvoke_with_metadata(["hello"]))

    assert "OPENAI_API_KEY" in setup.startup_error
    assert result.content == "offline_stub"


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
