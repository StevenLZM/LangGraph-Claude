from __future__ import annotations

import pytest

from config import llm as llm_module


def _base_url(llm) -> str:
    return str(llm.openai_api_base).rstrip("/")


def test_get_llm_uses_deepseek_strong_model(monkeypatch):
    llm_module.get_llm.cache_clear()
    monkeypatch.setattr(llm_module.settings, "deepseek_api_key", "sk-test")
    monkeypatch.setattr(llm_module.settings, "deepseek_base_url", "https://api.deepseek.com")
    monkeypatch.setattr(llm_module.settings, "deepseek_max_model", "deepseek-v4-pro")
    monkeypatch.setattr(llm_module.settings, "deepseek_light_model", "deepseek-v4-flash")

    llm = llm_module.get_llm("max", temperature=0.1)

    assert llm.model_name == "deepseek-v4-pro"
    assert _base_url(llm) == "https://api.deepseek.com"
    assert llm.temperature == 0.1


def test_get_llm_uses_deepseek_light_model(monkeypatch):
    llm_module.get_llm.cache_clear()
    monkeypatch.setattr(llm_module.settings, "deepseek_api_key", "sk-test")
    monkeypatch.setattr(llm_module.settings, "deepseek_base_url", "https://api.deepseek.com")
    monkeypatch.setattr(llm_module.settings, "deepseek_max_model", "deepseek-v4-pro")
    monkeypatch.setattr(llm_module.settings, "deepseek_light_model", "deepseek-v4-flash")

    llm = llm_module.get_llm("turbo")

    assert llm.model_name == "deepseek-v4-flash"
    assert _base_url(llm) == "https://api.deepseek.com"


def test_get_llm_requires_deepseek_key(monkeypatch):
    llm_module.get_llm.cache_clear()
    monkeypatch.setattr(llm_module.settings, "deepseek_api_key", "")

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        llm_module.get_llm("max")
