from __future__ import annotations

from config import LLMConfig
from rag import embedder


def test_deepseek_is_preferred_chat_provider(monkeypatch):
    monkeypatch.setattr(LLMConfig, "DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setattr(LLMConfig, "DASHSCOPE_API_KEY", "sk-dashscope")
    monkeypatch.setattr(LLMConfig, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(LLMConfig, "OPENAI_API_KEY", "")

    assert LLMConfig.provider() == "deepseek"


def test_embeddings_still_prefer_dashscope_when_deepseek_chat_is_configured(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(LLMConfig, "DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setattr(LLMConfig, "DASHSCOPE_API_KEY", "sk-dashscope")
    monkeypatch.setattr(LLMConfig, "OPENAI_API_KEY", "")
    monkeypatch.setattr(embedder, "_get_dashscope_embeddings", lambda: sentinel)

    assert embedder.get_embeddings() is sentinel
