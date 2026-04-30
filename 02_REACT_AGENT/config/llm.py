"""DeepSeek LLM factory.

DeepSeek exposes an OpenAI-compatible chat endpoint, so the project keeps
LangChain's ChatOpenAI integration while changing only provider settings.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langchain_openai import ChatOpenAI

from config.settings import settings

Tier = Literal["max", "turbo"]


@lru_cache(maxsize=8)
def get_llm(tier: Tier = "max", *, temperature: float = 0.2, streaming: bool = False) -> ChatOpenAI:
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 未配置。请在 02_REACT_AGENT/.env 中填入。")

    model = settings.deepseek_max_model if tier == "max" else settings.deepseek_light_model
    return ChatOpenAI(
        model=model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=temperature,
        streaming=streaming,
        extra_body={"thinking": {"type": "disabled"}},
        timeout=120,
        max_retries=2,
    )
