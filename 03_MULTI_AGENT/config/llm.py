"""LLM 客户端工厂 —— 通过 DeepSeek OpenAI 兼容端点调用 V4 系列。

为什么用 ChatOpenAI(base_url=DeepSeek)：
  - ChatOpenAI 的 with_structured_output() 走 function calling，对 Pydantic 解析最稳
  - 与 LangChain 主线生态一致，方便切换其他 OpenAI 兼容供应商

档位约定：
  - "max"  → deepseek-v4-pro    （Planner / Reflector / Writer，重推理）
  - "turbo"→ deepseek-v4-flash  （Researcher 提炼、Supervisor，省钱）
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langchain_openai import ChatOpenAI

from config.settings import settings

Tier = Literal["max", "turbo"]


@lru_cache(maxsize=4)
def get_llm(tier: Tier = "max", *, temperature: float = 0.2, streaming: bool = False) -> ChatOpenAI:
    if not settings.deepseek_api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未配置。请在 03_MULTI_AGENT/.env 中填入。"
        )
    model = settings.deepseek_max_model if tier == "max" else settings.deepseek_light_model
    return ChatOpenAI(
        model=model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=temperature,
        streaming=streaming,
        timeout=240,
        max_retries=2,
    )
