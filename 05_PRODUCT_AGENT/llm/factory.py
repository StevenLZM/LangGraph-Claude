from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm.resilient_llm import OfflineStubLLM, ResilientLLM


@dataclass(frozen=True)
class CustomerServiceLLMSetup:
    llm: Any
    startup_error: str = ""


class UnavailableLLM:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    async def ainvoke(self, messages: list[object]) -> str:
        del messages
        raise RuntimeError(self.reason)

    async def ainvoke_with_metadata(self, messages: list[object]) -> Any:
        del messages
        raise RuntimeError(self.reason)


def build_customer_service_llm(settings: Any) -> CustomerServiceLLMSetup:
    llm_mode = getattr(settings, "llm_mode", "deepseek")
    if llm_mode == "offline_stub":
        return _unavailable_setup("LLM_MODE=offline_stub is disabled. Configure a real LLM provider such as deepseek.")

    if llm_mode == "deepseek":
        deepseek_key = getattr(settings, "deepseek_api_key", "")
        if not deepseek_key:
            return _unavailable_setup("LLM_MODE=deepseek, but DEEPSEEK_API_KEY is not configured.")
        try:
            primary_client = _build_deepseek_client(
                model=getattr(settings, "deepseek_max_model", "deepseek-v4-pro"),
                api_key=deepseek_key,
                base_url=getattr(settings, "deepseek_base_url", "https://api.deepseek.com"),
            )
            fallback_client = _build_deepseek_client(
                model=getattr(settings, "deepseek_light_model", "deepseek-v4-flash"),
                api_key=deepseek_key,
                base_url=getattr(settings, "deepseek_base_url", "https://api.deepseek.com"),
            )
        except Exception as exc:
            return _unavailable_setup(str(exc))
        return CustomerServiceLLMSetup(
            llm=ResilientLLM(
                primary_client=primary_client,
                fallback_client=fallback_client,
            )
        )

    openai_key = getattr(settings, "openai_api_key", "")
    anthropic_key = getattr(settings, "anthropic_api_key", "")
    if not openai_key and not anthropic_key:
        return _unavailable_setup("LLM_MODE requires OPENAI_API_KEY or ANTHROPIC_API_KEY when not using deepseek.")

    try:
        primary_client = _build_primary_client(settings)
        fallback_client = _build_fallback_client(settings)
    except Exception as exc:
        return _unavailable_setup(str(exc))

    return CustomerServiceLLMSetup(
        llm=ResilientLLM(
            primary_client=primary_client,
            fallback_client=fallback_client,
        )
    )


def _unavailable_setup(reason: str) -> CustomerServiceLLMSetup:
    return CustomerServiceLLMSetup(llm=UnavailableLLM(reason), startup_error=reason)


def _build_primary_client(settings: Any) -> Any:
    primary_model = getattr(settings, "primary_model", "gpt-4o-mini")
    if primary_model.casefold().startswith("claude") and getattr(settings, "anthropic_api_key", ""):
        return _build_anthropic_client(
            model=primary_model,
            api_key=getattr(settings, "anthropic_api_key", ""),
        )
    if getattr(settings, "openai_api_key", ""):
        return _build_openai_client(
            model=primary_model,
            api_key=getattr(settings, "openai_api_key", ""),
            base_url=getattr(settings, "openai_base_url", ""),
        )
    if getattr(settings, "anthropic_api_key", ""):
        return _build_anthropic_client(
            model=primary_model,
            api_key=getattr(settings, "anthropic_api_key", ""),
        )
    return OfflineStubLLM()


def _build_fallback_client(settings: Any) -> Any:
    fallback_model = getattr(settings, "fallback_model", "gpt-4o-mini")
    if fallback_model.casefold().startswith("claude") and getattr(settings, "anthropic_api_key", ""):
        return _build_anthropic_client(
            model=fallback_model,
            api_key=getattr(settings, "anthropic_api_key", ""),
        )
    if getattr(settings, "openai_api_key", ""):
        return _build_openai_client(
            model=fallback_model,
            api_key=getattr(settings, "openai_api_key", ""),
            base_url=getattr(settings, "openai_base_url", ""),
        )
    if getattr(settings, "anthropic_api_key", ""):
        return _build_anthropic_client(
            model=fallback_model,
            api_key=getattr(settings, "anthropic_api_key", ""),
        )
    return OfflineStubLLM()


def _build_openai_client(*, model: str, api_key: str, base_url: str) -> Any:
    chat_openai = _load_chat_openai()
    kwargs = {"model": model, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return chat_openai(**kwargs)


def _build_deepseek_client(*, model: str, api_key: str, base_url: str) -> Any:
    chat_openai = _load_chat_openai()
    return chat_openai(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
        timeout=240,
        max_retries=2,
    )


def _build_anthropic_client(*, model: str, api_key: str) -> Any:
    chat_anthropic = _load_chat_anthropic()
    return chat_anthropic(model=model, api_key=api_key)


def _load_chat_openai() -> Any:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError("langchain-openai is required when OPENAI_API_KEY is configured.") from exc
    return ChatOpenAI


def _load_chat_anthropic() -> Any:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise RuntimeError("langchain-anthropic is required when ANTHROPIC_API_KEY is configured.") from exc
    return ChatAnthropic
