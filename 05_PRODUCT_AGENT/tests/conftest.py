from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEST_MEMORY_DB = Path("/private/tmp/05_product_agent_pytest_memory.db")
if TEST_MEMORY_DB.exists():
    TEST_MEMORY_DB.unlink()
os.environ.setdefault("MEMORY_DB", str(TEST_MEMORY_DB))
TEST_CHAT_REQUEST_DB = Path("/private/tmp/05_product_agent_pytest_chat_requests.db")
if TEST_CHAT_REQUEST_DB.exists():
    TEST_CHAT_REQUEST_DB.unlink()
os.environ.setdefault("CHAT_REQUEST_DB", str(TEST_CHAT_REQUEST_DB))
os.environ["LLM_MODE"] = "deepseek"
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test")
os.environ["REDIS_URL"] = ""
os.environ["DATABASE_URL"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "false"


class TestRealLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.messages: list[list[object]] = []

    async def ainvoke(self, messages: list[object]) -> str:
        self.calls += 1
        self.messages.append(messages)
        return _extract_draft_answer(messages)

    async def ainvoke_with_metadata(self, messages: list[object]) -> SimpleNamespace:
        content = await self.ainvoke(messages)
        return SimpleNamespace(
            content=content,
            model_used="test-real-llm",
            fallback_used=False,
            attempts=1,
            circuit_state="closed",
        )


def _extract_draft_answer(messages: list[object]) -> str:
    import ast

    if not messages:
        return "LLM 测试回答"
    content = str(getattr(messages[-1], "content", messages[-1]))
    if "后端上下文：" not in content:
        return "LLM 测试回答"
    try:
        context = ast.literal_eval(content.split("后端上下文：", 1)[1])
    except (SyntaxError, ValueError):
        return "LLM 测试回答"
    return str(context.get("draft_answer") or "LLM 测试回答")


@pytest.fixture(autouse=True)
def reset_rate_limiter_state():
    try:
        import api.main as main
    except ImportError:
        yield
        return

    main.rate_limiter.reset()
    main.customer_service_llm = TestRealLLM()
    main.customer_service_llm_setup = SimpleNamespace(startup_error="")
    yield
    main.rate_limiter.reset()
