# DeepSeek LLM Switch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch only chat/reasoning LLM calls from DashScope/Qwen to DeepSeek V4 while keeping DashScope search and embeddings intact.

**Architecture:** Keep the existing `langchain_openai.ChatOpenAI` integration because DeepSeek exposes an OpenAI-compatible API. Add DeepSeek chat settings next to existing DashScope settings, and update only LLM factory branches. DashScope search and embedding code paths continue to use `DASHSCOPE_API_KEY`.

**Tech Stack:** Python, LangChain `ChatOpenAI`, Pydantic Settings, pytest, python-dotenv.

---

## File Structure

- Modify `03_MULTI_AGENT/config/settings.py`: add DeepSeek chat settings while retaining DashScope settings for search and embeddings.
- Modify `03_MULTI_AGENT/config/llm.py`: create `ChatOpenAI` clients with DeepSeek settings and V4 model tier mapping.
- Create `03_MULTI_AGENT/tests/test_deepseek_llm.py`: verify model mapping, DeepSeek base URL, and missing-key error.
- Modify `03_MULTI_AGENT/.env.example`: document DeepSeek LLM plus DashScope search/embedding.
- Modify `03_MULTI_AGENT/.env`: configure the provided DeepSeek key locally.
- Modify `01_RAG/config.py`: add DeepSeek as first chat provider and keep DashScope available for embeddings.
- Modify `01_RAG/rag/chain.py`: use DeepSeek for chat generation.
- Modify `01_RAG/rag/query_rewriter.py`: use DeepSeek for rewrite LLM.
- Modify `01_RAG/rag/date_extractor.py`: use DeepSeek for date extraction LLM fallback.
- Modify `01_RAG/rag/embedder.py`: choose DashScope embeddings directly from `has_dashscope()` so DeepSeek chat does not bypass DashScope embeddings.
- Create `01_RAG/tests/test_deepseek_llm_provider.py`: verify chat provider preference and embedding preservation.
- Modify `01_RAG/.env.example`: document DeepSeek chat plus DashScope embedding.
- Modify `01_RAG/.env`: configure the provided DeepSeek key locally.

---

### Task 1: 03_MULTI_AGENT DeepSeek LLM Factory

**Files:**
- Create: `03_MULTI_AGENT/tests/test_deepseek_llm.py`
- Modify: `03_MULTI_AGENT/config/settings.py`
- Modify: `03_MULTI_AGENT/config/llm.py`
- Modify: `03_MULTI_AGENT/.env.example`
- Modify: `03_MULTI_AGENT/.env`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import pytest

from config import llm as llm_module


def _base_url(llm) -> str:
    return str(llm.openai_api_base).rstrip("/")


def test_get_llm_uses_deepseek_light_model(monkeypatch):
    llm_module.get_llm.cache_clear()
    monkeypatch.setattr(llm_module.settings, "deepseek_api_key", "sk-test")
    monkeypatch.setattr(llm_module.settings, "deepseek_base_url", "https://api.deepseek.com")
    monkeypatch.setattr(llm_module.settings, "deepseek_max_model", "deepseek-v4-pro")
    monkeypatch.setattr(llm_module.settings, "deepseek_light_model", "deepseek-v4-flash")

    llm = llm_module.get_llm("turbo", temperature=0.1)

    assert llm.model_name == "deepseek-v4-flash"
    assert _base_url(llm) == "https://api.deepseek.com"
    assert llm.temperature == 0.1


def test_get_llm_uses_deepseek_strong_model(monkeypatch):
    llm_module.get_llm.cache_clear()
    monkeypatch.setattr(llm_module.settings, "deepseek_api_key", "sk-test")
    monkeypatch.setattr(llm_module.settings, "deepseek_base_url", "https://api.deepseek.com")
    monkeypatch.setattr(llm_module.settings, "deepseek_max_model", "deepseek-v4-pro")
    monkeypatch.setattr(llm_module.settings, "deepseek_light_model", "deepseek-v4-flash")

    llm = llm_module.get_llm("max")

    assert llm.model_name == "deepseek-v4-pro"
    assert _base_url(llm) == "https://api.deepseek.com"


def test_get_llm_requires_deepseek_key(monkeypatch):
    llm_module.get_llm.cache_clear()
    monkeypatch.setattr(llm_module.settings, "deepseek_api_key", "")

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        llm_module.get_llm("max")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_deepseek_llm.py -q` from `03_MULTI_AGENT`

Expected: failure because `settings.deepseek_api_key` and the DeepSeek LLM factory branch do not exist yet.

- [ ] **Step 3: Implement the minimal settings change**

In `03_MULTI_AGENT/config/settings.py`, add:

```python
    # LLM (DeepSeek)
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_max_model: str = "deepseek-v4-pro"
    deepseek_light_model: str = "deepseek-v4-flash"

    # DashScope (search / embeddings)
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_max_model: str = "qwen-max"
    qwen_light_model: str = "qwen-plus"
    qwen_embedding_model: str = "text-embedding-v3"
```

- [ ] **Step 4: Implement the minimal LLM factory change**

In `03_MULTI_AGENT/config/llm.py`, use:

```python
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
```

- [ ] **Step 5: Update env files**

Add these names to `03_MULTI_AGENT/.env.example` and set matching values in `03_MULTI_AGENT/.env`:

```dotenv
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MAX_MODEL=deepseek-v4-pro
DEEPSEEK_LIGHT_MODEL=deepseek-v4-flash
```

Leave `DASHSCOPE_API_KEY` in place for `DashScopeSearchTool`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_deepseek_llm.py -q` from `03_MULTI_AGENT`

Expected: all tests in this file pass.

---

### Task 2: 01_RAG DeepSeek Chat Provider With DashScope Embeddings

**Files:**
- Create: `01_RAG/tests/test_deepseek_llm_provider.py`
- Modify: `01_RAG/config.py`
- Modify: `01_RAG/rag/chain.py`
- Modify: `01_RAG/rag/query_rewriter.py`
- Modify: `01_RAG/rag/date_extractor.py`
- Modify: `01_RAG/rag/embedder.py`
- Modify: `01_RAG/.env.example`
- Modify: `01_RAG/.env`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_deepseek_llm_provider.py -q` from `01_RAG`

Expected: failure because `DEEPSEEK_API_KEY` and DeepSeek provider selection are not implemented.

- [ ] **Step 3: Implement DeepSeek config**

In `01_RAG/config.py`, add:

```python
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
```

and class fields:

```python
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "deepseek-v4-pro")
    REWRITE_MODEL: str = os.getenv("REWRITE_MODEL", "deepseek-v4-flash")
```

Then add:

```python
    @classmethod
    def has_deepseek(cls) -> bool:
        return bool(cls.DEEPSEEK_API_KEY and cls.DEEPSEEK_API_KEY.startswith("sk-"))
```

and make `provider()` return `"deepseek"` before checking DashScope.

- [ ] **Step 4: Implement DeepSeek chat branches**

Update each LLM factory in `chain.py`, `query_rewriter.py`, and `date_extractor.py`:

```python
if provider == "deepseek":
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=llm_config.DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        max_retries=2,
    )
```

For `chain.py`, use `temperature=0.3` and `max_retries=3` to preserve existing behavior.

- [ ] **Step 5: Preserve DashScope embeddings**

Update `01_RAG/rag/embedder.py` so `get_embeddings()` does not depend on the chat provider:

```python
def get_embeddings() -> Embeddings:
    if llm_config.has_dashscope():
        return _get_dashscope_embeddings()
    if llm_config.has_openai():
        return _get_openai_embeddings()
    return _get_huggingface_embeddings()
```

- [ ] **Step 6: Update env files**

Add these names to `01_RAG/.env.example` and set matching values in `01_RAG/.env`:

```dotenv
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
CHAT_MODEL=deepseek-v4-pro
REWRITE_MODEL=deepseek-v4-flash
```

Leave `DASHSCOPE_API_KEY` available for DashScope embeddings.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `pytest tests/test_deepseek_llm_provider.py -q` from `01_RAG`

Expected: all tests in this file pass.

---

### Task 3: Regression Verification

**Files:**
- Test only.

- [ ] **Step 1: Verify DashScope search remains intact**

Run: `pytest tests/test_dashscope_search.py -q` from `03_MULTI_AGENT`

Expected: existing DashScope search tests pass.

- [ ] **Step 2: Verify 03_MULTI_AGENT focused tests**

Run: `pytest tests/test_deepseek_llm.py tests/test_dashscope_search.py -q` from `03_MULTI_AGENT`

Expected: all selected multi-agent tests pass.

- [ ] **Step 3: Verify 01_RAG focused tests**

Run: `pytest tests/test_deepseek_llm_provider.py tests/test_query_rewriter.py -q` from `01_RAG`

Expected: all selected RAG tests pass.

- [ ] **Step 4: Inspect final diff**

Run: `git diff --stat`

Expected: only planned files plus uncommitted pre-existing user changes appear. Do not revert unrelated changes.
