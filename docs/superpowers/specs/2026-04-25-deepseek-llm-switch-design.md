# DeepSeek LLM Switch Design

Date: 2026-04-25

## Scope

Only runtime LLM calls move from DashScope/Qwen to DeepSeek. DashScope remains in use for:

- `03_MULTI_AGENT` web fallback search through `DashScopeSearchTool`
- `01_RAG` embeddings through `DashScopeEmbeddings`

The change must not remove or rename DashScope search and embedding code paths.

## Architecture

Both app areas already use `langchain_openai.ChatOpenAI` for DashScope-compatible chat calls. DeepSeek also exposes an OpenAI-compatible chat API, so the migration should keep `ChatOpenAI` and change only provider-specific configuration:

- API key: `DEEPSEEK_API_KEY`
- Base URL: `https://api.deepseek.com`
- Strong model: `deepseek-v4-pro`
- Light model: `deepseek-v4-flash`

`03_MULTI_AGENT/config/llm.py` remains the centralized LLM factory for the multi-agent app. `01_RAG` keeps its current local factory functions, but the chat provider branch becomes DeepSeek-aware.

## Components

### 03_MULTI_AGENT

- `config/settings.py` adds DeepSeek settings while leaving DashScope settings available for search and embeddings.
- `config/llm.py` reads DeepSeek settings for all `get_llm()` calls.
- `.env.example` and `.env` document and configure DeepSeek LLM settings.
- `tools/dashscope_search_tool.py` and `app/bootstrap.py` stay functionally unchanged.

### 01_RAG

- `config.py` adds DeepSeek as the first chat provider.
- `rag/chain.py`, `rag/query_rewriter.py`, and `rag/date_extractor.py` use DeepSeek for chat/rewrite/date extraction when `DEEPSEEK_API_KEY` is present.
- `rag/embedder.py` keeps DashScope embedding behavior unchanged.
- `.env.example` and `.env` document and configure DeepSeek chat settings.

## Data Flow

For chat and reasoning:

1. Config loads `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, and model names.
2. The relevant LLM factory creates `ChatOpenAI` with DeepSeek credentials.
3. Existing chains, structured-output calls, and graph nodes consume the factory result unchanged.

For search and embeddings:

1. DashScope search continues to read `DASHSCOPE_API_KEY` in `03_MULTI_AGENT`.
2. DashScope embeddings continue to read `DASHSCOPE_API_KEY` in `01_RAG`.
3. No DeepSeek search or embedding path is introduced.

## Error Handling

Missing DeepSeek chat credentials should produce provider-specific errors that mention `DEEPSEEK_API_KEY`. DashScope-related error messages should remain only in search or embedding contexts.

Existing retry behavior should be preserved:

- `03_MULTI_AGENT` LLM factory keeps `max_retries=2` and long timeout.
- `01_RAG` chat factories keep their existing retry counts.

## Testing

Add or update focused tests to verify:

- `03_MULTI_AGENT` LLM factory uses DeepSeek key, base URL, and model mapping.
- `01_RAG` provider selection prefers DeepSeek for chat.
- DashScope search no-key behavior still returns empty results.
- DashScope embedding path is not removed.

Run the relevant existing test suites after implementation.
