# 01_RAG LangSmith Audit Tracing and Env Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `01_RAG` 增加可审计、凭证安全且不改变业务结果的 LangSmith Trace，并让 `01`～`05` 项目的真实 `.env` 从后续 Git 提交中退出。

**Architecture:** 使用一个带输入、输出和 metadata 递归凭证脱敏的 LangSmith Client，统一承接 LangChain 自动 Trace 与显式 RAG Span；根请求、Query Rewrite、七阶段检索和 Generation 都只包装现有一次执行，不重放业务调用。`.gitignore` 使用项目根锚定规则，已跟踪 `.env` 通过 `git rm --cached` 退出当前索引但保留本地文件与历史。

**Tech Stack:** Python 3.11、LangChain Core、LangSmith 0.7.x、pytest、Streamlit、Git

## Global Constraints

- 以 `docs/superpowers/specs/2026-07-27-01-rag-langsmith-audit-tracing-design.md` 为验收基准。
- 审计模式保留用户问题、历史、重写文本、完整 Chunk/Parent 正文、完整证据 Prompt、回答和引用。
- 只按字段名隐藏凭证；不得隐藏、截断或 hash RAG 文档正文。
- 所有凭证命中值替换为 `[REDACTED_CREDENTIAL]`。
- Trace 不得修改输入对象，不得再次执行 Query Rewrite、检索、Rerank 或生成。
- LangSmith 未启用、未配置或发送失败时，RAG 主链路继续运行。
- 七阶段名称必须与 `rag.retrieval_trace.REQUIRED_EVAL_STAGES` 完全一致。
- 保留现有并行 BM25/Dense 行为，并正确传播 tracing context。
- 不修改检索排名、分数、候选数量、回答校验或评测公式。
- 不暂存或提交真实 `.env` 的当前内容。
- `git rm --cached` 只针对明确列出的四个当前已跟踪文件；不得删除本地文件或改写历史。
- 实现采用测试先行并分小提交。

---

## Task 1: 实现只隐藏凭证的递归 sanitizer

**Files:**

- Create: `01_RAG/rag/langsmith_tracing.py`
- Create: `01_RAG/tests/test_langsmith_tracing.py`

- [ ] **Step 1: 写嵌套凭证脱敏失败测试**

```python
def test_sanitize_trace_credentials_redacts_nested_credential_fields():
    payload = {
        "question": "保修期多久？",
        "DASHSCOPE_API_KEY": "unique-api-key-value",
        "headers": {
            "Authorization": "Bearer unique-token",
            "client-secret": "unique-client-secret",
        },
        "items": [
            {"refresh_token": "unique-refresh-token"},
            {"cookie": "unique-cookie"},
        ],
    }

    sanitized = sanitize_trace_credentials(payload)

    assert sanitized["DASHSCOPE_API_KEY"] == "[REDACTED_CREDENTIAL]"
    assert sanitized["headers"]["Authorization"] == "[REDACTED_CREDENTIAL]"
    assert sanitized["headers"]["client-secret"] == "[REDACTED_CREDENTIAL]"
    assert sanitized["items"][0]["refresh_token"] == "[REDACTED_CREDENTIAL]"
    assert sanitized["items"][1]["cookie"] == "[REDACTED_CREDENTIAL]"
```

参数化覆盖：

```text
api_key, apikey, authorization, password, passwd, token,
access_token, refresh_token, secret, client_secret, cookie
```

以及大小写、连字符、下划线差异。

- [ ] **Step 2: 写保留完整审计正文和不修改输入的失败测试**

```python
def test_sanitizer_preserves_audit_content_without_mutating_input():
    payload = {
        "question": "unique_question_text",
        "page_content": "unique_chunk_secret_text",
        "context": "完整证据 [S1]",
        "answer": "保修期十二个月 [S1]",
        "metadata": {
            "doc_id": "doc-1",
            "parent_id": "parent-1",
            "rerank_score": 0.91,
        },
    }
    original = deepcopy(payload)

    sanitized = sanitize_trace_credentials(payload)

    assert sanitized == original
    assert payload == original
    assert sanitized is not payload
```

这里的 `unique_chunk_secret_text` 是文档正文测试标记，不是凭证，必须保留。

- [ ] **Step 3: 写常见容器和 LangChain 对象兼容测试**

覆盖：

- tuple 保持 tuple。
- list 保持 list。
- mapping 转为同等普通 mapping 内容。
- `Document` 实例可序列化为保留 `page_content` 和 `metadata` 的结构。
- 基础值、`None` 不变。
- 未知对象用安全、有限的字符串表示，不调用会改变业务状态的方法。

- [ ] **Step 4: 运行测试并确认红灯**

Run:

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent \
  python -m pytest tests/test_langsmith_tracing.py -q
```

Expected: import `rag.langsmith_tracing` 失败。

- [ ] **Step 5: 实现递归 sanitizer**

在 `rag/langsmith_tracing.py` 定义：

```python
REDACTED_CREDENTIAL = "[REDACTED_CREDENTIAL]"
_CREDENTIAL_KEYS = frozenset(
    {
        "apikey",
        "authorization",
        "password",
        "passwd",
        "token",
        "accesstoken",
        "refreshtoken",
        "secret",
        "clientsecret",
        "cookie",
    }
)


def _normalize_key(key: object) -> str:
    return re.sub(r"[_-]", "", str(key)).casefold()


def _is_credential_key(key: object) -> bool:
    normalized = _normalize_key(key)
    return normalized in _CREDENTIAL_KEYS or any(
        normalized.endswith(marker)
        for marker in _CREDENTIAL_KEYS
    )


def sanitize_trace_credentials(payload: Any) -> Any:
    return _sanitize_trace_value(payload, seen=set())
```

递归规则：

- Mapping：新建结果；若 key 归一化后等于或以后缀形式命中集合，值直接替换。这样 `DASHSCOPE_API_KEY`、`ES_PASSWORD`、`custom_access_token` 也会被隐藏。
- `Document`：转换为包含原始 `page_content` 与 `metadata` 两个字段的新 mapping 后递归。
- list/tuple：递归创建同类容器。
- dataclass：通过 `dataclasses.fields()` 读取公开字段后递归，不使用原地修改。
- `str/int/float/bool/None`：原样返回。
- 其他对象：使用 `repr()`，并设置最大长度仅用于未知对象；已知 RAG 正文不得截断。
- 使用循环检测集合避免自引用无限递归。

- [ ] **Step 6: 运行 sanitizer 测试并确认绿灯**

Run:

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent \
  python -m pytest tests/test_langsmith_tracing.py -q
```

Expected: 当前 Task 测试全部通过。

- [ ] **Step 7: 提交 sanitizer**

```bash
git add 01_RAG/rag/langsmith_tracing.py \
  01_RAG/tests/test_langsmith_tracing.py
git commit -m "01_RAG 增加 LangSmith 凭证脱敏"
```

---

## Task 2: 增加安全 Client、no-op context 和容错 Span API

**Files:**

- Modify: `01_RAG/rag/langsmith_tracing.py`
- Modify: `01_RAG/tests/test_langsmith_tracing.py`

- [ ] **Step 1: 写配置兼容和禁用模式失败测试**

通过显式 env mapping 或 monkeypatch 测试：

```python
def test_tracing_is_disabled_without_flag_or_api_key():
    assert resolve_langsmith_settings({}).enabled is False


def test_settings_accept_langchain_compatibility_names():
    settings = resolve_langsmith_settings(
        {
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_API_KEY": "key",
            "LANGCHAIN_PROJECT": "project",
        }
    )
    assert settings.enabled is True
    assert settings.project_name == "project"
```

优先级固定为 `LANGSMITH_*` 高于 `LANGCHAIN_*`。

- [ ] **Step 2: 写安全 Client 工厂失败测试**

Fake `Client` 捕获构造参数：

```python
def test_safe_client_applies_sanitizer_to_all_payload_channels(monkeypatch):
    client = get_safe_langsmith_client(settings)

    assert client.kwargs["hide_inputs"] is sanitize_trace_credentials
    assert client.kwargs["hide_outputs"] is sanitize_trace_credentials
    assert client.kwargs["hide_metadata"] is sanitize_trace_credentials
```

断言禁用时不构造 Client，并且单进程内重复获取返回同一实例。

- [ ] **Step 3: 写 no-op 和错误隔离失败测试**

```python
def test_trace_span_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(
        tracing,
        "resolve_langsmith_settings",
        lambda environ=None: LangSmithSettings(False, "", ""),
    )
    called = []
    with trace_span("rag.request") as span:
        called.append(span)
    assert called == [None]
```


Fake LangSmith context/trace 在进入、`end()` 或退出时分别抛错，测试中的业务函数返回 `"business-result"`，断言最终结果不变且业务函数调用计数严格等于 1。

- [ ] **Step 4: 运行测试并确认红灯**

Run:

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent \
  python -m pytest tests/test_langsmith_tracing.py -q
```

Expected: 新 settings/client/context API 尚不存在。

- [ ] **Step 5: 实现配置和 Client API**

定义：

```python
@dataclass(frozen=True)
class LangSmithSettings:
    enabled: bool
    api_key: str
    project_name: str
```

实现 `resolve_langsmith_settings(environ: Mapping[str, str] | None = None) -> LangSmithSettings`、`get_safe_langsmith_client(settings: LangSmithSettings | None = None) -> Client | None`，并用带 `@lru_cache(maxsize=4)` 的 `_build_safe_langsmith_client(settings: LangSmithSettings) -> Client` 保存按配置区分的 Client。

`get_safe_langsmith_client()` 只有在 enabled 且 key 非空时才创建：

```python
Client(
    api_key=settings.api_key,
    hide_inputs=sanitize_trace_credentials,
    hide_outputs=sanitize_trace_credentials,
    hide_metadata=sanitize_trace_credentials,
)
```

不得记录 `settings.api_key`。

- [ ] **Step 6: 实现 context 和 Span API**

公开接口固定为：

- `rag_tracing_context(*, tags: Sequence[str] = (), metadata: Mapping[str, Any] | None = None) -> Iterator[None]`
- `trace_span(name: str, *, run_type: str = "chain", inputs: Mapping[str, Any] | None = None, tags: Sequence[str] = (), metadata: Mapping[str, Any] | None = None) -> Iterator[Any | None]`
- `end_trace_span(span: Any | None, *, outputs: Mapping[str, Any]) -> None`

行为要求：

- `rag_tracing_context()` 用 `langsmith.tracing_context(client=client, project_name=settings.project_name, enabled=True, tags=tags, metadata=safe_metadata)`。
- `trace_span()` 用 `langsmith.trace(name=name, run_type=run_type, client=client, inputs=safe_inputs, tags=tags, metadata=safe_metadata)`。
- 所有显式 inputs/outputs/metadata 先走 sanitizer。
- disabled 时三个函数均为低成本 no-op。
- LangSmith 客户端或 transport 异常写 `logging.warning()`，不向业务层传播。
- 不捕获 `with` 块内部业务函数抛出的异常；业务异常必须按原语义继续抛出。

- [ ] **Step 7: 运行测试并确认绿灯**

Run:

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent \
  python -m pytest tests/test_langsmith_tracing.py -q
```

- [ ] **Step 8: 提交 Client 和 context**

```bash
git add 01_RAG/rag/langsmith_tracing.py \
  01_RAG/tests/test_langsmith_tracing.py
git commit -m "01_RAG 增加安全 LangSmith trace context"
```

---

## Task 3: 在根请求、Query Rewrite、Retrieval 和 Generation 上建立审计 Span

**Files:**

- Modify: `01_RAG/rag/chain.py`
- Modify: `01_RAG/tests/test_rag_execution_trace.py`
- Modify: `01_RAG/tests/test_langsmith_tracing.py`

- [ ] **Step 1: 写根、rewrite、retrieval、generation Span 失败测试**

Monkeypatch `rag.chain.trace_span` 为 Fake context manager，记录名称、输入和输出：

```python
def test_rag_execution_records_root_rewrite_retrieval_and_generation_spans(
    monkeypatch,
):
    execution = chain.run_rag_with_trace(
        "保修期多久？",
        chat_history=[HumanMessage(content="上一轮问题")],
        trace_metadata={"session_id": "session-1"},
        trace_tags=("01-rag",),
    )

    assert [span.name for span in spans] == [
        "rag.request",
        "query.rewrite",
        "retrieval",
        "generation",
    ]
    assert spans[0].inputs["question"] == "保修期多久？"
    assert spans[0].inputs["chat_history"][0].content == "上一轮问题"
    assert spans[1].outputs["rewritten_query"]
    assert spans[2].outputs["documents"]
    assert spans[3].outputs["answer"] == execution.answer
```

同时断言根 outputs 包含 `answer`、完整 source documents 和 evaluation trace。

- [ ] **Step 2: 写无证据时不虚构 generation 的失败测试**

无最终 documents 时：

- `rag.request`、`query.rewrite` 和 `retrieval` 存在。
- `generation` 可作为明确的 skipped span，outputs 包含 `{"skipped": True, "reason": "no_evidence"}`，但不得调用生成模型。
- 业务回答仍为 `EVIDENCE_INSUFFICIENT_ANSWER`。

- [ ] **Step 3: 写引用重试不会被重复 Trace 执行的失败测试**

让第一次生成返回无效引用、第二次返回有效引用，断言：

- 生成函数恰好调用两次，这是现有业务重试语义。
- tracing 包装没有额外第三次调用。
- generation Span outputs 记录 `attempts=2` 和最终回答。

- [ ] **Step 4: 运行失败测试**

Run:

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent \
  python -m pytest \
  tests/test_rag_execution_trace.py \
  tests/test_langsmith_tracing.py \
  -q
```

- [ ] **Step 5: 给 `run_rag_with_trace()` 增加 tracing 元数据参数**

签名改为：

```python
def run_rag_with_trace(
    question: str,
    *,
    chat_history: Sequence[Any] = (),
    auth_context: dict[str, Any] | None = None,
    trace_tags: Sequence[str] = (),
    trace_metadata: Mapping[str, Any] | None = None,
) -> RagExecution:
```

在 `rag_tracing_context()` 内包裹一次完整执行，并依次创建：

```text
rag.request
query.rewrite
retrieval
generation
```

根 inputs 包含当前 question、history、auth context；根 outputs 包含回答、最终 documents、七阶段 EvaluationTrace 和 generation latency。

Rewrite outputs 包含 `rewritten_query`、`time_intent` 和原始 rewrite 结果。`retrieval` inputs 包含 rewritten query、time intent 和 auth context，outputs 包含实际生效的 metadata filter、最终完整 documents 与 EvaluationTrace；实际 filter 由 `retrieve_with_trace()` 计算一次并放入 trace metadata，不能在 chain 中复制业务规则。七个阶段 Span 在这个父 Span 内执行。Generation inputs 包含完整 question、history、documents、rendered evidence context 和 system prompt；不自行拼装与实际 LLM 不同的 Prompt，直接复用 `format_docs_for_context()` 与 `SYSTEM_PROMPT`。

- [ ] **Step 6: 让 Runnable 读取标准 `RunnableConfig` metadata/tags**

把内部函数改为可接收 config：

```python
from langchain_core.runnables import RunnableConfig


def invoke(
    input_dict: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    execution = run_rag_with_trace(
        str(input_dict["question"]),
        chat_history=input_dict.get("chat_history") or (),
        auth_context=input_dict.get("auth_context") or {},
        trace_tags=tuple(config.get("tags") or ()),
        trace_metadata=dict(config.get("metadata") or {}),
    )
```

不得把 `configurable` 中的非审计对象整体发送；session id 由 app 显式放到 metadata。

- [ ] **Step 7: 运行测试并确认绿灯**

Run:

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent \
  python -m pytest \
  tests/test_rag_execution_trace.py \
  tests/test_langsmith_tracing.py \
  -q
```

- [ ] **Step 8: 提交链路 Trace**

```bash
git add 01_RAG/rag/chain.py \
  01_RAG/tests/test_rag_execution_trace.py \
  01_RAG/tests/test_langsmith_tracing.py
git commit -m "01_RAG 追踪请求重写与答案生成"
```

---

## Task 4: 给七个生产检索阶段增加准确 Span

**Files:**

- Modify: `01_RAG/rag/retriever.py`
- Modify: `01_RAG/tests/test_seven_stage_trace.py`
- Modify: `01_RAG/tests/test_langsmith_tracing.py`

- [ ] **Step 1: 写七阶段名称与 payload 失败测试**

复用 `RetrievalPipelineComponents` 的纯 Fake pipeline：

```python
def test_retrieval_records_all_required_langsmith_stage_spans(monkeypatch):
    execution = retrieve_with_trace(
        "保修期",
        retrieval_context={"tenant_id": "tenant-1"},
        components=components,
    )

    assert tuple(span.name for span in spans) == REQUIRED_EVAL_STAGES
    assert execution.trace.require_complete() is None
```

对每个 span 断言：

- inputs 包含 query、metadata filter 和前一阶段候选。
- outputs 包含本阶段完整 documents、configured_k、score_type、latency_ms。
- document payload 包含 `page_content` 和 metadata 中的 ID、source、page、所有已有 score。

- [ ] **Step 2: 写每阶段业务函数只调用一次的失败测试**

每个 Fake pipeline callable 自增计数，执行后断言全部等于 1。BM25 或 Dense 降级失败时，相应 span 输出包含 error 和空 candidates，但另一路及后续流程不被 Trace 重放。

- [ ] **Step 3: 写 BM25/Dense 并行 context 传播失败测试**

使用 `ContextVar` 作为轻量代理，主线程设置 trace marker，两条 worker callable 都必须读到同一 marker：

```python
def test_parallel_recall_copies_trace_context_to_each_worker():
    marker.set("rag-request-1")
    retrieve_with_trace(
        "保修期",
        retrieval_context={"trace_id": "trace-1"},
        components=components,
    )
    assert seen_by_bm25 == ["rag-request-1"]
    assert seen_by_dense == ["rag-request-1"]
```

同一个 `contextvars.Context` 不能被两个线程同时进入，因此测试应促使实现为每个 `executor.submit()` 分别调用一次 `copy_context()`。

- [ ] **Step 4: 运行失败测试**

Run:

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent \
  python -m pytest \
  tests/test_seven_stage_trace.py \
  tests/test_langsmith_tracing.py \
  -q
```

- [ ] **Step 5: 实现候选序列化和 stage wrapper**

在 `rag/langsmith_tracing.py` 增加：

```python
def serialize_documents_for_trace(
    documents: Sequence[Document],
) -> list[dict[str, Any]]:
    return [
        {
            "page_content": document.page_content,
            "metadata": dict(document.metadata),
        }
        for document in documents
    ]
```

在 `rag/retriever.py` 增加 `_run_traced_stage()`；它固定接收 `name`、`function`、位置参数 tuple、审计 inputs、`configured_k` 和 `score_type`，返回业务结果与毫秒耗时组成的 tuple。

wrapper 在真实业务调用前开始计时，调用恰好一次，结束时记录完整 candidates、configured k、score type 和 measured latency。若业务调用抛错，先为 span 记录 error 后原样抛出，现有 `_parallel_recall()` 再按原规则降级。

- [ ] **Step 6: 在七个阶段复用 wrapper**

阶段与配置必须一一对应：

```python
(
    ("bm25_child", rag_config.BM25_TOP_K, "bm25_score"),
    ("dense_child", rag_config.SEMANTIC_TOP_K, "dense_score"),
    ("rrf_child", rag_config.RRF_TOP_K, "rrf_score"),
    ("cross_encoder_child", rag_config.RERANK_TOP_K, "rerank_score"),
    (
        "business_fused_child",
        rag_config.BUSINESS_FUSION_TOP_K,
        "business_score",
    ),
    (
        "diversified_parent",
        rag_config.DIVERSIFIED_PARENT_TOP_K,
        "mmr_score",
    ),
    (
        "final_context_parent",
        rag_config.FINAL_PARENT_TOP_K,
        "context_score",
    ),
)
```

继续用 `StageRecorder` 记录真实评测 trace；LangSmith span 是旁路观察，不替代 EvaluationTrace。

在构造 `StageRecorder` 时把 `effective_context["metadata_filter"]` 放入 trace metadata，供 `retrieval` 父 Span 回传审计；它只是已计算结果的快照，不再次调用 filter builder。

- [ ] **Step 7: 正确传播并行 tracing context**

提交 BM25/Dense worker 时为每个 future 创建独立 context：

```python
bm25_context = copy_context()
dense_context = copy_context()
futures = {
    "bm25_child": executor.submit(bm25_context.run, bm25_call),
    "dense_child": executor.submit(dense_context.run, dense_call),
}
```

其中 `bm25_call` 和 `dense_call` 分别是提前构造好的零参数 `functools.partial`，目标函数均为 `_run_traced_stage`，并绑定对应函数、query、context、top k 和 score type；每个 partial 只执行一次。

禁止复用同一个 `Context` 实例。

- [ ] **Step 8: 运行七阶段和降级回归**

Run:

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent \
  python -m pytest \
  tests/test_seven_stage_trace.py \
  tests/test_rag_execution_trace.py \
  tests/test_langsmith_tracing.py \
  -q
```

Expected: all passed；原有七阶段 EvaluationTrace 顺序和降级语义不变。

- [ ] **Step 9: 提交七阶段 Trace**

```bash
git add 01_RAG/rag/retriever.py \
  01_RAG/rag/langsmith_tracing.py \
  01_RAG/tests/test_seven_stage_trace.py \
  01_RAG/tests/test_langsmith_tracing.py
git commit -m "01_RAG 追踪七阶段混合检索"
```

---

## Task 5: 从 Streamlit 传递会话审计 metadata

**Files:**

- Modify: `01_RAG/app.py`
- Create or Modify: `01_RAG/tests/test_app_trace_config.py`

- [ ] **Step 1: 写调用 config 失败测试**

把构造 config 的纯函数放在 app 中，避免测试启动 Streamlit UI：

```python
def test_build_rag_invoke_config_includes_session_metadata():
    config = build_rag_invoke_config("session-123")

    assert config["configurable"]["session_id"] == "session-123"
    assert config["metadata"]["session_id"] == "session-123"
    assert config["metadata"]["application"] == "01_RAG"
    assert config["tags"] == ["01-rag", "streamlit"]
```

- [ ] **Step 2: 运行失败测试**

Run:

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent \
  python -m pytest tests/test_app_trace_config.py -q
```

- [ ] **Step 3: 实现并在 `_process_query()` 中使用**

```python
def build_rag_invoke_config(session_id: str) -> dict[str, Any]:
    return {
        "configurable": {"session_id": session_id},
        "metadata": {
            "session_id": session_id,
            "application": "01_RAG",
            "interface": "streamlit",
        },
        "tags": ["01-rag", "streamlit"],
    }
```

替换当前只有 `configurable.session_id` 的内联 dict。不得把 Streamlit session state 整体放进 metadata。

- [ ] **Step 4: 运行测试并提交**

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent \
  python -m pytest tests/test_app_trace_config.py -q
git add 01_RAG/app.py 01_RAG/tests/test_app_trace_config.py
git commit -m "01_RAG 标记 LangSmith 会话元数据"
```

---

## Task 6: 文档化 LangSmith 审计模式

**Files:**

- Modify: `01_RAG/.env.example`
- Modify: `01_RAG/README.md`

- [ ] **Step 1: 增加无 Secret 的配置示例**

```dotenv
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=01-rag-local
```

说明也兼容 `LANGCHAIN_TRACING_V2`、`LANGCHAIN_API_KEY`、`LANGCHAIN_PROJECT`，但新配置优先使用 `LANGSMITH_*`。

- [ ] **Step 2: 写清楚“隐藏了什么”和“没有隐藏什么”**

README 必须明确：

- 隐藏：API key、authorization、password、token、secret、cookie 等凭证字段的值。
- 显示为：`[REDACTED_CREDENTIAL]`。
- 不隐藏：问题、历史、完整 Chunk/Parent 正文、证据 context、Prompt、回答、引用、业务 metadata 和分数。
- 这是为了审计；LangSmith 项目权限与数据保留由部署方管理。
- 关闭或未配置 LangSmith 时不影响 RAG。
- 完整正文会提高 Trace 存储量。

- [ ] **Step 3: 校验文档没有真实 Key**

Run:

```bash
rg -n "LANGSMITH_|LANGCHAIN_|REDACTED_CREDENTIAL|审计" \
  01_RAG/README.md 01_RAG/.env.example
git diff -- 01_RAG/.env.example 01_RAG/README.md
```

人工确认所有 Key 都是空值或占位说明。

- [ ] **Step 4: 提交教学文档**

```bash
git add 01_RAG/.env.example 01_RAG/README.md
git commit -m "01_RAG 文档化 LangSmith 审计 trace"
```

---

## Task 7: 让 `01`～`05` 的真实 `.env` 退出后续 Git 提交

**Files:**

- Modify: `.gitignore`
- Untrack but keep locally:
  - `01_RAG/.env`
  - `02_REACT_AGENT/.env`
  - `03_MULTI_AGENT/.env`
  - `05_PRODUCT_AGENT/.env`

- [ ] **Step 1: 记录处理前文件存在和跟踪状态，不读取内容**

Run:

```bash
for path in \
  01_RAG/.env \
  02_REACT_AGENT/.env \
  03_MULTI_AGENT/.env \
  05_PRODUCT_AGENT/.env
do
  test -f "$path" && echo "local_exists=$path"
done
git ls-files '*/.env' '*/.env.example'
```

Expected: 四个 `.env` 本地存在且当前被跟踪；`.env.example` 也被跟踪。

- [ ] **Step 2: 更新根 `.gitignore`**

把与当前“私有仓库提交 `.env`”相冲突的旧注释改为明确规则，并加入：

```gitignore
# Local project secrets: keep files on disk, never include future commits.
/01_RAG/.env
/02_REACT_AGENT/.env
/03_MULTI_AGENT/.env
/04_HUMAN_IN_THE_LOOP/.env
/05_PRODUCT_AGENT/.env
```

保留通用 `.env` 规则和 `!.env.example` 行为；如果没有显式例外也不新增会忽略 example 的规则。

- [ ] **Step 3: 只从 Git 索引移除明确的四个文件**

Run:

```bash
git rm --cached \
  01_RAG/.env \
  02_REACT_AGENT/.env \
  03_MULTI_AGENT/.env \
  05_PRODUCT_AGENT/.env
```

该命令已由用户明确授权。不得使用文件系统删除命令。

- [ ] **Step 4: 证明本地文件仍在且规则生效**

Run:

```bash
for path in \
  01_RAG/.env \
  02_REACT_AGENT/.env \
  03_MULTI_AGENT/.env \
  05_PRODUCT_AGENT/.env
do
  test -f "$path" && echo "local_preserved=$path"
  git check-ignore -v "$path"
done
git ls-files '*/.env'
git ls-files '*/.env.example'
```

Expected:

- 四个 local file 全部 preserved。
- 五个项目的 `.env` 都有根锚定 ignore 规则。
- `git ls-files '*/.env'` 无输出。
- 各项目 `.env.example` 继续输出。

- [ ] **Step 5: 审计暂存区不包含任何 `.env` 内容**

Run:

```bash
git diff --cached --name-status
git diff --cached -- .gitignore
```

只允许看到 `.gitignore` 修改和四个 `.env` 的 Git 删除记录；不得把任一 `.env` 的当前工作区内容重新 `git add`。

- [ ] **Step 6: 提交忽略规则和 untrack 记录**

```bash
git add .gitignore
git commit -m "chore: stop tracking project env files"
```

此提交保留历史提交，不做 rebase/filter-repo，不删除本地文件。

---

## Task 8: 完整离线回归和真实 LangSmith smoke

**Files:**

- No new files expected

- [ ] **Step 1: 运行 LangSmith 与 RAG 定向测试**

Run:

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent \
  python -m pytest \
  tests/test_langsmith_tracing.py \
  tests/test_rag_execution_trace.py \
  tests/test_seven_stage_trace.py \
  tests/test_app_trace_config.py \
  -q
```

Expected: all passed。

- [ ] **Step 2: 运行 `01_RAG` 完整离线测试**

Run:

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent \
  python -m pytest tests -q -m "not es_integration and not real_eval"
```

Expected: all selected tests passed。

- [ ] **Step 3: 只输出 LangSmith 配置状态**

Run:

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent python -c \
  "from rag.langsmith_tracing import resolve_langsmith_settings; s=resolve_langsmith_settings(); print('enabled=' + str(s.enabled)); print('project_configured=' + str(bool(s.project_name))); print('api_key_configured=' + str(bool(s.api_key)))"
```

Expected: 三个布尔/公开状态正确；不输出 key 或 project 中的敏感业务内容。

- [ ] **Step 4: 在正式空 ES 上执行一次真实 RAG smoke**

使用现有配置调用一次没有证据的安全问题：

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent python -c \
  "from rag.chain import run_rag_with_trace; r=run_rag_with_trace('当前空知识库中是否有保修说明？', trace_tags=('01-rag','smoke'), trace_metadata={'session_id':'langsmith-smoke'}); print(r.answer); print(tuple(r.trace.stages))"
```

空库预期：

- 回答为证据不足安全文案。
- 七阶段 EvaluationTrace 仍完整。
- LangSmith 中有 `rag.request`、`query.rewrite`、`retrieval` 和七个 retrieval stage。
- `generation` 标记为 `no_evidence`，不会为了 smoke 摄取测试文档或调用答案生成。

这一步会调用当前配置的 Query Rewrite 模型并向 LangSmith 写入审计 Trace；若外部网络失败，记录 warning 并确认本地业务结果仍返回。

- [ ] **Step 5: 在 LangSmith UI 人工审计一次 Trace**

在 `LANGSMITH_PROJECT` 对应项目确认：

- 根名称是 `rag.request`。
- session metadata/tag 正确。
- Query Rewrite、`retrieval` 父 Span 和七阶段名称完整。
- 问题与无证据上下文可见。
- 任何测试注入的 credential 字段显示为 `[REDACTED_CREDENTIAL]`。
- 没有真实 API Key、Authorization 或 Token。

真实空库无法产生真实 Chunk 与答案 Prompt，因此“完整文档/Prompt 保留”由离线 Fake payload 测试证明；以后有正式文档的首个真实请求再做数据级审计，不为验证而摄取旧数据。

- [ ] **Step 6: 审计 Git 状态和提交边界**

Run:

```bash
git status --short --branch
git ls-files '*/.env'
git ls-files '*/.env.example'
git diff origin/main...HEAD --name-only
git log --oneline --decorate -12
```

确认：

- 当前分支是 `main`。
- `*/.env` 无跟踪结果。
- `*/.env.example` 仍跟踪。
- 本地真实 `.env` 仍存在，但不再出现在 status。
- 所有实现提交都在 main，且没有意外文件。

- [ ] **Step 7: 最终交付记录**

最终答复必须包括：

- sanitizer 隐藏的凭证字段和替换标记。
- 明确说明文档、Prompt、历史、回答没有被隐藏。
- 七个阶段名称。
- 测试命令和结果。
- 真实 smoke 结果及 LangSmith UI 仍需/已经人工确认的边界。
- 四个历史 `.env` 保留在旧提交和本地，但当前版本已 untrack，之后提交不会再包含。
