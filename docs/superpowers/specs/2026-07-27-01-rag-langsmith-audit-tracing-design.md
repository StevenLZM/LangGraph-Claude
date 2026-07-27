# 01_RAG LangSmith 审计 Trace 与 .env 管理设计

## 目标

为 `01_RAG` 增加可用于审计和问题定位的 LangSmith Trace，并规范 `01`～`05` 项目的 `.env` Git 行为。

本次 Trace 必须：

- 保留用户问题、对话历史、完整 RAG 文档正文、证据 Prompt 和最终回答。
- 展示 Query Rewrite 与七个生产检索阶段。
- 展示每阶段候选、排名、分数、来源、页码和耗时。
- 屏蔽 API Key、Authorization、Password、Token、Secret 等凭证。
- LangSmith 不可用时不影响 RAG 主链路。

## Trace 方案

采用“LangChain 自动子 Trace + 显式 RAG 阶段 Span + 凭证脱敏 Client”。

只启用环境变量的自动 Trace 无法清晰表达 BM25、Dense、RRF、Cross-Encoder、业务融合、多样化和最终上下文阶段；只写手工 Trace 又会丢失 Prompt、LLM 和 Runnable 的内部调用。因此保留两者：

```text
rag.request
├── query.rewrite
├── retrieval
│   ├── bm25_child
│   ├── dense_child
│   ├── rrf_child
│   ├── cross_encoder_child
│   ├── business_fused_child
│   ├── diversified_parent
│   └── final_context_parent
└── generation
    ├── prompt
    └── llm
```

## 配置

使用 `01_RAG/.env` 中现有配置：

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<LangSmith API Key>
LANGSMITH_PROJECT=<项目名>
```

兼容旧变量名：

```text
LANGCHAIN_TRACING_V2
LANGCHAIN_API_KEY
LANGCHAIN_PROJECT
```

应用启动时只检查 Key 是否存在，不输出 Key。Trace 默认使用 `LANGSMITH_PROJECT` 指定的项目。

## 审计内容

### 保留原文

以下内容不做隐藏：

- 当前用户问题。
- 对话历史。
- Query Rewrite 前后文本。
- Child/Parent Chunk 的 `page_content`。
- 组装后的完整证据上下文。
- 发送给生成模型的完整 Prompt。
- 模型回答和引用。
- `doc_id`、`parent_id`、`child_id`、source、page、section。
- BM25、Dense、RRF、Rerank、业务融合、MMR 分数。
- 权限与 Metadata Filter 的业务字段。

这是显式的审计模式。LangSmith 项目的访问控制和数据保留策略由部署方负责。

### 屏蔽凭证

新增递归函数：

```python
sanitize_trace_credentials(payload: Any) -> Any
```

它只根据字段名屏蔽凭证，不屏蔽 RAG 文档：

```text
api_key
apikey
authorization
password
passwd
token
access_token
refresh_token
secret
client_secret
cookie
```

匹配规则不区分大小写，并忽略 `_`、`-` 差异。命中后的值统一替换为：

```text
[REDACTED_CREDENTIAL]
```

例如：

```json
{
  "question": "保修期多久？",
  "page_content": "产品保修期为 12 个月",
  "DASHSCOPE_API_KEY": "[REDACTED_CREDENTIAL]"
}
```

问题和文档正文保留，Key 永远不进入 LangSmith。

## 安全 Client 如何生效

创建单例 LangSmith Client：

```python
Client(
    hide_inputs=sanitize_trace_credentials,
    hide_outputs=sanitize_trace_credentials,
    hide_metadata=sanitize_trace_credentials,
)
```

每次 RAG 调用进入：

```python
with tracing_context(client=safe_client, ...):
    ...
```

这样 LangChain 自动生成的 Runnable、Prompt 和 LLM 子 Trace 与手工 Span 使用同一个 Client，并在发送前统一经过凭证脱敏。显式阶段 Span 还会使用 `process_inputs` / `process_outputs` 规范候选结构，但不会删除正文。

## 代码边界

新增：

- `rag/langsmith_tracing.py`
  - LangSmith 配置检测。
  - 安全 Client 单例。
  - 递归凭证脱敏。
  - RAG tracing context。
  - 阶段 Span 输入/输出格式。
- `tests/test_langsmith_tracing.py`
  - 凭证字段在任意嵌套层级都被替换。
  - 问题、回答和 Chunk 正文保持原样。
  - 七阶段 Span 名称和安全 Payload 正确。
  - 未启用 LangSmith 时返回 no-op context。

调整：

- `rag/chain.py`
  - `run_rag_with_trace()` 建立 `rag.request`、rewrite 和 generation Span。
- `rag/retriever.py`
  - 七阶段执行时建立对应 Span。
- `app.py`
  - 为 Streamlit 会话传入 `session_id` tag/metadata。
- `.env.example`
  - 增加 LangSmith 变量说明，不包含真实 Key。
- `README.md`
  - 说明如何启用 Trace、审计内容和凭证隐藏方式。

不改变检索结果、排序分数、回答内容或评测公式。

## 错误与性能

- LangSmith 未配置或关闭：使用 no-op context，RAG 正常运行。
- LangSmith 网络或后台发送失败：记录本地 warning，不让用户请求失败。
- Trace 记录逻辑不得修改传入对象。
- Trace 只增加观测，不重新执行 Query、检索或生成。
- 完整文档和 Prompt 会增加 Trace 存储量，这是审计模式的明确成本。

## 测试

离线测试不访问 LangSmith 服务，通过 Fake Client/Span 验证 Payload：

```bash
conda run -n langgraph-cc-multiagent \
  python -m pytest tests/test_langsmith_tracing.py -q
```

核心断言：

```text
unique_chunk_secret_text 仍存在于脱敏后的 page_content
unique_api_key_value 不存在于任何脱敏 Payload
[REDACTED_CREDENTIAL] 存在
七个 REQUIRED_EVAL_STAGES 都有对应 Span
禁用 Trace 时业务结果不变
```

真实验证使用一个问题，并在 LangSmith 项目中确认：

- 根 Trace 是 `rag.request`。
- Query Rewrite、七阶段和 generation 层级完整。
- Chunk 与 Prompt 正文可审计。
- 所有凭证字段显示为 `[REDACTED_CREDENTIAL]`。

## 01～05 `.env` Git 管理

根 `.gitignore` 增加：

```gitignore
/01_RAG/.env
/02_REACT_AGENT/.env
/03_MULTI_AGENT/.env
/04_HUMAN_IN_THE_LOOP/.env
/05_PRODUCT_AGENT/.env
```

`.env.example` 继续跟踪。

由于 `01`、`02`、`03`、`05` 的 `.env` 已经被 Git 跟踪，仅添加 `.gitignore` 不能阻止后续提交。为满足“之后的提交不再包含 `.env`”，实现时执行：

```bash
git rm --cached \
  01_RAG/.env \
  02_REACT_AGENT/.env \
  03_MULTI_AGENT/.env \
  05_PRODUCT_AGENT/.env
```

这只从当前 Git 索引移除文件：

- 本地 `.env` 保留。
- 历史提交中的 `.env` 保留。
- 后续修改不再出现在 Git status 或提交中。
- 不改写 Git 历史。

## 验收标准

1. `01_RAG` 的真实请求在 LangSmith 中形成完整层级 Trace。
2. RAG 文档、Prompt 和回答正文可审计。
3. 凭证内容不会出现在 Trace Payload。
4. 七个检索阶段都可按名称、耗时和候选结果检查。
5. LangSmith 关闭或失败不影响回答。
6. `01`～`05` 的 `.env` 规则存在，现有本地文件保留。
7. 后续 Git 提交不再包含任何真实 `.env` 文件。
