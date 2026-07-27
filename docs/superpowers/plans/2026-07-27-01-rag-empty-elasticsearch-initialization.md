# 01_RAG Empty Elasticsearch Initialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `01_RAG` 增加可重复执行的正式 Elasticsearch 空索引初始化命令，并以回读校验证明 Mapping、1024 维向量字段和读写别名正确。

**Architecture:** 复用 `ElasticsearchChildStore` 作为索引结构的唯一真相源，把 `ensure_index()` 扩展为严格幂等的状态机；单独的 CLI 只负责配置预检、ES 连通性、调用初始化和最终回读。命令不迁移、不摄取、不删除任何数据，索引维度或别名冲突时直接失败。

**Tech Stack:** Python 3.11、Elasticsearch 8.19、`elasticsearch` Python client 8.x、pytest、conda 环境 `langgraph-cc-multiagent`

## Global Constraints

- 以 `docs/superpowers/specs/2026-07-27-01-rag-empty-elasticsearch-initialization-design.md` 为验收基准。
- 只操作正式物理索引 `rag-child-chunks-v1` 及别名 `rag-child-chunks-read`、`rag-child-chunks-write`。
- 不迁移旧 ES 索引、SQLite Parent Store 或 `data/documents/`。
- 不删除或修改 `products`、`products_vec`、`rag-chunks` 等现有索引。
- 不提供 `--force`、删除、重建或别名自动改绑功能。
- 运行时不得打印 `DASHSCOPE_API_KEY`、ES 密码或 ES API Key。
- 不暂存、提交或改写任何真实 `.env` 内容。
- 所有 Python 命令从 `01_RAG/` 执行，并使用 `conda run -n langgraph-cc-multiagent`。
- 实现采用测试先行：每个行为先看到预期失败，再写最小实现，再运行回归。

---

## Task 1: 将索引初始化行为定义为可测试的幂等状态机

**Files:**

- Modify: `01_RAG/rag/elasticsearch_store.py`
- Modify: `01_RAG/tests/test_elasticsearch_store.py`

- [ ] **Step 1: 为首次创建结果补充失败测试**

把现有首次创建测试扩展为验证结构化返回值：

```python
def test_ensure_index_creates_versioned_index_and_aliases():
    from rag.elasticsearch_store import ElasticsearchChildStore

    client = _CreateClient()
    store = ElasticsearchChildStore(client=client, config=_config())

    result = store.ensure_index(vector_dims=3)

    assert result.status == "created"
    assert result.physical_index == "rag-child-chunks-v1"
    assert result.vector_dims == 3
    assert result.read_alias == "rag-child-chunks-read"
    assert result.write_alias == "rag-child-chunks-write"
```

- [ ] **Step 2: 为已有正确索引的真正 no-op 补充失败测试**

Fake indices 记录 `create_calls` 和 `update_alias_calls`，并让 `get_alias()` 返回两个正确别名：

```python
def test_ensure_index_is_noop_when_mapping_and_aliases_match():
    result = store.ensure_index(vector_dims=3)

    assert result.status == "already_initialized"
    assert client.indices.create_calls == 0
    assert client.indices.update_alias_calls == []
```

- [ ] **Step 3: 为只补齐缺失别名补充失败测试**

分别覆盖缺 read alias、缺 write alias；断言只发送缺失的 `add` action：

```python
def test_ensure_index_repairs_only_missing_write_alias():
    result = store.ensure_index(vector_dims=3)

    assert result.status == "aliases_repaired"
    assert client.indices.update_alias_calls == [
        {
            "actions": [
                {
                    "add": {
                        "index": "rag-child-chunks-v1",
                        "alias": "rag-child-chunks-write",
                        "is_write_index": True,
                    }
                }
            ]
        }
    ]
```

- [ ] **Step 4: 为别名冲突和 Mapping 异常补充失败测试**

覆盖以下情况：

```python
@pytest.mark.parametrize(
    ("alias_name", "other_index"),
    [
        ("rag-child-chunks-read", "another-index"),
        ("rag-child-chunks-write", "another-index"),
    ],
)
def test_ensure_index_rejects_alias_bound_to_another_index(
    alias_name,
    other_index,
):
    with pytest.raises(ValueError, match="别名冲突"):
        store.ensure_index(vector_dims=3)

    assert client.indices.update_alias_calls == []
```

同时增加：

- `embedding` 字段不存在时失败。
- `embedding.type != "dense_vector"` 时失败。
- 已有维度与请求维度不一致时失败且不更新别名。
- write alias 已指向本索引但 `is_write_index` 不是 `true` 时失败，不静默覆盖。

- [ ] **Step 5: 为并发创建竞争补充失败测试**

Fake client 第一次 `exists()` 返回 false，`create()` 抛出正文包含 `resource_already_exists_exception` 的异常，随后 Mapping 和 aliases 可正常回读：

```python
def test_ensure_index_recovers_from_concurrent_create_race():
    result = store.ensure_index(vector_dims=3)

    assert result.status == "already_initialized"
    assert client.indices.create_calls == 1
    assert client.indices.update_alias_calls == []
```

另加一个非 `resource_already_exists_exception` 创建错误，断言原样抛出。

- [ ] **Step 6: 运行测试并确认红灯来自缺失的新契约**

Run:

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent \
  python -m pytest tests/test_elasticsearch_store.py -q
```

Expected: 新增断言因 `ensure_index()` 仍返回 `None`、未检查别名冲突而失败；原有非初始化测试继续通过。

- [ ] **Step 7: 增加初始化结果类型和严格校验实现**

在 `rag/elasticsearch_store.py` 增加：

```python
from dataclasses import dataclass
from typing import Literal

InitializationStatus = Literal[
    "created",
    "aliases_repaired",
    "already_initialized",
]


@dataclass(frozen=True)
class IndexInitializationResult:
    status: InitializationStatus
    physical_index: str
    vector_dims: int
    read_alias: str
    write_alias: str
```

把 `ensure_index()` 改为：

```python
def ensure_index(self, vector_dims: int) -> IndexInitializationResult:
    self._validate_vector_dims(vector_dims)
    index = self.config.PHYSICAL_INDEX

    if not self.client.indices.exists(index=index):
        try:
            self._create_index(vector_dims)
        except Exception as exc:
            if not _is_resource_already_exists(exc):
                raise
        else:
            return self._initialization_result("created", vector_dims)

    return self._validate_existing_index(vector_dims)
```

拆出 `_create_index(vector_dims)`、`_validate_existing_index(vector_dims)`、`_mapping_vector_dims(mapping)`、`_alias_actions_for_missing_aliases(aliases)` 和 `_initialization_result(status, vector_dims)` 五个私有方法，分别负责创建、已有结构验证、维度读取、补别名 action 构造和结果对象构造。

别名检查必须先用 `indices.exists_alias(name=alias_name)` 判断，再用 `indices.get_alias(name=alias_name)` 获取所有目标；若目标集合不是 `{physical_index}`，或 write alias 在本索引上的 `is_write_index` 不为 `True`，抛出包含别名和冲突目标索引名的 `ValueError`。只有所有检查通过后才调用一次原子的 `update_aliases()`；补齐后立即再次回读两个别名并执行同一冲突校验，避免并发改绑被误报为成功。

`_is_resource_already_exists()` 必须兼容异常对象的 `error`、`body.error.type` 和字符串形式，但不得吞掉其他 4xx/5xx：

```python
def _is_resource_already_exists(exc: Exception) -> bool:
    error = getattr(exc, "error", "")
    body = getattr(exc, "body", None)
    body_type = (
        body.get("error", {}).get("type", "")
        if isinstance(body, Mapping)
        and isinstance(body.get("error"), Mapping)
        else ""
    )
    return "resource_already_exists_exception" in {
        str(error),
        str(body_type),
    } or "resource_already_exists_exception" in str(exc)
```

- [ ] **Step 8: 运行初始化状态机测试并确认绿灯**

Run:

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent \
  python -m pytest tests/test_elasticsearch_store.py -q
```

Expected: all passed。

- [ ] **Step 9: 提交幂等状态机**

```bash
git add 01_RAG/rag/elasticsearch_store.py \
  01_RAG/tests/test_elasticsearch_store.py
git commit -m "01_RAG 严格化 ES 索引幂等初始化"
```

提交前用 `git diff --cached --name-only` 确认没有 `.env`。

---

## Task 2: 增加配置预检和正式初始化 CLI

**Files:**

- Create: `01_RAG/rag/init_elasticsearch.py`
- Create: `01_RAG/tests/test_elasticsearch_initialization.py`

- [ ] **Step 1: 先写配置预检失败测试**

测试必须通过依赖注入的配置对象完成，不修改进程环境：

```python
@pytest.mark.parametrize(
    ("api_key", "model", "dims", "message"),
    [
        ("", "qwen3.7-text-embedding", 1024, "DASHSCOPE_API_KEY"),
        ("你的真实百炼APIKey", "qwen3.7-text-embedding", 1024, "DASHSCOPE_API_KEY"),
        ("sk-valid-value", "text-embedding-v3", 1024, "EMBEDDING_MODEL"),
        ("sk-valid-value", "qwen3.7-text-embedding", 768, "ES_EMBEDDING_DIMS"),
    ],
)
def test_validate_initialization_config_rejects_invalid_values(
    api_key,
    model,
    dims,
    message,
):
    with pytest.raises(ValueError, match=message):
        validate_initialization_config(
            api_key=api_key,
            embedding_model=model,
            embedding_dims=dims,
        )
```

占位值判断至少拒绝空串、`你的真实百炼APIKey`、`your-api-key`、`replace-me`；测试不得输出这些值。

- [ ] **Step 2: 写 ES 不可用时零写入测试**

```python
def test_initialize_stops_before_index_write_when_ping_fails():
    client = FakeClient(ping_result=False)
    store = FakeStore()

    with pytest.raises(ConnectionError, match="Elasticsearch"):
        initialize_elasticsearch(
            client=client,
            store=store,
            api_key="sk-valid-value",
            embedding_model="qwen3.7-text-embedding",
            embedding_dims=1024,
        )

    assert store.ensure_calls == []
```

- [ ] **Step 3: 写成功回读和安全摘要测试**

Fake client 返回 Mapping、aliases、count，Fake store 返回三个状态中的一个：

```python
def test_initialize_returns_verified_empty_index_summary():
    summary = initialize_elasticsearch(
        client=client,
        store=store,
        api_key="sk-valid-value",
        embedding_model="qwen3.7-text-embedding",
        embedding_dims=1024,
    )

    assert summary.status == "created"
    assert summary.physical_index == "rag-child-chunks-v1"
    assert summary.vector_dims == 1024
    assert summary.document_count == 0
    assert summary.read_alias == "rag-child-chunks-read"
    assert summary.write_alias == "rag-child-chunks-write"
```

增加参数化测试，确保 `created`、`aliases_repaired`、`already_initialized` 都会进行统一回读；Mapping 或 alias 回读不一致时失败。

- [ ] **Step 4: 写 CLI 输出不泄露 Key 的测试**

在测试中把运行时 `DASHSCOPE_API_KEY` 设为唯一值 `sk-unique-secret-value`，并把 ES Client/Store 工厂替换成成功 Fake；调用 `main()` 后捕获 stdout/stderr，断言包含 `status=already_initialized` 和 `documents=0`，同时 stdout、stderr 与 caplog 都不包含该唯一 Key。

- [ ] **Step 5: 运行新测试并确认红灯**

Run:

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent \
  python -m pytest tests/test_elasticsearch_initialization.py -q
```

Expected: import `rag.init_elasticsearch` 失败。

- [ ] **Step 6: 实现初始化模块**

使用固定常量和不可变摘要：

```python
REQUIRED_EMBEDDING_MODEL = "qwen3.7-text-embedding"
REQUIRED_EMBEDDING_DIMS = 1024


@dataclass(frozen=True)
class ElasticsearchInitializationSummary:
    status: InitializationStatus
    physical_index: str
    vector_dims: int
    read_alias: str
    write_alias: str
    document_count: int
```

核心函数固定为 `validate_initialization_config(*, api_key: str, embedding_model: str, embedding_dims: int) -> None`、`initialize_elasticsearch(*, client: Any, store: ElasticsearchChildStore, api_key: str, embedding_model: str, embedding_dims: int) -> ElasticsearchInitializationSummary`、`initialize_from_environment() -> ElasticsearchInitializationSummary` 和 `main() -> int`。`initialize_from_environment()` 负责读取现有 config 单例和构造 Client/Store，使 `main()` 的输出格式可以被 Fake 安全测试。

执行顺序固定为：

1. 预检 DashScope Key、模型、维度。
2. `client.ping()`。
3. `store.ensure_index(1024)`。
4. `indices.get_mapping(index=physical_index)`。
5. `indices.get_alias(index=physical_index)`。
6. `client.count(index=physical_index)`。
7. 对 Mapping、两个别名、write flag 进行最终断言。
8. 输出不含任何认证字段的单行摘要。

CLI 入口：

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

预检失败或验证失败使用非零退出码；异常消息只提配置字段名和预期值，不拼接 Key。

- [ ] **Step 7: 运行 CLI 单元测试并确认绿灯**

Run:

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent \
  python -m pytest tests/test_elasticsearch_initialization.py -q
```

Expected: all passed。

- [ ] **Step 8: 运行相关回归测试**

Run:

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent \
  python -m pytest \
  tests/test_elasticsearch_store.py \
  tests/test_elasticsearch_initialization.py \
  tests/test_elasticsearch_retrievers.py \
  -q
```

Expected: all passed。

- [ ] **Step 9: 提交 CLI 和测试**

```bash
git add 01_RAG/rag/init_elasticsearch.py \
  01_RAG/tests/test_elasticsearch_initialization.py
git commit -m "01_RAG 增加 ES 空库初始化命令"
```

---

## Task 3: 补充配置模板和教学文档

**Files:**

- Modify: `01_RAG/.env.example`
- Modify: `01_RAG/README.md`

- [ ] **Step 1: 更新 `.env.example` 的固定配置**

保留空 Key，不放真实值：

```dotenv
DASHSCOPE_API_KEY=
EMBEDDING_MODEL=qwen3.7-text-embedding
ES_EMBEDDING_DIMS=1024
ES_PHYSICAL_INDEX=rag-child-chunks-v1
ES_INDEX_READ_ALIAS=rag-child-chunks-read
ES_INDEX_WRITE_ALIAS=rag-child-chunks-write
ES_NUMBER_OF_SHARDS=1
ES_NUMBER_OF_REPLICAS=0
```

- [ ] **Step 2: 在 README 增加“初始化全新 Elasticsearch”章节**

必须说明：

- 启动本机 ES：

  ```bash
  cd ~/Downloads/elasticsearch-8
  ./bin/elasticsearch
  ```

- 在 `01_RAG/.env` 配置真实 Key、固定模型和 1024 维。
- 安装项目已声明的 ES client 依赖。
- 运行 `python -m rag.init_elasticsearch`。
- 三种幂等状态的含义。
- 重复执行不会摄取、删除或迁移数据。
- 维度冲突、别名冲突只会失败，不自动修复数据结构。
- 只有 `*.env.example` 允许进入后续提交。

- [ ] **Step 3: 校验文档没有 Secret 或错误模型名**

Run:

```bash
rg -n "qwen3\\.7-text-embedding|ES_EMBEDDING_DIMS|init_elasticsearch" \
  01_RAG/README.md 01_RAG/.env.example
rg -n "DASHSCOPE_API_KEY=.+[^=[:space:]]" \
  01_RAG/README.md 01_RAG/.env.example
```

Expected: 第一条能看到固定模型、维度和命令；第二条无真实值匹配。

- [ ] **Step 4: 提交配置示例和文档**

```bash
git add 01_RAG/.env.example 01_RAG/README.md
git commit -m "01_RAG 文档化 ES 空库初始化"
```

---

## Task 4: 在真实本机 ES 上初始化并证明幂等

**Files:**

- No repository file changes

- [ ] **Step 1: 检查运行环境依赖**

Run:

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent \
  python -c "import elasticsearch; print(elasticsearch.__version__)"
```

若仅因 `ModuleNotFoundError` 失败，安装仓库已经声明的版本范围：

```bash
conda run -n langgraph-cc-multiagent \
  python -m pip install "elasticsearch>=8.19,<9"
```

安装是本地 conda 环境变更，不提交 site-packages。若网络或环境写入需要授权，先取得工具审批。

- [ ] **Step 2: 只输出布尔值和公开配置的运行时预检**

Run:

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent python -c \
  "from config import llm_config, elasticsearch_config; print('dashscope_key_configured=' + str(bool(llm_config.DASHSCOPE_API_KEY))); print('embedding_model=' + llm_config.EMBEDDING_MODEL); print('embedding_dims=' + str(elasticsearch_config.EMBEDDING_DIMS))"
```

Expected:

```text
dashscope_key_configured=True
embedding_model=qwen3.7-text-embedding
embedding_dims=1024
```

- [ ] **Step 3: 在初始化前记录旧索引集合**

Run:

```bash
curl -sS 'http://127.0.0.1:9200/_cat/indices?h=index&format=json'
```

保存输出到终端证据中，不写入仓库。确认 ES 版本可访问，且后续不删除任何既有索引。

- [ ] **Step 4: 第一次执行正式初始化**

Run:

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent \
  python -m rag.init_elasticsearch
```

Expected: `status=created`；若此前已由同一正式命令成功创建，则允许 `status=already_initialized` 或 `aliases_repaired`，但必须通过最终回读。

- [ ] **Step 5: 第二次执行并验证 no-op**

Run:

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent \
  python -m rag.init_elasticsearch
```

Expected: `status=already_initialized`，文档数仍为 0。

- [ ] **Step 6: 回读 Mapping、aliases 和 count**

Run:

```bash
curl -sS \
  'http://127.0.0.1:9200/rag-child-chunks-v1/_mapping'
curl -sS \
  'http://127.0.0.1:9200/_cat/aliases/rag-child-chunks-*?v'
curl -sS \
  'http://127.0.0.1:9200/rag-child-chunks-v1/_count'
```

Expected:

- `embedding.type` 是 `dense_vector`。
- `embedding.dims` 是 `1024`。
- 两个 alias 都指向 `rag-child-chunks-v1`。
- write alias 的 `is_write_index` 为 `true`。
- `_count.count` 是 `0`。

- [ ] **Step 7: 对比旧索引仍然存在**

再次执行 `_cat/indices`，确认 Task 4 Step 3 中的既有索引集合仍是当前集合的子集。

---

## Task 5: 最终回归与提交边界审计

**Files:**

- No new files expected

- [ ] **Step 1: 运行 `01_RAG` 完整离线测试**

Run:

```bash
cd 01_RAG
conda run -n langgraph-cc-multiagent \
  python -m pytest tests -q -m "not es_integration and not real_eval"
```

Expected: all selected tests passed；无外部 API Key 需求。

- [ ] **Step 2: 检查工作区和提交内容**

Run:

```bash
git status --short --branch
git log --oneline --decorate -8
git diff origin/main...HEAD --name-only
```

确认：

- 实现提交都在 `main`。
- 没有提交 `01_RAG/.env` 或任何真实 `.env`。
- 除计划列出的文件外没有意外改动。
- 用户本地 `.env` 内容未被覆盖。

- [ ] **Step 3: 记录最终证据**

最终交付中列出：

- 两次 CLI 状态。
- 正式索引名、1024 维、aliases 和空文档数。
- 单元/回归测试命令与结果。
- 明确说明旧索引未迁移、未删除。
