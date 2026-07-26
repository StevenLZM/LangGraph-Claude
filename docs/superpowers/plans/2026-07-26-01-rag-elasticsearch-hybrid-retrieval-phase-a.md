# 01_RAG Elasticsearch 混合检索阶段 A 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:test-driven-development` task by task. Use `superpowers:verification-before-completion` before claiming completion. Do not start stage B from this plan.

**Goal:** 将 `01_RAG` 的 Milvus Lite Dense 检索和进程内 BM25 替换为现有 Elasticsearch 8.19，同时保留父子 Chunk、应用层 Weighted RRF、SQLite Parent DocStore、Cross-Encoder 和现有 Chain 调用边界。

**Architecture:** 新增基于官方 `elasticsearch` Python Client 的 Child Store。Child 文本、向量和 Metadata 写入一个版本化 ES 索引；BM25 与 kNN 使用相同 Filter。`rag/vectorstore.py` 保留现有公开函数作为兼容 Facade，`rag/retriever.py` 继续负责应用层 RRF 和 Parent Hydration。

**Tech Stack:** Python 3.11+、Elasticsearch 8.19、官方 `elasticsearch` Python Client、LangChain、SQLite、pytest。

**Design reference:** `docs/superpowers/specs/2026-07-26-01-rag-elasticsearch-hybrid-retrieval-design.md`

---

## 实施边界

本计划只完成阶段 A：

- Elasticsearch BM25 Top K。
- Elasticsearch Dense kNN Top K。
- 同一个 Metadata Filter 下推到两条召回路径。
- 应用层 Weighted RRF，以 `child_id` 为逻辑身份。
- Child 召回、Parent Hydration。
- 版本化 `storage_id`、staging/active 写入和失败补偿。
- 移除 Milvus Lite 与进程内 BM25 运行依赖。
- 更新 Streamlit 错误信息、配置和文档。

本计划不实现：

- 真实认证系统和完整 ACL。
- 新的实体识别模型。
- Cross-Encoder 阈值校准和业务特征打分。
- 相似 Parent 去重、每文档配额、MMR。
- Token Budget Assembler 和生成后引用校验。
- 对象存储、PostgreSQL/MySQL、多节点 ES。

这些能力属于设计文档中的阶段 B/C，必须在阶段 A 建立质量和延迟基线后另立计划。

## 工作区保护

- 当前仓库存在用户修改：`05_PRODUCT_AGENT/.env`。
- 每次 `git add` 必须使用明确的 `01_RAG/...` 或本文档路径。
- 不得运行 `git add .`。
- 不得修改用户现有索引 `products_vec`、`products`、`rag-chunks`。
- ES 集成测试只能操作带随机后缀的 `rag-test-*` 索引。

---

### Task 1: 建立迁移前基线

**Files:**

- No production file changes.
- Read: `01_RAG/evals/dataset.jsonl`
- Read: `01_RAG/evals/run.py`

- [ ] **Step 1: 确认当前工作区和 ES**

Run:

```bash
git status --short
curl -sS http://127.0.0.1:9200
curl -sS "http://127.0.0.1:9200/_cluster/health?pretty"
```

Expected:

- `05_PRODUCT_AGENT/.env` 仍是唯一的用户修改。
- ES 返回 `version.number=8.19.0`。
- 所有 primary shard active；单节点已有索引可能仍为 yellow。

- [ ] **Step 2: 运行离线测试基线**

Run:

```bash
cd 01_RAG
pytest tests/ -q -k "not slow"
python -m evals.run --dry-run
```

Expected: 记录准确的通过、失败、跳过数量。已有失败不能在迁移后被误报为新回归。

- [ ] **Step 3: 可选地记录真实检索基线**

仅在当前 Milvus Lite 可正常打开且 Embedding API 可用时运行：

```bash
cd 01_RAG
python -m evals.run --rerank-disabled --run-id pre-es-milvus-baseline
python -m evals.run --rerank-enabled --run-id pre-es-milvus-rerank
```

Expected: 生成 Recall、MRR、Hit 和 RAGAS 基线。环境不满足时记录原因，不伪造结果。

---

### Task 2: Elasticsearch 配置与依赖边界

**Files:**

- Modify: `01_RAG/tests/test_rag_pipeline.py`
- Modify: `01_RAG/config.py`
- Modify: `01_RAG/requirements.txt`
- Modify: `01_RAG/.env.example`

- [ ] **Step 1: 写失败的 ElasticsearchConfig 测试**

在 `TestConfig` 中删除 Milvus URI 兼容测试，增加：

```python
def test_elasticsearch_config_matches_local_es_8_defaults(self):
    from config import ElasticsearchConfig

    assert ElasticsearchConfig.URL == "http://127.0.0.1:9200"
    assert ElasticsearchConfig.PHYSICAL_INDEX == "rag-child-chunks-v1"
    assert ElasticsearchConfig.READ_ALIAS == "rag-child-chunks-read"
    assert ElasticsearchConfig.WRITE_ALIAS == "rag-child-chunks-write"
    assert ElasticsearchConfig.NUMBER_OF_SHARDS == 1
    assert ElasticsearchConfig.NUMBER_OF_REPLICAS == 0
    assert ElasticsearchConfig.VERIFY_CERTS is False


def test_elasticsearch_config_has_production_auth_fields(self):
    from config import ElasticsearchConfig

    assert hasattr(ElasticsearchConfig, "USERNAME")
    assert hasattr(ElasticsearchConfig, "PASSWORD")
    assert hasattr(ElasticsearchConfig, "API_KEY")
    assert hasattr(ElasticsearchConfig, "CA_CERTS")
```

同时更新 `test_rag_config_defaults`：

```python
assert RAGConfig.SEMANTIC_TOP_K == 50
assert RAGConfig.BM25_TOP_K == 50
assert RAGConfig.RRF_TOP_K == 80
assert RAGConfig.RRF_K == 60
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
cd 01_RAG
pytest tests/test_rag_pipeline.py::TestConfig -q
```

Expected: FAIL，因为 `ElasticsearchConfig` 和新的 RRF 配置尚不存在。

- [ ] **Step 3: 实现最小配置**

在 `config.py`：

- 删除 `_consume_legacy_milvus_uri()`、`_resolve_milvus_uri()`、`MilvusConfig` 和 `milvus_config`。
- 新增 `ElasticsearchConfig` 与 `elasticsearch_config`。
- 增加以下配置：

```text
ES_URL
ES_PHYSICAL_INDEX
ES_INDEX_READ_ALIAS
ES_INDEX_WRITE_ALIAS
ES_NUMBER_OF_SHARDS
ES_NUMBER_OF_REPLICAS
ES_VERIFY_CERTS
ES_USERNAME
ES_PASSWORD
ES_API_KEY
ES_CA_CERTS
ES_REQUEST_TIMEOUT
ES_MAX_RETRIES
ES_BULK_CHUNK_SIZE
ES_EMBEDDING_DIMS
ES_DENSE_NUM_CANDIDATES
```

`ES_EMBEDDING_DIMS` 默认设为 `0`，表示第一次写入时根据实际向量确定 Mapping；如果用户显式配置非零值，写入前必须校验。

更新 RAG 默认值：

```text
SEMANTIC_TOP_K=50
BM25_TOP_K=50
RRF_TOP_K=80
RRF_K=60
MAX_HYDRATED_PARENTS=6
```

- [ ] **Step 4: 更新依赖**

在 `requirements.txt`：

- 删除 `langchain-milvus`。
- 删除 `pymilvus[milvus-lite]`。
- 删除只为进程内 BM25 使用的 `rank-bm25` 和 `jieba`。
- 增加 `elasticsearch>=8.19,<9`。

- [ ] **Step 5: 更新 `.env.example`**

删除 Milvus 路径，增加本地 ES 配置和候选数配置。不要写入真实密码或 API Key。

- [ ] **Step 6: 运行测试确认通过**

Run:

```bash
cd 01_RAG
pytest tests/test_rag_pipeline.py::TestConfig -q
```

Expected: PASS。

- [ ] **Step 7: 提交**

```bash
git add 01_RAG/config.py 01_RAG/requirements.txt 01_RAG/.env.example 01_RAG/tests/test_rag_pipeline.py
git commit -m "01_RAG configure Elasticsearch 8 backend"
```

---

### Task 3: 实现 Elasticsearch Child Store

**Files:**

- Create: `01_RAG/rag/elasticsearch_store.py`
- Create: `01_RAG/tests/test_elasticsearch_store.py`

- [ ] **Step 1: 写 Client 和 Mapping 失败测试**

测试必须使用 Fake Client，不连接真实 ES：

```python
def test_create_client_uses_local_url_without_auth(monkeypatch):
    ...
    assert captured["hosts"] == ["http://127.0.0.1:9200"]
    assert "basic_auth" not in captured


def test_ensure_index_creates_versioned_index_and_aliases():
    ...
    assert body["settings"]["number_of_replicas"] == 0
    assert body["mappings"]["properties"]["content"]["analyzer"] == "cjk"
    assert body["mappings"]["properties"]["embedding"]["dims"] == 3
    assert body["mappings"]["properties"]["embedding"]["similarity"] == "cosine"


def test_ensure_index_rejects_embedding_dimension_mismatch():
    ...
    with pytest.raises(ValueError, match="向量维度"):
        store.ensure_index(vector_dims=4)
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
cd 01_RAG
pytest tests/test_elasticsearch_store.py -q
```

Expected: FAIL，因为模块尚不存在。

- [ ] **Step 3: 实现 Client 工厂**

`create_elasticsearch_client()` 规则：

- URL 来自配置。
- 有 API Key 时使用 `api_key`。
- 否则用户名和密码同时存在时使用 `basic_auth`。
- 本地无认证时不传空认证元组。
- 生产配置 CA 时传 `ca_certs`。
- 透传 `verify_certs`、timeout、retry_on_timeout 和 max_retries。

- [ ] **Step 4: 实现索引初始化**

`ElasticsearchChildStore.ensure_index(vector_dims)`：

- 如果读/写别名已存在，读取 Mapping 并校验向量维度。
- 如果物理索引不存在，使用明确 Mapping 创建。
- 创建后绑定 read/write alias。
- 不覆盖同名现有索引。
- Mapping 至少包含设计文档第 10 节字段。
- 本地 `number_of_shards=1`、`number_of_replicas=0`。

- [ ] **Step 5: 写 Filter Builder 失败测试**

```python
def test_build_es_filters_supports_date_range_and_status():
    filters = build_es_filters({
        "$and": [
            {"status": "active"},
            {"has_doc_date": True},
            {"doc_date_min": {"$lte": 20241231}},
            {"doc_date_max": {"$gte": 20240101}},
        ]
    })

    assert {"term": {"status": "active"}} in filters
    assert {"term": {"has_doc_date": True}} in filters
    assert {"range": {"doc_date_min": {"lte": 20241231}}} in filters
    assert {"range": {"doc_date_max": {"gte": 20240101}}} in filters


def test_build_es_filters_rejects_unknown_operator():
    with pytest.raises(ValueError):
        build_es_filters({"doc_id": {"$regex": ".*"}})
```

- [ ] **Step 6: 实现严格 Filter 翻译**

支持当前项目需要的：

- equality / inequality。
- `$gt`、`$gte`、`$lt`、`$lte`。
- `$in`。
- `$and`、`$or`。

字段名必须经过白名单校验，禁止用户输入任意 ES 字段。

- [ ] **Step 7: 实现 Hit 与 Document 转换**

提供：

```python
document_to_source(doc, vector, *, storage_id, status, ingest_run_id) -> dict
hit_to_document(hit, retrieval_source) -> Document
```

返回的 `Document.metadata` 至少保留：

- `child_id`、`parent_id`、`doc_id`、`doc_version`。
- 来源和页码。
- `retrieval_source`。
- ES 原始 `_score`，仅用于追踪，不与另一路原始分数直接相加。

- [ ] **Step 8: 运行测试并提交**

Run:

```bash
cd 01_RAG
pytest tests/test_elasticsearch_store.py -q
```

Expected: PASS。

Commit:

```bash
git add 01_RAG/rag/elasticsearch_store.py 01_RAG/tests/test_elasticsearch_store.py
git commit -m "01_RAG add Elasticsearch child store"
```

---

### Task 4: 实现版本化写入、激活与补偿

**Files:**

- Modify: `01_RAG/rag/elasticsearch_store.py`
- Modify: `01_RAG/rag/vectorstore.py`
- Modify: `01_RAG/rag/docstore.py`
- Modify: `01_RAG/tests/test_elasticsearch_store.py`
- Modify: `01_RAG/tests/test_rag_pipeline.py`
- Modify: `01_RAG/tests/test_chunking_v2.py`

- [ ] **Step 1: 写版本化存储身份测试**

```python
def test_storage_id_includes_document_version():
    assert build_storage_id(
        doc_id="doc-1",
        doc_version="v2",
        child_id="child-9",
    ) == "doc-1:v2:child-9"
```

增加测试证明相同 `child_id` 在两个版本中的 `storage_id` 不同。

- [ ] **Step 2: 写 Parent 版本清理测试**

为 `ParentDocStore` 增加并测试：

```python
delete_document_version(doc_id, doc_version)
delete_versions_except(doc_id, active_doc_version)
```

测试使用 `tmp_path` SQLite，不能操作真实 Parent DB。

- [ ] **Step 3: 写 Bulk 生命周期测试**

Fake Store 记录调用顺序：

```text
upsert parents
bulk staging children
activate new version
deactivate old versions
delete old parent versions
```

另写失败测试：

- Bulk 单个 Item 失败时抛出异常。
- 删除本次 `ingest_run_id` staging Child。
- 删除本次新 Parent 版本。
- 旧 active ES 版本不被删除。

- [ ] **Step 4: 运行测试确认失败**

Run:

```bash
cd 01_RAG
pytest tests/test_elasticsearch_store.py tests/test_rag_pipeline.py tests/test_chunking_v2.py -q
```

Expected: 新生命周期测试 FAIL。

- [ ] **Step 5: 实现 Store 写操作**

`ElasticsearchChildStore` 增加：

```text
bulk_stage_children(...)
activate_version(doc_id, doc_version, ingest_run_id)
deactivate_other_versions(doc_id, active_doc_version)
delete_ingest_run(ingest_run_id)
delete_document(doc_id)
count_children()
aggregate_documents()
```

Bulk 必须逐项检查 `errors`，不能只判断 HTTP 状态。

- [ ] **Step 6: 将 `vectorstore.py` 改为兼容 Facade**

保留公开函数名：

```text
get_vectorstore
add_documents
delete_document
list_documents
get_collection_stats
build_time_filter
similarity_search_with_threshold
```

删除：

- Milvus import 和 ORM connection workaround。
- Milvus filter 翻译。
- 全量扫描 Child 构建 BM25 的 `get_all_child_documents()`。

`add_documents()` 使用 `embed_with_retry()` 生成 Child 向量，根据第一条向量确定或校验 Mapping dims。

- [ ] **Step 7: 修正删除顺序**

`delete_document()`：

1. 先使 ES Child 不可检索或删除。
2. ES 成功后删除 Parent。
3. ES 删除失败时抛出可观察错误，不静默删除 Parent。

这与当前捕获所有异常后继续删除 Parent 的行为不同，测试必须锁定新语义。

- [ ] **Step 8: 运行测试并提交**

Run:

```bash
cd 01_RAG
pytest tests/test_elasticsearch_store.py tests/test_rag_pipeline.py tests/test_chunking_v2.py -q
```

Expected: PASS。

Commit:

```bash
git add 01_RAG/rag/elasticsearch_store.py 01_RAG/rag/vectorstore.py 01_RAG/rag/docstore.py 01_RAG/tests/test_elasticsearch_store.py 01_RAG/tests/test_rag_pipeline.py 01_RAG/tests/test_chunking_v2.py
git commit -m "01_RAG index versioned child chunks in Elasticsearch"
```

---

### Task 5: Elasticsearch BM25、Dense 和应用层 RRF

**Files:**

- Create: `01_RAG/rag/elasticsearch_retrievers.py`
- Create: `01_RAG/tests/test_elasticsearch_retrievers.py`
- Modify: `01_RAG/rag/retriever.py`
- Modify: `01_RAG/tests/test_chunking_v2.py`
- Modify: `01_RAG/tests/test_rag_pipeline.py`

- [ ] **Step 1: 写 BM25 Query 测试**

验证 Query Body 包含：

```text
match content
status=active
时间 Filter
size=BM25_TOP_K
```

测试同时断言返回 `Document.metadata["retrieval_source"] == "bm25"`。

- [ ] **Step 2: 写 Dense kNN 测试**

注入 Fake Embeddings，并验证：

```python
assert knn["field"] == "embedding"
assert knn["k"] == 50
assert knn["num_candidates"] == 300
assert knn["filter"] == expected_shared_filters
```

返回结果标记 `retrieval_source="dense"`。

- [ ] **Step 3: 写统一 Filter 测试**

同一个 `metadata_filter` 输入必须在 BM25 bool filter 和 kNN filter 中生成等价条件。至少覆盖日期区间与 `status=active`。

- [ ] **Step 4: 写单路降级测试**

- Dense Embedding 抛错时，BM25 结果仍可进入融合。
- BM25 抛错时，Dense 结果仍可进入融合。
- 两路都失败时返回空列表并记录错误，不抛出 Milvus 相关信息。

- [ ] **Step 5: 运行测试确认失败**

Run:

```bash
cd 01_RAG
pytest tests/test_elasticsearch_retrievers.py -q
```

Expected: FAIL，因为 Retriever 尚不存在。

- [ ] **Step 6: 实现两个 BaseRetriever**

新增：

```python
class ElasticsearchBM25Retriever(BaseRetriever): ...
class ElasticsearchDenseRetriever(BaseRetriever): ...
class SafeRetriever(BaseRetriever): ...
```

两个 Retriever 共享同一个 `ElasticsearchChildStore` 和已经构造好的 Filter，不允许各自解释时间条件。

- [ ] **Step 7: 改造 `build_hybrid_retriever()`**

删除：

- `BM25Retriever.from_documents()`。
- `_RetrieverBase` 和 `_base_cache`。
- `TimeFilteredBM25Wrapper`。
- Milvus `as_retriever()` 和 expr。

构造：

```python
ensemble = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[semantic_weight, 1 - semantic_weight],
    c=rag_config.RRF_K,
    id_key="child_id",
)
```

如果当前已安装的 `langchain-classic` 不支持 `id_key` 或 `c`，先写兼容性测试，再实现小型 `WeightedRRFFusion`；不得退回按全文内容去重。

- [ ] **Step 8: 限制 RRF Window 并 Hydrate Parent**

`ParentChildHybridRetriever.invoke()`：

1. 获取融合 Child。
2. 截断到 `RRF_TOP_K`。
3. 调用现有 `hydrate_parent_results()`。
4. 最多加载 `MAX_HYDRATED_PARENTS`。

阶段 A 保留当前“Parent Hydration 后 Cross-Encoder”行为，以控制迁移范围。Cross-Encoder 前移到 Child、阈值和业务特征属于阶段 B。

- [ ] **Step 9: 移除全量 Child 加载**

`get_hybrid_retriever()` 只检查 ES active Child count，不读取所有 Child。增加测试：

```python
def test_get_hybrid_retriever_never_scans_all_children(...):
    ...
```

- [ ] **Step 10: 运行测试并提交**

Run:

```bash
cd 01_RAG
pytest tests/test_elasticsearch_retrievers.py tests/test_chunking_v2.py tests/test_rag_pipeline.py -q
```

Expected: PASS。

Commit:

```bash
git add 01_RAG/rag/elasticsearch_retrievers.py 01_RAG/rag/retriever.py 01_RAG/tests/test_elasticsearch_retrievers.py 01_RAG/tests/test_chunking_v2.py 01_RAG/tests/test_rag_pipeline.py
git commit -m "01_RAG retrieve BM25 and dense candidates from Elasticsearch"
```

---

### Task 6: Chain、Streamlit 和公开文档兼容

**Files:**

- Modify: `01_RAG/rag/chain.py`
- Modify: `01_RAG/app.py`
- Modify: `01_RAG/tests/test_app_startup.py`
- Modify: `01_RAG/README.md`

- [ ] **Step 1: 写 Streamlit 启动测试**

替换 Milvus lock 测试，验证：

- 初始页面仍可只从 Parent DocStore 列文档，不强制连接 ES。
- ES 统计失败时返回 `backend="elasticsearch"` 和可读错误。
- Chain 初始化失败时提示检查 `http://127.0.0.1:9200`，不再提示 Milvus 文件锁。

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
cd 01_RAG
pytest tests/test_app_startup.py -q
```

Expected: 新断言 FAIL。

- [ ] **Step 3: 移除 Milvus UI 逻辑**

在 `app.py`：

- 导入 `elasticsearch_config`，不再导入 `milvus_config`。
- 删除 `fcntl` 和 `_detect_milvus_lock()`。
- 将后台名称和错误信息改为 Elasticsearch。
- 上传进度文案改为“向量化并写入 Elasticsearch”。
- 保留现有上传、删除和 Chain 重建调用方式。

- [ ] **Step 4: 保持 Chain fallback API**

`chain.py` 继续通过：

```text
get_vectorstore
get_hybrid_retriever
retrieve_with_hybrid
similarity_search_with_threshold
```

调用检索，不在 Chain 内拼 ES DSL。

阶段 A 不改变 Prompt 引用格式；严格 `[Sx]` 引用校验属于阶段 B。

- [ ] **Step 5: 更新 README**

更新：

- 技术栈和架构图。
- ES 启动命令：

```bash
cd ~/Downloads/elasticsearch-8
./bin/elasticsearch
```

- `curl http://127.0.0.1:9200` 健康检查。
- 配置表。
- 重新摄取说明。
- 删除 Milvus 文件锁、Chroma/Milvus 迁移遗留描述。

不得声称阶段 B/C 功能已经实现。

- [ ] **Step 6: 运行测试并提交**

Run:

```bash
cd 01_RAG
pytest tests/test_app_startup.py tests/test_rag_pipeline.py tests/test_chunking_v2.py -q
```

Expected: PASS。

Commit:

```bash
git add 01_RAG/rag/chain.py 01_RAG/app.py 01_RAG/tests/test_app_startup.py 01_RAG/README.md
git commit -m "01_RAG update app for Elasticsearch retrieval"
```

---

### Task 7: 真实 Elasticsearch 8.19 集成测试

**Files:**

- Create: `01_RAG/tests/test_elasticsearch_integration.py`
- Modify: `01_RAG/pytest.ini` if it exists; otherwise create it only if marker registration is needed.

- [ ] **Step 1: 写隔离的集成测试**

测试要求：

- 仅当 `RUN_ES_INTEGRATION=1` 时运行，否则 skip。
- 每次创建 `rag-test-<uuid>` 物理索引和对应测试别名。
- 使用三维固定向量，不调用外部 Embedding API。
- 在 `finally` 中只删除本次随机测试索引。

覆盖：

1. 创建 Mapping 和 alias。
2. Bulk 写入 active Child。
3. BM25 召回精确关键词。
4. kNN 召回语义近邻。
5. 相同时间 Filter 对两路结果一致。
6. staging Child 不可召回。
7. 删除测试 `doc_id` 后不再召回。

- [ ] **Step 2: 运行集成测试**

确认用户的 ES 正在运行后：

```bash
cd 01_RAG
RUN_ES_INTEGRATION=1 pytest tests/test_elasticsearch_integration.py -q
```

Expected: PASS，且 `_cat/indices` 中不存在遗留 `rag-test-*`。

- [ ] **Step 3: 运行离线全套测试**

```bash
cd 01_RAG
pytest tests/ -q -k "not slow"
python -m evals.run --dry-run
```

Expected: PASS；如有环境失败，保留完整错误文本并先修复根因。

- [ ] **Step 4: 提交**

```bash
git add 01_RAG/tests/test_elasticsearch_integration.py 01_RAG/pytest.ini
git commit -m "01_RAG test Elasticsearch hybrid retrieval integration"
```

如果没有创建或修改 `pytest.ini`，不得把它写入 `git add`。

---

### Task 8: 迁移验证和质量 Gate

**Files:**

- Modify only if necessary: `01_RAG/evals/`
- Modify only if necessary: `01_RAG/README.md`

- [ ] **Step 1: 确认运行时无 Milvus/BM25 本地依赖**

Run:

```bash
rg -n "langchain_milvus|pymilvus|MilvusConfig|milvus_config|BM25Retriever|rank_bm25|import jieba|get_all_child_documents" 01_RAG/config.py 01_RAG/rag 01_RAG/app.py 01_RAG/requirements.txt 01_RAG/.env.example 01_RAG/README.md
```

Expected: 生产代码和活跃 README 无匹配。历史设计文档不在检查范围。

- [ ] **Step 2: 检查 ES 索引**

```bash
curl -sS "http://127.0.0.1:9200/_cat/indices/rag-child-chunks-v1?v"
curl -sS "http://127.0.0.1:9200/_cat/aliases/rag-child-chunks-*?v"
curl -sS "http://127.0.0.1:9200/rag-child-chunks-read/_count?pretty"
```

Expected:

- 主分片 active。
- 本地副本为 0。
- read/write alias 指向预期物理索引。

- [ ] **Step 3: 重新摄取测试文档**

通过现有 Streamlit 上传或受控脚本重新摄取。不要尝试读取 Milvus 文件做自动迁移。

验证：

- Child 有 `storage_id`、`child_id`、`parent_id`、`doc_version`。
- active Child 数量与 UI 统计一致。
- Parent SQLite 可以根据命中 `parent_id` 回填。

- [ ] **Step 4: 运行迁移后评测**

```bash
cd 01_RAG
python -m evals.run --rerank-disabled --run-id es-hybrid-baseline
python -m evals.run --rerank-enabled --run-id es-hybrid-rerank
```

比较：

- BM25/Dense/Fusion Recall。
- MRR 和 Hit。
- Cross-Encoder 前后变化。
- 无结果率。
- P50/P95 检索延迟。

如果 Recall 明显低于迁移前基线，优先检查 analyzer、Embedding 维度、`num_candidates`、Filter 和 RRF 身份，不直接提高最终 Top K 掩盖问题。

- [ ] **Step 5: 最终验证**

```bash
cd 01_RAG
pytest tests/ -q -k "not slow"
RUN_ES_INTEGRATION=1 pytest tests/test_elasticsearch_integration.py -q
python -m evals.run --dry-run
```

Expected: 所有适用检查通过。

- [ ] **Step 6: 检查提交范围**

```bash
git status --short
git diff --check
git log --oneline -8
```

Expected:

- `05_PRODUCT_AGENT/.env` 仍未被暂存或提交。
- 没有生成的 ES 数据、模型缓存、测试结果或密钥进入 Git。

---

## 阶段 A 完成 Gate

必须同时满足：

1. BM25 和 Dense 都由 ES 8.19 执行。
2. 不读取全部 Child 构建 BM25。
3. 两路查询使用等价的 `status` 和时间 Filter。
4. Weighted RRF 使用 `child_id`，ES 存储使用版本化 `storage_id`。
5. Parent SQLite Hydration 和现有 Chain API 保持可用。
6. staging 数据不会被查询。
7. 单路检索失败能够降级，ES 整体失败能够明确返回。
8. 所有离线测试和真实 ES 隔离测试通过。
9. 迁移前后效果和延迟完成对比。
10. 用户的 `05_PRODUCT_AGENT/.env` 没有进入任何提交。

达到 Gate 后，再为阶段 B 单独使用 brainstorming + writing-plans：

- Child Cross-Encoder 80 → 15。
- 阈值校准和业务特征。
- Parent 去重、每文档配额和 MMR。
- Token Budget。
- `[Sx]` 引用生成后校验。
