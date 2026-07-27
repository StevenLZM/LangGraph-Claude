# 01_RAG 全新 Elasticsearch 索引初始化设计

## 目标

为 `01_RAG` 初始化一套全新的空 Elasticsearch Child Chunk 索引，作为后续文档摄取的正式数据库。

本次只创建索引结构和别名：

- 不迁移旧 Elasticsearch 数据。
- 不迁移 SQLite Parent Store 数据。
- 不扫描或摄取 `data/documents/`。
- 不删除现有 `products`、`products_vec`、`rag-chunks` 等索引。
- 不读取、输出或提交 DashScope API Key。

## 固定配置

运行环境：

```text
conda environment = langgraph-cc-multiagent
```

Embedding 配置：

```text
provider = DashScope
model = qwen3.7-text-embedding
dense dimension = 1024
output type = dense
```

Elasticsearch 配置：

```text
physical index = rag-child-chunks-v1
read alias = rag-child-chunks-read
write alias = rag-child-chunks-write
shards = 1
replicas = 0
```

`qwen3.7-text-embedding` 支持多种 Dense 维度，默认值为 1024。本项目固定使用 1024，避免 API 输出和 ES Mapping 产生隐式维度漂移。

## 配置来源

本地 `01_RAG/.env` 必须包含：

```dotenv
DASHSCOPE_API_KEY=<真实百炼 API Key>
EMBEDDING_MODEL=qwen3.7-text-embedding
ES_EMBEDDING_DIMS=1024
```

`.env` 必须保持不被 Git 跟踪。初始化命令只能输出“配置是否有效”，不得输出 Key 内容。

当前实现通过 `config.py` 的 `load_dotenv()` 加载 `01_RAG/.env`，再由：

- `llm_config.DASHSCOPE_API_KEY`
- `llm_config.EMBEDDING_MODEL`
- `elasticsearch_config.EMBEDDING_DIMS`

提供运行配置。

## 方案选择

### 采用：显式初始化命令

增加幂等命令：

```bash
conda run -n langgraph-cc-multiagent \
  python -m rag.init_elasticsearch
```

命令执行顺序：

1. 校验 DashScope Key 已配置且不是示例占位值。
2. 校验模型严格等于 `qwen3.7-text-embedding`。
3. 校验 `ES_EMBEDDING_DIMS` 严格等于 `1024`。
4. 连接 `ES_URL` 并执行 ping。
5. 调用现有 `ElasticsearchChildStore.ensure_index(1024)`。
6. 读取物理索引 Mapping 和别名进行回读验证。
7. 输出索引名、维度、读写别名和当前文档数，不输出 Secret。

采用显式命令而不是应用启动时自动创建，原因是：

- 初始化属于运维动作，不应隐藏在 Streamlit 启动副作用中。
- 错误模型或维度必须在创建前失败。
- 命令可在部署、教学和 CI 环境中重复执行并留下清晰日志。

### 不采用：首次文档上传时延迟创建

现有写入链路会在首次 Bulk 前创建索引，但它无法满足“部署后立即看到空正式索引”的需求，并且配置错误要到向量化后才暴露。

### 不采用：手工执行 Elasticsearch PUT

手工请求容易让 Mapping 与项目代码漂移，也无法复用项目已有的 Mapping、认证和别名配置。

## 幂等与错误处理

- 物理索引不存在：创建索引并同时建立读写别名。
- 索引存在且向量维度为 1024：不重建，仅确保别名存在。
- 索引存在但维度不同：明确失败，不删除、不覆盖。
- 任一别名指向其他索引：明确失败，不自动切换。
- ES 不可用、Key 缺失、模型或维度错误：在任何 ES 写操作前失败。
- 命令不提供 `--force`、删除或重建选项。

## 代码边界

计划增加：

- `rag/init_elasticsearch.py`
  - 配置预检。
  - 调用索引初始化。
  - 回读并输出安全摘要。
- `tests/test_elasticsearch_initialization.py`
  - 配置错误必须在 ES 写入前失败。
  - 空索引初始化结果正确。
  - 重复初始化保持幂等。
  - 维度不一致和别名冲突不会破坏已有数据。

计划调整：

- `rag/elasticsearch_store.py`
  - 对已有别名目标执行冲突检查，禁止静默改绑。
- `README.md`
  - 补充 DashScope、模型、维度和初始化命令。

不修改文档摄取、检索、RRF、Rerank、评测或 Parent Store 业务逻辑。

## 验证

离线验证：

```bash
conda run -n langgraph-cc-multiagent \
  python -m pytest tests/test_elasticsearch_initialization.py -q
```

真实 ES 验证：

```bash
conda run -n langgraph-cc-multiagent \
  python -m rag.init_elasticsearch

curl -sS 'http://127.0.0.1:9200/_cat/indices/rag-child-chunks-v1?v'
curl -sS 'http://127.0.0.1:9200/_cat/aliases/rag-child-chunks-*?v'
```

成功标准：

1. `rag-child-chunks-v1` 存在且文档数为 0。
2. `embedding` Mapping 类型为 `dense_vector`，维度为 1024。
3. 读写别名都指向该物理索引，写别名具有 `is_write_index=true`。
4. 现有旧索引和 SQLite Parent Store 内容保持不变。
5. 重复运行初始化命令不会新增索引、删除数据或改变维度。

## 安全说明

项目当前运行配置仍必须以实际进程读取结果为准，不能只检查 `.env` 文本是否存在。正式初始化前会在 `langgraph-cc-multiagent` 环境中再次确认：

```text
DashScope Key configured = true
Embedding model = qwen3.7-text-embedding
Embedding dims = 1024
```

如果任一值不符合，初始化停止。
