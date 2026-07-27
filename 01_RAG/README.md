# 🧠 智能知识库问答系统 (RAG)

> **项目编号 01** | LangChain + Elasticsearch 8 + Streamlit | 混合检索 · 多轮对话 · 来源可追溯

---

## 🎯 项目亮点

| 特性 | 技术实现 |
|------|---------|
| **混合检索（Hybrid RAG）** | Elasticsearch Dense + BM25，共享 Filter，应用层 Weighted RRF |
| **问题改写（Query Rewriting）** | 多轮对话中消解代词，用小模型省成本 |
| **来源可追溯** | 每条答案标注原始文档名 + 页码 |
| **版本化索引** | staging 写入、原子激活和失败补偿，不覆盖旧的 active 版本 |
| **MCP 集成** | 通过 MCP Filesystem Server 管理文档 |
| **生产级架构** | ES Child 索引 · SQLite Parent Store · 会话隔离 · 完整测试套件 |

---

## 📁 项目结构

```
01_RAG/
├── app.py                    # Streamlit 主程序（生产级 UI）
├── config.py                 # 全局配置中心
├── generate_sample_pdfs.py   # 示例 PDF 生成脚本
├── requirements.txt
├── .env.example              # 环境变量模板
├── .mcp.json                 # MCP Server 配置
├── .streamlit/
│   └── config.toml           # UI 主题配置
├── rag/
│   ├── loader.py             # PDF 解析（PyMuPDF + 备用 pypdf）
│   ├── chunker.py            # 文本分块（Recursive + 语义感知）
│   ├── embedder.py           # Embedding 封装（OpenAI + HuggingFace 兜底）
│   ├── elasticsearch_store.py       # ES Mapping、Filter 和版本化 Child 生命周期
│   ├── elasticsearch_retrievers.py  # ES BM25、Dense 和 Weighted RRF
│   ├── vectorstore.py        # Child/Parent 摄取兼容 Facade
│   ├── docstore.py           # SQLite Parent Store
│   ├── retriever.py          # Child 混合召回与 Parent Hydration
│   └── chain.py              # LCEL RAG Chain（问题改写 → 检索 → 生成）
├── memory/
│   └── session.py            # 会话记忆管理（多会话隔离 + 自动裁剪）
├── mcp/
│   └── filesystem_client.py  # MCP Filesystem Client 适配层
├── data/
│   ├── documents/            # PDF 存储目录
│   └── docstore/             # SQLite Parent Chunk 数据
└── tests/
    └── test_rag_pipeline.py  # 完整测试套件（无 API 单元测试 + 集成测试）
```

### 设计文档索引

- `01_rag_knowledge_base.md`：PRD、功能范围和验收标准
- `01_rag_engineering.md`：系统工程设计、链路拆分和目录结构
- `02_rag_chunking_v2_design.md`：parent-child chunking 设计
- `03_rag_date_aware_retrieval_design.md`：日期感知检索设计
- `04_rag_structured_chunking.md`：结构化切分设计
- `05_rag_ragas_evaluation_design.md`：RAGAS + 传统 IR 评估体系设计

---

## 🚀 快速开始

### 第一步：安装依赖

```bash
cd 01_RAG
pip install -r requirements.txt
```

### 第二步：配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入您的 API Key
```

`.env` 最小配置（二选一）：

```bash
# 方案 A：使用 Claude（推荐）
ANTHROPIC_API_KEY=sk-ant-xxxx
OPENAI_API_KEY=sk-xxxx          # Embedding 仍需要 OpenAI

# 方案 B：仅使用 OpenAI
OPENAI_API_KEY=sk-xxxx          # 同时用于 Embedding 和对话
```

### 第三步：启动本机 Elasticsearch

```bash
cd ~/Downloads/elasticsearch-8
./bin/elasticsearch
```

另开终端确认版本和服务状态：

```bash
curl http://127.0.0.1:9200
```

项目按 Elasticsearch 8.19 和 Basic License 设计，不依赖 Docker 或 Enterprise 原生 RRF。

### 初始化全新 Elasticsearch

在 `01_RAG/` 中先复制模板并编辑 `.env`：`DASHSCOPE_API_KEY` 必须填入您自己的真实 Key；初始化命令固定使用 `EMBEDDING_MODEL=qwen3.7-text-embedding` 与 `ES_EMBEDDING_DIMS=1024`。同时保持物理索引和别名为模板中的固定值。

先启动本机 Elasticsearch：

```bash
cd ~/Downloads/elasticsearch-8
./bin/elasticsearch
```

项目声明的 ES client 依赖包含在 `requirements.txt` 中；在 `01_RAG/` 目录安装后运行初始化命令：

```bash
pip install -r requirements.txt
python -m rag.init_elasticsearch
```

命令只初始化并校验空库，不会摄取文档。输出中的幂等状态含义如下：

- `created`：创建了固定物理索引、1024 维 mapping 和读写别名。
- `aliases_repaired`：已存在且 mapping 正确的物理索引缺少别名，命令补齐了缺失别名。
- `already_initialized`：索引、mapping 和读写别名均已正确存在，无需更改。

可安全重复执行：它不会摄取、删除或迁移任何数据。若发现向量维度冲突或别名指向其他索引等冲突，命令会直接失败，不会自动修复或改写已有数据结构。

`.env` 是本机私密配置；后续提交中只有 `*.env.example` 可以进入版本控制，绝不提交真实 Key。

### LangSmith 审计 Trace（可选）

本项目的 LangSmith Trace 用于审计 RAG 请求、检索和生成过程。默认关闭；在 `.env` 中配置后启用：

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=01-rag-local
```

请在本机 `.env` 的 `LANGSMITH_API_KEY=` 后填入自己的 Key，示例和版本控制中必须保持空值。新配置优先使用 `LANGSMITH_*`，同时兼容旧变量 `LANGCHAIN_TRACING_V2`、`LANGCHAIN_API_KEY` 和 `LANGCHAIN_PROJECT`。

这是显式的审计模式：API key、authorization、password、token、secret、cookie 等凭证字段的值会被隐藏，并统一显示为 `[REDACTED_CREDENTIAL]`。为保留完整审计证据，下列内容**不会**隐藏：问题、历史、完整 Chunk/Parent 正文、证据 context、Prompt、回答、引用、业务 metadata 和分数。

因此，部署方必须自行负责 LangSmith 项目的访问控制和数据保留策略；完整正文也会增加 Trace 存储量和相应成本。关闭 LangSmith，或未配置可用的 LangSmith 凭证时，RAG 主流程不受影响，应用仍可正常运行。

### 第四步：生成示例 PDF（可选）

```bash
python generate_sample_pdfs.py
# 生成两份示例文档到 data/documents/
```

### 第五步：启动应用

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501`

---

## 📐 技术架构

### RAG 完整流程

```
用户问题
    │
    ▼ (1) Query Rewriting
多轮对话中消解代词 → 独立完整问题
    │
    ▼ (2) Hybrid Retrieval
┌──────────────────┬──────────────────┐
│ ES Dense kNN     │ ES BM25          │
│ Top-K=50         │ Top-K=50         │
└──────┬───────┴──────┬───────┘
       └──────┬────────┘
              ▼ Weighted RRF（child_id）→ Top-80
    │
    ▼ (3) Parent Hydration + Cross-Encoder
SQLite 回填 Parent，最终按配置截断
    │
    ▼ (4) LLM 生成
System Prompt + Context + History + Question
    │
    ▼ (5) 答案后处理
提取来源 metadata → 格式化展示
```

### 混合检索（Hybrid Retrieval）

```
ES Dense Retrieval       ES Sparse Retrieval
(kNN/HNSW)               (BM25/CJK analyzer)
     │                         │
     │     RRF Fusion          │
     └─────────┬───────────────┘
               ▼
          融合排序结果
     (权重: 0.6 : 0.4)
```

---

## ⚙️ 配置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CHAT_MODEL` | `claude-sonnet​-4-6` | 主对话模型 |
| `REWRITE_MODEL` | `claude-haiku-4-5-20251001` | 问题改写（用小模型省成本） |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding 模型 |
| `CHUNK_SIZE` | `500` | 每块字符数 |
| `CHUNK_OVERLAP` | `50` | 块间重叠字符数 |
| `SEMANTIC_TOP_K` | `50` | ES Dense 召回数 |
| `BM25_TOP_K` | `50` | ES BM25 召回数 |
| `RRF_TOP_K` | `80` | Weighted RRF 后保留的 Child 数 |
| `RRF_K` | `60` | RRF 排名常数 |
| `FINAL_TOP_K` | `4` | 最终使用的文档块数 |
| `SEMANTIC_WEIGHT` | `0.6` | 语义检索权重（BM25=0.4） |
| `SIMILARITY_THRESHOLD` | `0.3` | 相似度过滤阈值 |
| `ES_URL` | `http://127.0.0.1:9200` | Elasticsearch 地址 |
| `ES_PHYSICAL_INDEX` | `rag-child-chunks-v1` | 版本化物理 Child 索引 |
| `ES_INDEX_READ_ALIAS` | `rag-child-chunks-read` | 线上查询别名 |
| `ES_INDEX_WRITE_ALIAS` | `rag-child-chunks-write` | 摄取写别名 |
| `ES_DENSE_NUM_CANDIDATES` | `300` | kNN ANN 候选窗口 |
| `RERANK_ENABLED` | `true` | 是否启用 Cross-Encoder rerank；Streamlit 应用默认开启，baseline 评测可显式关闭 |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | Cross-Encoder rerank 模型 |
| `RERANK_TOP_N` | `4` | rerank 后保留的候选数 |

> 旧 Chroma/Milvus Lite 数据不会自动导入 ES。请通过现有上传流程重新摄取；Child 写入 ES，Parent 保存在 SQLite。

---

## 🧪 运行测试

```bash
# 运行所有不需要 API Key 的测试
pytest tests/ -v -k "not slow"

# 运行需要真实 API 的集成测试（需配置 .env）
pytest tests/ -v -m slow

# 运行隔离的真实 ES 8.19 测试（只操作随机 rag-test-* 索引）
RUN_ES_INTEGRATION=1 pytest tests/test_elasticsearch_integration.py -q

# 查看测试覆盖率
pytest tests/ --cov=rag --cov=memory --cov=mcp --cov-report=term-missing
```

### 离线评估

```bash
# 开发集：真实 ES、Embedding、Reranker、生成模型和 RAGAS Judge
python -m evals.run --split dev --run-id candidate-v1

# 冻结测试集发布门禁（Test 至少 200 条人工复核 Case）
python -m evals.run --split test --run-id release-v1 \
  --baseline-run evals/results/baseline-v1
```

产品评测没有模拟分数入口。数据集未复核、split 为空、ES 索引未就绪、模型不可用、任一检索阶段缺失或发生降级时，命令会失败且不生成正式报告。

评估结果写入 `evals/results/<run_id>/`。同一次生产链路执行记录 BM25@50、Dense@50、RRF@80、Cross-Encoder@15、业务融合@15、父块多样化@8、最终上下文@6，并在各阶段计算 Recall/NDCG/MRR/Hit；最终证据与答案再进行真实 RAGAS 评测。报告包含 95% bootstrap CI、权限泄漏、拒答/拒绝准确率和 P95 延迟门禁，不计算“RAG 总分”。

数据位于 `evals/datasets/{train,dev,test}.jsonl`，三者物理隔离；qrels 必须人工复核。完整说明见 `05_rag_ragas_evaluation_design.md`。

---

## 💡 面试亮点总结

1. **混合检索而非单一向量检索**：语义 + BM25 + RRF，这是大厂实际生产中的标准做法
2. **Query Rewriting**：多轮对话的关键技术，用小模型消解代词节省成本
3. **相似度阈值过滤**：防止低质量检索结果污染 LLM 输入，有效降低幻觉
4. **幂等索引**：版本化 `storage_id`，先写 staging、再激活新版本、最后下线旧版本
5. **MCP 集成**：体现对 Claude Agent 技术栈的理解
6. **离线评估体系**：用传统 IR 指标评估检索排序，用 RAGAS 评价上下文质量、语义相关性、忠实度和端到端答案正确性
7. **测试驱动**：核心模块均有单元测试，无需 API Key 即可验证
