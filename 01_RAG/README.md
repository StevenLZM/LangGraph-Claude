# 🧠 智能知识库问答系统 (RAG)

> **项目编号 01** | LangChain + ChromaDB + Streamlit | 混合检索 · 多轮对话 · 来源可追溯

---

## 🎯 项目亮点

| 特性 | 技术实现 |
|------|---------|
| **混合检索（Hybrid RAG）** | 语义检索（Dense）+ BM25（Sparse）+ RRF 融合，准确率提升 15-20% |
| **问题改写（Query Rewriting）** | 多轮对话中消解代词，用小模型省成本 |
| **来源可追溯** | 每条答案标注原始文档名 + 页码 |
| **增量索引** | 新增/删除文档无需重建全量索引 |
| **MCP 集成** | 通过 MCP Filesystem Server 管理文档 |
| **生产级架构** | 配置中心 · 单例向量库 · 会话隔离 · 完整测试套件 |

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
│   ├── vectorstore.py        # ChromaDB 管理（增删改查）
│   ├── retriever.py          # 混合检索器（语义 + BM25 + RRF）
│   └── chain.py              # LCEL RAG Chain（问题改写 → 检索 → 生成）
├── memory/
│   └── session.py            # 会话记忆管理（多会话隔离 + 自动裁剪）
├── mcp/
│   └── filesystem_client.py  # MCP Filesystem Client 适配层
├── data/
│   ├── documents/            # PDF 存储目录
│   └── vectorstore/          # ChromaDB 持久化目录
└── tests/
    └── test_rag_pipeline.py  # 完整测试套件（无 API 单元测试 + 集成测试）
```

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

### 第三步：生成示例 PDF（可选）

```bash
python generate_sample_pdfs.py
# 生成两份示例文档到 data/documents/
```

### 第四步：启动应用

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
┌──────────────┬──────────────┐
│  语义检索    │   BM25检索   │
│  Top-K=6    │   Top-K=6    │
└──────┬───────┴──────┬───────┘
       └──────┬────────┘
              ▼ RRF 融合排序 → Top-4
    │
    ▼ (3) 相似度阈值过滤
去除低质量检索结果（默认 threshold=0.3）
    │
    ▼ (4) LLM 生成
System Prompt + Context + History + Question
    │
    ▼ (5) 答案后处理
提取来源 metadata → 格式化展示
```

### 混合检索（Hybrid Retrieval）

```
Dense Retrieval          Sparse Retrieval (BM25)
(向量相似度)              (关键词匹配)
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
| `SEMANTIC_TOP_K` | `6` | 语义检索召回数 |
| `BM25_TOP_K` | `6` | BM25 检索召回数 |
| `FINAL_TOP_K` | `4` | 最终使用的文档块数 |
| `SEMANTIC_WEIGHT` | `0.6` | 语义检索权重（BM25=0.4） |
| `SIMILARITY_THRESHOLD` | `0.3` | 相似度过滤阈值 |

---

## 🧪 运行测试

```bash
# 运行所有不需要 API Key 的测试
pytest tests/ -v -k "not slow"

# 运行需要真实 API 的集成测试（需配置 .env）
pytest tests/ -v -m slow

# 查看测试覆盖率
pytest tests/ --cov=rag --cov=memory --cov=mcp --cov-report=term-missing
```

### 离线检索评测

```bash
# 验证评测管道和报告生成，不访问向量库或 LLM
python -m evals.run --dry-run

# 只评检索层：Recall@K、MRR、nDCG、Parent Hit、时间过滤准确率
python -m evals.run

# 同时评生成层：答案关键词、引用来源、拒答与禁用词
python -m evals.run --with-generation

# 发布前可选：增加 LLM Judge 语义评分
python -m evals.run --with-generation --with-judge
```

评测结果写入 `evals/results/<run_id>/`，包含 `results.jsonl`、`summary.json` 和 `REPORT.md`。

---

## 💡 面试亮点总结

1. **混合检索而非单一向量检索**：语义 + BM25 + RRF，这是大厂实际生产中的标准做法
2. **Query Rewriting**：多轮对话的关键技术，用小模型消解代词节省成本
3. **相似度阈值过滤**：防止低质量检索结果污染 LLM 输入，有效降低幻觉
4. **幂等索引**：文档重新上传时先删旧版本再插入，保证数据一致性
5. **MCP 集成**：体现对 Claude Agent 技术栈的理解
6. **离线评测体系**：用人工金标样本量化 Recall、MRR、Parent Hit 和生成引用质量
7. **测试驱动**：核心模块均有单元测试，无需 API Key 即可验证
