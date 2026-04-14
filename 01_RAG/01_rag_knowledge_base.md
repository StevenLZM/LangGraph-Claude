# PRD：RAG 知识库问答系统

> 项目编号：01 | 难度：⭐⭐ | 预计周期：2 周

---

## 一、项目背景与目标

### 背景

企业内部积累了大量 PDF 文档（产品手册、合同、规章制度等），员工查找信息效率低下。传统搜索只能关键词匹配，无法理解语义，无法给出完整答案。

### 目标

构建一个基于 RAG（Retrieval-Augmented Generation）的智能问答系统，用户上传 PDF 文档后，可用自然语言提问，系统从文档中检索相关内容并生成准确答案。

### 学习目标

| 技能点 | 掌握内容 |
|--------|----------|
| 文档解析 | PyMuPDF / pdfplumber 解析 PDF |
| 文本分块 | RecursiveCharacterTextSplitter 策略 |
| Embedding | OpenAI / HuggingFace Embedding 模型 |
| 向量数据库 | Chroma 本地存储与检索 |
| RAG Chain | LangChain LCEL 构建检索问答链 |
| 对话记忆 | ConversationBufferMemory 多轮对话 |

---

## 二、用户故事

```
作为一名企业员工
我想上传公司的产品手册 PDF
以便我可以用自然语言直接提问，快速找到答案
而不是在几十页文档里手动翻找
```

---

## 三、功能需求

### 3.1 文档管理

- **F01** 支持上传单个或多个 PDF 文件（≤50MB/个）
- **F02** 上传后自动解析、分块、向量化并存入数据库
- **F03** 显示已上传文档列表，支持删除单个文档
- **F04** 支持增量添加文档，无需重建全量索引

### 3.2 智能问答

- **F05** 用户输入自然语言问题，系统返回答案
- **F06** 答案中标注来源文档名称及页码
- **F07** 支持多轮对话，理解上下文（如"它的价格是多少？"中的"它"）
- **F08** 无相关内容时明确告知用户，不胡乱生成

### 3.3 检索配置

- **F09** 可调整 Top-K 召回数量（默认 4）
- **F10** 支持相似度阈值过滤，低于阈值的结果不采用

---

## 四、非功能需求

| 指标 | 要求 |
|------|------|
| 响应时间 | 单次问答 ≤ 8 秒 |
| 文档解析 | 100 页 PDF ≤ 30 秒完成向量化 |
| 准确率 | 答案来源可追溯，不凭空捏造 |

---

## 五、技术架构

```
用户界面 (Streamlit / Gradio)
        │
        ▼
   文档处理层
  ┌─────────────────────────────┐
  │  PDF解析 → 文本分块 → Embedding │
  └─────────────────────────────┘
        │
        ▼
   向量数据库 (Chroma)
        │
        ▼
   RAG Chain (LangChain LCEL)
  ┌─────────────────────────────┐
  │  检索器 → Prompt模板 → LLM  │
  └─────────────────────────────┘
        │
        ▼
   对话记忆 (ConversationBufferMemory)
```

### 技术选型

| 层次 | 技术 |
|------|------|
| UI | Streamlit |
| 文档解析 | PyMuPDF (fitz) |
| 文本分块 | LangChain RecursiveCharacterTextSplitter |
| Embedding | text-embedding-3-small (OpenAI) |
| 向量库 | Chroma (本地) |
| LLM | Claude claude-sonnet-4-6 / GPT-4o |
| 框架 | LangChain LCEL |

---

## 六、核心实现要点

### 6.1 文本分块策略

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每块字符数
    chunk_overlap=50,      # 块间重叠，保证上下文连贯
    separators=["\n\n", "\n", "。", "！", "？", " "]
)
```

### 6.2 RAG Chain 构建

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)
```

### 6.3 来源追溯

```python
# 使用 RetrievalQAWithSourcesChain 或手动提取 metadata
docs = retriever.get_relevant_documents(query)
sources = [(doc.metadata["source"], doc.metadata["page"]) for doc in docs]
```

---

## 七、评估标准

问答系统上线前需通过以下测试：

- [ ] 上传 3 份不同类型 PDF，均能正确解析
- [ ] 针对文档内容提问 10 个问题，≥8 个答案来源可追溯
- [ ] 提问超出文档范围时，系统拒绝回答而非编造
- [ ] 多轮对话连续 5 轮，上下文理解正确
- [ ] 删除文档后，相关问题不再从该文档中检索

---

## 八、项目交付物

1. `app.py` — Streamlit 主程序
2. `rag/loader.py` — PDF 解析与分块
3. `rag/vectorstore.py` — 向量库管理
4. `rag/chain.py` — RAG 链构建
5. `README.md` — 部署说明与使用截图
6. `.env.example` — 环境变量模板

---

## 九、扩展方向（完成后可尝试）

- 支持 Word、Excel、网页 URL 等多种数据源
- 引入 HyDE（假设文档嵌入）提升检索质量
- 使用 Reranker 对召回结果二次排序
- 部署到云端，支持多用户隔离的知识库
