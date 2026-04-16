# 01_RAG 智能知识库问答系统 — 完整学习指南

> 适用对象：系统学习 AI Agent 开发的工程师  
> 学习目标：掌握生产级 RAG 系统的设计、实现、测试与调优  
> 预计时长：8-12 小时（含动手实验）

---

## 目录

1. [项目全景与架构速览](#一-项目全景与架构速览)
2. [环境搭建与运行](#二-环境搭建与运行)
3. [核心代码逐模块讲解](#三-核心代码逐模块讲解)
4. [通过测试学习 RAG 流程](#四-通过测试学习-rag-流程)
5. [分块策略调优指南](#五-分块策略调优指南)
6. [检索策略调优指南](#六-检索策略调优指南)
7. [面试高频问题与回答](#七-面试高频问题与回答)
8. [延伸阅读与进阶路径](#八-延伸阅读与进阶路径)

---

## 一、项目全景与架构速览

### 1.1 这个项目解决什么问题

传统搜索引擎（如 Elasticsearch）基于关键词匹配，对语义理解很差：
- 用户问"性能怎么样"，搜不到写着"吞吐量"、"延迟"的段落
- 用户问"上一个问题的技术怎么实现"，无法理解"上一个问题"指什么

RAG（Retrieval-Augmented Generation）的核心思路：
1. **先检索**：从文档库里找到最相关的段落
2. **再生成**：把段落作为上下文喂给大模型，让 LLM 用自然语言回答

本项目是一个**生产级** RAG 系统，包含了玩具 Demo 通常没有的关键特性：
多轮对话感知、混合检索、幂等索引、会话隔离、MCP 集成。

---

### 1.2 完整数据流（v2 必须背下来）

这一版系统已经不是“文档切块后直接建向量索引”的单层 RAG，而是**写入链路**和**查询链路**明确分离的 `parent-child` 架构。

#### 写入链路：文档如何变成索引

```
PDF 文件
  │
  ▼
[1] 文档解析 (loader.py)
    PyMuPDF / pypdf 提取文本
    清洗空行、页码噪声、断句问题
  │
  ▼
[2] 分层分块 (chunker.py)
    ├─ 先按标题、空行、列表边界切 section
    ├─ 再生成 parent chunks（大块，给 LLM 看）
    └─ 再生成 child chunks（小块，给检索用）
  │
  ▼
[3] 稳定元数据生成 (chunker.py)
    doc_version / parent_id / child_id
    解决“重建索引后块 ID 漂移”问题
  │
  ▼
[4] 双存储写入 (vectorstore.py + docstore.py)
    ├─ child chunks → ChromaDB
    └─ parent chunks → SQLite docstore
  │
  ▼
[5] 文档级幂等更新
    同 doc_id 先删旧版本，再写新版本
```

#### 查询链路：问题如何变成答案

```
用户提问
  │
  ▼
[1] 查询改写 (chain.py)
    小模型把“它的性能怎么样”改写成“LangGraph 的性能怎么样”
  │
  ▼
[2] child 级混合检索 (retriever.py)
    ├─ Dense：问题向量 → ChromaDB Top-K
    └─ Sparse：BM25 关键词匹配 Top-K
  │
  ▼
[3] 结果融合与父块回填 (retriever.py)
    按 parent_id 聚合 child hits
    去重后从 docstore 取回 parent 原文
  │
  ▼
[4] Prompt 组装 (chain.py)
    parent 内容 + section_path + page_range + 历史对话 + 用户问题
  │
  ▼
[5] LLM 生成
    严格基于 hydrated parent docs 回答
  │
  ▼
[6] 来源展示 (app.py)
    答案 + 文档名 + 页码范围 + 章节路径
```

**这套设计为什么更接近生产级？**
- child 小块提升召回精度
- parent 大块提升回答完整性
- docstore 让“召回”和“给模型看的上下文”解耦
- 稳定 ID 让重建索引、调试和灰度迁移更可靠

---

### 1.3 技术栈总览

| 层次 | 组件 | 技术选型 | 说明 |
|------|------|---------|------|
| **文档解析** | `loader.py` | PyMuPDF / pypdf | PyMuPDF 更准确，pypdf 纯 Python 兜底 |
| **层级分块** | `chunker.py` | `RecursiveCharacterTextSplitter.from_tiktoken_encoder` | 结构优先 + token 约束 + parent-child 双层切分 |
| **向量化** | `embedder.py` | DashScope / OpenAI / HuggingFace | 工厂模式，按 API Key 自动选择 |
| **child 向量存储** | `vectorstore.py` | ChromaDB | 仅保存 child chunks，负责 Dense 检索 |
| **parent 文档存储** | `docstore.py` | SQLite | 保存 parent 原文，供检索后回填 |
| **关键词检索** | `retriever.py` | rank-bm25 | 在 child chunks 上做 Sparse 召回 |
| **混合聚合** | `retriever.py` | Ensemble + parent hydration | 两路召回后按 parent_id 聚合 |
| **LLM 链** | `chain.py` | LangChain LCEL | 查询改写 → 检索 → 上下文拼装 → 生成 |
| **多轮记忆** | `session.py` | 自研 SessionManager | FIFO 自动截断，会话隔离 |
| **UI** | `app.py` | Streamlit | 上传文档、索引、问答、调试展示 |
| **文件管理** | `filesystem_client.py` | MCP + 直接 IO | Model Context Protocol 集成 |

---

## 二、环境搭建与运行

### 2.1 安装依赖

```bash
cd 01_RAG
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2.2 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key（至少一个）
```

最简配置（二选一）：

```bash
# 选项 A：使用 Anthropic Claude（推荐，效果最好）
ANTHROPIC_API_KEY=sk-ant-xxxx

# 选项 B：使用 DashScope 通义千问（国内访问快）
DASHSCOPE_API_KEY=sk-xxxx
```

### 2.3 生成测试文档

```bash
python generate_sample_pdfs.py
# 会在 data/documents/ 生成两个演示 PDF：
# - AI_Agent_技术白皮书.pdf（10页）
# - LangGraph_开发手册.pdf（8页）
```

### 2.4 启动应用

```bash
streamlit run app.py
# 浏览器自动打开 http://localhost:8501
```

---

## 三、核心代码逐模块讲解

### 3.1 配置中心 `config.py`

**知识点：单一职责 + 配置集中管理**

```python
# config.py:30-45 — LLM 提供商优先级
class LLMConfig:
    @property
    def provider(self) -> str:
        if self.dashscope_api_key:
            return "dashscope"      # 优先级1：通义千问
        elif self.anthropic_api_key:
            return "anthropic"      # 优先级2：Claude
        elif self.openai_api_key:
            return "openai"         # 优先级3：GPT
        raise ValueError("未配置任何 LLM API Key")
```

**为什么这样设计？**

生产系统需要在多个 LLM 提供商之间灵活切换（成本、可用性、合规要求）。
把选择逻辑集中在一处，业务代码不感知具体提供商。

**核心 RAG 参数（`config.py:70-85`）：**

```python
TOKENIZER_NAME = "cl100k_base"
PARENT_TARGET_TOKENS = 900
PARENT_MAX_TOKENS = 1200
PARENT_OVERLAP_TOKENS = 100
CHILD_TARGET_TOKENS = 280
CHILD_MAX_TOKENS = 360
CHILD_OVERLAP_TOKENS = 60
SEMANTIC_TOP_K = 6        # 向量检索取前6个
BM25_TOP_K = 6            # BM25检索取前6个
FINAL_TOP_K = 4           # 最终返回的 parent 数量
SEMANTIC_WEIGHT = 0.6     # 语义检索权重
SIMILARITY_THRESHOLD = 0.3
MAX_HYDRATED_PARENTS = 6
```

---

### 3.2 PDF 解析 `rag/loader.py`

**知识点：文档预处理是 RAG 精度的基础**

```python
# loader.py — 双引擎解析策略
def load_pdf(file_path: str) -> list[Document]:
    try:
        return load_pdf_pymupdf(file_path)   # 主引擎：准确率高
    except Exception:
        return load_pdf_pypdf(file_path)      # 备用：纯Python，无系统依赖
```

**文本清洗逻辑（loader.py:60-85）：**

```python
def _clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)   # 连续空行 → 双换行
    text = re.sub(r' {2,}', ' ', text)        # 多空格 → 单空格
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)  # 删除纯数字行（页码）
    return text.strip()
```

**为什么清洗很重要？**
未清洗的 PDF 常见问题：页眉页脚（"第3页 共25页"）会污染检索结果，
纯页码数字块会浪费向量索引空间，连续空行导致分块不均匀。

**跨页合并（loader.py:90-110）：**
PDF 按页切割后，一个句子可能被截断在页面边界。
代码检测句子末尾（中英文标点），对未结束的句子与下一页首句合并。

**元数据结构（每个 Document 必带）：**
```python
metadata = {
    "source": "AI_Agent_技术白皮书.pdf",  # 文件名（显示给用户）
    "file_path": "/abs/path/to/file.pdf",
    "page": 3,                              # 页码（用于来源引用）
    "total_pages": 10,
    "doc_id": "a1b2c3d4e5f6"              # MD5 哈希前12位（用于幂等删除）
}
```

---

### 3.3 文本分块 `rag/chunker.py`

**知识点：v2 的核心不是“语义分块四个字”，而是“结构优先 + token 约束 + 双层 chunk”。**

当前实现不再把 `chunk_size=500` 当作系统中心，而是把分块拆成 4 个工程问题：

1. **怎么识别 section 边界**
2. **怎么控制 parent 大小**
3. **怎么控制 child 大小**
4. **怎么给每一层 chunk 一个稳定身份**

#### 3.3.1 为什么不用“纯语义分块”直接替换

很多教程会说：“把固定大小分块换成 SemanticChunker 就升级了。”

这在 Demo 里成立，在生产里通常不够：

- 纯 embedding 语义切点成本高
- PDF/OCR 噪声会让语义断点不稳定
- chunk 大小不稳定，容易打爆上下文
- 索引重建后块边界漂移，运维和回归测试困难

本项目选择的折中方案是：

```text
先按结构切 section
  → 再按 token 预算切 parent
    → 再切 child
      → child 负责召回，parent 负责给 LLM 上下文
```

#### 3.3.2 `ChunkingResult`：为什么返回对象而不是列表

```python
@dataclass
class ChunkingResult(Sequence[Document]):
    parents: List[Document]
    children: List[Document]
    stats: dict
```

设计原因：

- **索引层**需要 child chunks
- **docstore**需要 parent chunks
- **UI/调试**需要统计信息
- **旧测试/旧调用**还希望它像 `List[Document]` 一样可迭代

这是一个典型的兼容性设计：
对外升级为 richer result，对内又保留 sequence 行为，避免所有旧调用立刻失效。

#### 3.3.3 分块的真实流程

```python
def chunk_documents(documents, strategy="hierarchical_v2") -> ChunkingResult:
    sections = _build_sections(doc_group)
    parent_splitter = _build_splitter(..., chunk_size=parent_max)
    child_splitter = _build_splitter(..., chunk_size=child_max)

    for section in sections:
        parent_texts = parent_splitter.split_text(section.text)
        for parent_text in parent_texts:
            parent_doc = Document(...)
            child_texts = child_splitter.split_text(parent_text)
            for child_text in child_texts:
                child_doc = Document(...)
```

#### 3.3.4 section 切分原理

section 不是靠 embedding 算出来的，而是靠**结构线索**：

- 标题行，如 `第一章`、`1.`、`一、`
- 空行块
- 列表起始符
- 文本长度和换行形式

核心判断是 `_is_heading()`：

```python
heading_patterns = [
    r"^第[一二三四五六七八九十百千\\d]+[章节篇部分]",
    r"^[0-9]+[.)、．]",
    r"^[一二三四五六七八九十]+、",
]
```

这比“按页切”或“按固定字符切”更接近人类阅读的逻辑边界。

#### 3.3.5 token 驱动而不是字符驱动

旧版指南里的 `chunk_size=500` 是字符语义；v2 真正使用的是 token 预算：

```python
TOKENIZER_NAME = "cl100k_base"
PARENT_TARGET_TOKENS = 900
PARENT_MAX_TOKENS = 1200
CHILD_TARGET_TOKENS = 280
CHILD_MAX_TOKENS = 360
```

为什么一定要改成 token？

- embedding 计费按 token
- LLM 上下文限制按 token
- 中英混排时字符数完全不可靠
- “500 个汉字”和“500 个英文字符”对模型负担完全不同

#### 3.3.6 为什么要 parent / child 两层

| 层级 | 大小 | 主要用途 | 典型问题 |
|------|------|---------|---------|
| **child** | 小 | 召回 | 太小会丢上下文 |
| **parent** | 大 | 给 LLM 上下文 | 太大会增加噪声 |

生产上的关键认知是：
**召回最优块大小** 和 **回答最优上下文大小** 往往不是同一个值。

所以：
- child 尽量短，利于相似度匹配
- parent 保留完整段落语义，利于回答

#### 3.3.7 稳定 ID 与 `doc_version`

这是这轮升级里最容易被忽视、但最“工程化”的部分。

```python
doc_version = sha256(all_text)[:12]
parent_id = f"{doc_id}:p:{sha256(parent_text)[:12]}"
child_id = f"{doc_id}:c:{sha256(child_text)[:12]}"
```

为什么不能只靠 `chunk_index`？

- 一旦文档前面插入一段内容，后面所有块序号都变
- 重建索引时无法判断“内容没变，只是位置变了”
- 调试检索结果时很难追踪同一个语义块

`chunk_index` 仍保留用于兼容和调试，但身份语义已经让位给内容哈希。

---

### 3.4 向量化 `rag/embedder.py`

**知识点：Embedding 是语义检索的核心**

Embedding 模型把文本转换为高维向量（如 1536 维），语义相似的文本在向量空间中距离近。

```python
# embedder.py — 工厂模式
def get_embeddings() -> Embeddings:
    provider = llm_config.provider()
    if provider == "dashscope":
        return DashScopeEmbeddings(model=llm_config.EMBEDDING_MODEL)
    elif provider == "openai":
        return OpenAIEmbeddings(model=llm_config.EMBEDDING_MODEL)
    else:
        return HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
```

**批处理 + 重试（embedder.py:60-90）：**
```python
# 每批25个文档，避免 API 限流
# 指数退避重试（1s → 2s → 4s），最多3次
# 批间延迟 0.3s，保护 API 配额
```

**v2 的关键变化：只对 child chunks 做 Embedding。**

这是非常重要的工程决策：

- parent 主要给 LLM 看，不参与 Dense 检索
- 如果 parent 和 child 都做 embedding，会付出双倍索引成本
- child 更短，向量表达更聚焦，检索精度通常更高

因此，`embedder.py` 本身变化不大，但它在系统中的职责变了：
**它服务于 child retrieval，不再服务于“所有可展示文本”。**

---

### 3.5 Parent DocStore `rag/docstore.py`

**知识点：不是所有检索相关文本都应该进向量库。**

```python
class ParentDocStore:
    def upsert_parents(self, parents): ...
    def get_parents(self, parent_ids): ...
    def delete_document(self, doc_id): ...
    def list_documents(self): ...
```

为什么要单独做一个 SQLite docstore？

- parent 不需要向量检索
- parent 需要完整原文和 metadata
- parent 量通常比 child 少，适合本地 KV/SQLite 存储
- 检索命中 child 后，需要快速按 `parent_id` 回填原文

这是一个很典型的生产化拆分：

```text
vector DB 负责“找”
docstore 负责“取回完整内容”
```

很多线上 RAG 系统都会有类似设计，只是底层可能换成 Redis / Postgres / S3 + KV 索引。

---

### 3.6 Child 向量存储 `rag/vectorstore.py`

**知识点：幂等性 + 双存储一致性，是这层的核心。**

```python
def add_documents(chunks: ChunkingResult, doc_id: str, ...):
    delete_document(doc_id, vs, docstore)
    docstore.upsert_parents(chunks.parents)
    vs.add_documents(documents=chunks.children, ids=child_ids)
```

#### 3.6.1 为什么这是“文档级原子性”的近似实现

如果只写一半：

- child 写成功，parent 写失败 → 检索能命中，但无法回填原文
- parent 写成功，child 写失败 → 文档存在，但永远检索不到

所以当前实现采用非常朴素但有效的策略：

1. 先删旧版本
2. 写新 parent
3. 写新 child
4. 如果任何步骤失败，执行 `delete_document(doc_id)` 回滚

这不是数据库教科书里的严格事务，但对本地 SQLite + Chroma 组合已经足够实用。

#### 3.6.2 collection 版本化

```python
COLLECTION_NAME = f"rag_knowledge_base_{ACTIVE_INDEX_VERSION}_children"
DB_PATH = data/docstore/parents_v2.sqlite
```

这是为迁移准备的：

- v1 collection 可以保留
- v2 collection 独立重建
- 回滚和对比更简单

生产里这类做法通常叫 **index versioning**。

#### 3.6.3 `list_documents()` 为什么要同时看两边

现在文档统计不仅要看总块数，还要看：

- `child_count`
- `parent_count`
- `doc_version`

因为“这个文档被索引了”不再等于“向量库里有一些 chunk”。
它必须满足：

```text
child collection 有数据
且
parent docstore 也有对应记录
```

**相似度过滤依然保留，但只发生在 child 层。**

---

### 3.7 混合检索 + 父块回填 `rag/retriever.py`

**知识点：这是本轮升级技术含量最高的模块。**

#### 3.7.1 为什么要混合检索

| 检索方式 | 优点 | 缺点 | 适用场景 |
|---------|------|------|---------|
| 纯语义（向量） | 理解语义，同义词/近义词 | 对专有名词、代码效果差 | 概念性问题 |
| 纯关键词（BM25） | 精确匹配，对术语敏感 | 不理解语义，无法处理同义词 | 精确查找 |
| **混合** | 兼顾两者 | 实现复杂 | **生产场景默认选择** |

举例：用户问"模型推理延迟"
- 语义检索能找到"响应时间"、"吞吐量"相关的段落（语义相近）
- BM25 能精确找到含有"延迟"字样的段落
- 混合：两者都找到，最终排名更准

#### 3.7.2 v2 新增：为什么检索结果不能直接返回 child

如果直接把 child 送给 LLM，会遇到两个典型问题：

1. 命中了关键句，但上下文不完整
2. 多个 child 命中同一段语义，LLM 收到重复碎片

所以 `ParentChildHybridRetriever` 的工作流是：

```python
child_hits = ensemble_retriever.invoke(query)
hydrated_parents = hydrate_parent_results(child_hits, parent_docstore)
```

`hydrate_parent_results()` 做了 4 件事：

1. 按 `parent_id` 聚合 child hits
2. 收集 `matched_child_ids`
3. 计算 parent 排名
4. 从 docstore 取回 parent 原文

这是“召回”和“阅读上下文”解耦的关键。

#### 3.7.3 BM25 原理

```
BM25(q, d) = Σ IDF(t) × [tf(t,d) × (k1+1)] / [tf(t,d) + k1 × (1-b+b×|d|/avgdl)]

其中：
- IDF(t)：词 t 的逆文档频率（稀有词权重高）
- tf(t,d)：词 t 在文档 d 中的频率
- k1=1.5, b=0.75：调节参数（可调优）
- |d|/avgdl：文档长度归一化
```

BM25 这里跑在 **child 集合** 上，而不是 parent 集合上。因为 child 更短，关键词命中更集中。

#### 3.7.4 当前融合实现的工程现实

```python
ensemble = EnsembleRetriever(
    retrievers=[semantic_retriever, bm25_retriever],
    weights=[semantic_weight, 1 - semantic_weight],
)
```

当前代码使用 LangChain 的 `EnsembleRetriever` 来做排序融合，而不是手写 `_rrf_fusion`。
学习上要理解两层概念：

- **算法概念**：混合检索本质上仍然是在做多路召回融合
- **工程实现**：是否手写 RRF 不是核心，核心是结果融合后还能稳定回填 parent

#### 3.7.5 面试时该怎么讲这个模块

推荐回答顺序：

1. 为什么要 Dense + Sparse
2. 为什么 child 负责召回
3. 为什么 parent 负责最终上下文
4. 为什么要 `parent_id` 聚合去重
5. 为什么 parent 要单独存 docstore

---

### 3.8 RAG 链 `rag/chain.py`

**知识点：LCEL（LangChain Expression Language）的声明式设计**

```python
# chain.py — 完整 Pipeline（伪代码结构）
chain = (
    {"question": RunnablePassthrough()}
    | RunnableLambda(rewrite_query)        # Step1: 查询改写
    | RunnableLambda(hybrid_retrieve)      # Step2: 混合检索
    | RunnableLambda(format_context)       # Step3: 组装上下文
    | chat_prompt                          # Step4: 填充 Prompt 模板
    | llm                                  # Step5: LLM 生成
    | StrOutputParser()                    # Step6: 提取文本
)

# 包装历史记忆
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_or_create_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)
```

v2 和旧版最大的区别，是 `hybrid_retrieve` 这一步返回的已经不是零碎 child，而是带有：

- `page_range`
- `section_path`
- `matched_child_ids`

的 hydrated parent documents。

**查询改写（chain.py:40-70）：**

```python
# 用配置里的轻量 rewrite 模型处理代词消解
rewrite_prompt = """
基于对话历史，将用户最新问题改写为独立完整的问题。
如果问题已经完整，直接返回原问题，不要解释。

历史：{chat_history}
问题：{question}
改写后：
"""
# 为什么重要？
# 用户说"它的实现原理是什么"→ LLM 不知道"它"指什么
# 改写后："LangGraph 的条件路由实现原理是什么"
```

**System Prompt（chain.py:90-110）：**

```python
SYSTEM_PROMPT = """你是一个严谨的知识库助手。请严格基于以下检索到的文档内容回答用户问题。

规则：
1. 只能基于提供的上下文回答，不得凭借自身训练知识推测
2. 如果上下文中没有相关信息，明确告知"文档中未找到相关内容"
3. 引用信息时注明来源文档和页码
4. 不得虚构或推断文档中未明确说明的内容

上下文：
{context}
"""
```

**上下文格式化（chain.py 里的 `format_docs_for_context`）：**

现在会展示：

- 文档名
- 页码范围 `page_range`
- 章节路径 `section_path`
- parent 原文

这意味着最终喂给模型的上下文已经具备“可解释引用”的生产基础。

---

### 3.9 会话记忆 `memory/session.py`

**知识点：多用户隔离 + 防内存泄漏**

```python
# session.py — SessionManager 核心逻辑
class SessionManager:
    def __init__(self, max_history: int = 10):
        self._sessions: dict[str, ChatMessageHistory] = {}
        self.max_history = max_history  # 最多保留10轮对话
    
    def add_exchange(self, session_id, user_msg, ai_msg):
        history = self.get_or_create(session_id)
        history.add_user_message(user_msg)
        history.add_ai_message(ai_msg)
        # FIFO 截断：超过 max_history × 2 条消息时，删除最早的
        self._trim_if_needed(session_id)
    
    def _trim_if_needed(self, session_id):
        messages = self._sessions[session_id].messages
        limit = self.max_history * 2  # 用户+AI各算一条
        if len(messages) > limit:
            # 保留最新的 limit 条
            self._sessions[session_id].messages = messages[-limit:]
```

**为什么要自动截断？**
不截断的后果：
1. 每次请求携带的 token 数量线性增长，成本爆炸
2. 超过 LLM 上下文窗口（如 200k tokens），请求直接报错
3. 内存不断增长，长时间运行后 OOM

---

### 3.10 UI 主程序 `app.py`

**知识点：Streamlit 的响应式状态管理**

```python
# app.py:116-145 — 会话状态初始化（Streamlit 每次交互会重新运行整个脚本）
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # 每个浏览器Tab独立会话

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None  # 懒加载，首次查询时初始化

if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []  # 已索引文档列表
```

**Streamlit 的执行模型（面试常考）：**
每次用户操作（点击、输入），Streamlit 从头到尾重新执行整个 Python 脚本。
`st.session_state` 是唯一在执行间持久保存数据的机制。
这与 Flask/Django 的请求-响应模型完全不同。

---

## 四、通过测试学习 RAG 流程

### 4.1 先跑完整非慢测试

```bash
cd 01_RAG
python -m pytest tests/ -q -m 'not slow'
```

当前基线应看到：

```text
30 passed, 1 deselected
```

这套测试分成两层：

- `tests/test_rag_pipeline.py`
  - 老的兼容性与基础模块测试
  - 重点保证 loader、session、filesystem、基础 chunk 行为不回归
- `tests/test_chunking_v2.py`
  - 这次升级新增的 parent-child 架构测试
  - 重点验证 `ChunkingResult`、双存储聚合、parent hydration、上下文格式化

### 4.2 你应该重点读哪些测试

#### 4.2.1 `test_chunk_documents_returns_hierarchical_result`

这个测试验证的不是“能不能切块”，而是 v2 最核心的契约：

- `chunk_documents()` 返回 `ChunkingResult`
- 同时生成 `parents` 和 `children`
- parent 含 `doc_version`
- child 含 `parent_id`

如果这条测试挂了，说明系统已经从“双层索引”退化回“单层切块”。

#### 4.2.2 `test_list_documents_includes_parent_and_child_counts`

这个测试教你一个很重要的工程思维：
**索引状态不应该只由向量库单边决定。**

它验证 `list_documents()` 会同时参考：

- child vectorstore
- parent docstore

从而输出：

- `child_count`
- `parent_count`
- `doc_version`

这就是生产里常说的“读模型聚合”。

#### 4.2.3 `test_parent_child_hybrid_retriever_hydrates_and_deduplicates`

这是理解 retrieval v2 的最佳入口。

它验证了 3 个关键行为：

1. 多个 child 命中同一个 parent 时，最终只返回一个 parent
2. `matched_child_ids` 会被保留下来，便于调试和解释性分析
3. 检索返回的是 hydrated parent，而不是原始 child

这条测试本质上在守护“召回层”和“阅读层”解耦。

#### 4.2.4 `test_format_docs_for_context_uses_page_range_for_parent_docs`

这个测试看起来很小，但它保证了 LLM 实际接收到的是 v2 语义：

- `page_range`
- `section_path`
- parent 原文

如果上下文格式还停留在 “第X页单块内容”，说明升级只做了底层，没有贯通到回答链路。

### 4.3 动手实验：按 v2 路径理解系统

#### 实验1：观察 loader 输出

```python
import sys
sys.path.insert(0, ".")

from rag.loader import load_pdf

docs = load_pdf("data/documents/AI_Agent_技术白皮书.pdf")
print("页数：", len(docs))
print("第一页 metadata：", docs[0].metadata)
print("第一页前 200 字：", docs[0].page_content[:200])
```

观察点：

- `doc_id` 是否全页一致
- `page`、`total_pages` 是否完整
- 清洗后是否还有明显页码噪声和空行噪声

#### 实验2：观察 `ChunkingResult`

```python
from rag.chunker import chunk_documents

result = chunk_documents(docs)
print("parent 数：", len(result.parents))
print("child 数：", len(result.children))
print("统计：", result.stats)

print("\n第一个 parent metadata：")
print(result.parents[0].metadata)

print("\n第一个 child metadata：")
print(result.children[0].metadata)
```

你应该重点看这些字段：

- parent: `parent_id`, `doc_version`, `section_path`, `page_range`
- child: `child_id`, `parent_id`, `token_count`

#### 实验3：理解 stable ID

```python
result1 = chunk_documents(docs)
result2 = chunk_documents(docs)

print(result1.parents[0].metadata["parent_id"])
print(result2.parents[0].metadata["parent_id"])
```

如果输入文档没变，这两个 `parent_id` 应该一致。

这就是内容哈希带来的稳定性。

#### 实验4：理解双存储

```python
from rag.vectorstore import get_vectorstore, add_documents, list_documents
from rag.docstore import get_parent_docstore

vs = get_vectorstore()
docstore = get_parent_docstore()

doc_id = docs[0].metadata["doc_id"]
added = add_documents(result, doc_id, vs, docstore)
print("写入 child 数：", added)
print("文档列表：", list_documents(vs, docstore))
```

观察点：

- `child_count` 是否等于 `len(result.children)`
- `parent_count` 是否等于 `len(result.parents)`
- `doc_version` 是否出现在聚合结果中

#### 实验5：理解 parent-child 检索

```python
from rag.retriever import get_hybrid_retriever

retriever = get_hybrid_retriever()
results = retriever.invoke("保修期是多久")

for doc in results:
    print("\n来源：", doc.metadata.get("source"))
    print("章节：", doc.metadata.get("section_path"))
    print("页码范围：", doc.metadata.get("page_range"))
    print("命中的 child：", doc.metadata.get("matched_child_ids"))
    print("内容：", doc.page_content[:200])
```

如果你看到的是完整 parent 段落，而不是零碎 child 句子，说明 hydration 生效了。

### 4.4 生产级测试方法

真正的生产测试不应只停留在“函数返回值正确”，还要覆盖以下 5 类风险。

#### 4.4.1 契约测试

验证输入输出结构稳定：

- `ChunkingResult` 结构
- parent / child metadata 字段完整性
- `list_documents()` 返回格式
- hydrated parent 的上下文字段

#### 4.4.2 稳定性测试

验证同一文档重复处理时：

- `doc_version` 不变
- `parent_id` 不变
- `child_id` 不变

这类测试决定你能不能做可重复索引、缓存和结果回放。

#### 4.4.3 一致性测试

验证 child vectorstore 和 parent docstore 不会分叉：

- 新增后两边都有数据
- 删除后两边都清空
- 半失败后不会遗留脏数据

#### 4.4.4 检索行为测试

重点不是“有没有结果”，而是：

- 是否正确按 `parent_id` 去重
- 是否真的返回 parent 而不是 child
- 是否保留 `matched_child_ids`
- page_range / section_path 是否可追踪

#### 4.4.5 端到端测试

最终还是要验证完整问答：

- 查询改写
- 检索命中
- 上下文组装
- LLM 回答
- 来源展示

这类测试通常单独标记为 slow，因为依赖真实模型 API。

### 4.5 建议自己补的测试

```python
def test_same_input_produces_stable_ids():
    result1 = chunk_documents(docs)
    result2 = chunk_documents(docs)
    assert result1.parents[0].metadata["parent_id"] == result2.parents[0].metadata["parent_id"]

def test_delete_document_removes_child_and_parent():
    add_documents(result, doc_id, vs, docstore)
    delete_document(doc_id, vs, docstore)
    assert not list_documents(vs, docstore)

def test_parent_hydration_deduplicates_children_from_same_parent():
    hydrated = hydrate_parent_results(child_hits, docstore)
    assert len(hydrated) < len(child_hits)
```

这些测试最能体现你对 v2 设计的理解。

---

## 五、分块策略调优指南

### 5.1 v2 应该调哪些参数

旧版的核心参数只有：

- `chunk_size`
- `chunk_overlap`

v2 之后，真正重要的是 6 组参数：

- `PARENT_TARGET_TOKENS`
- `PARENT_MAX_TOKENS`
- `PARENT_OVERLAP_TOKENS`
- `CHILD_TARGET_TOKENS`
- `CHILD_MAX_TOKENS`
- `CHILD_OVERLAP_TOKENS`

理解方法：

```text
parent 决定“回答完整性”
child 决定“召回精度”
```

### 5.2 哪些因素比分块参数更重要

按生产经验排序，通常是：

1. 文本清洗质量
2. section 边界质量
3. child 大小
4. parent 大小
5. overlap

也就是说，**先别急着调 token 数，先看文档有没有被正确切成 section。**

### 5.3 生产上如何调 parent / child

| 场景 | parent | child | 建议 |
|------|--------|-------|------|
| 技术文档 | 800-1200 tokens | 220-320 tokens | 当前默认较合适 |
| 法律/制度文档 | 1000-1500 tokens | 280-420 tokens | parent 适当更大 |
| FAQ / 短问答 | 300-600 tokens | 100-180 tokens | 往往不需要太大 parent |
| 含大量代码/API 文档 | 600-1000 tokens | 180-260 tokens | 要优先保护代码块完整性 |

#### 一个重要经验：

- **child 太小**：召回准，但问题一复杂就缺上下文
- **child 太大**：Dense 检索噪声变多
- **parent 太小**：LLM 看到的是碎片
- **parent 太大**：一个 parent 内混入太多无关语义

### 5.4 overlap 怎么看

```
overlap: 小 ←————————→ 大
        边界断裂风险高     重复内容和成本上升
```

建议：

- parent overlap 主要防止长段语义跨边界
- child overlap 主要防止关键句被截断
- 如果 section 切分已经足够好，overlap 不需要无限增大

### 5.5 结构切分比“纯语义分块”更重要

这一版最值得吸收的工程结论是：

> 生产里的“语义分块”通常不是先上 SemanticChunker，  
> 而是先把结构切分、token 预算、双层索引做好。

什么时候再考虑 embedding 驱动切点？

- 医学、法律、研究报告等高价值长文档
- 用户问题对语义边界极敏感
- 已经有离线评测体系，能证明收益覆盖成本

### 5.6 正确的分块评估方式

不要只看“切了多少块”，要看：

- `Recall@K`
- `MRR`
- `平均 child token`
- `平均 parent token`
- `命中后回答完整率`
- `索引成本 / 速度`

推荐最小评估集格式：

```python
eval_dataset = [
    {
        "question": "保修期多久",
        "expected_source": "test_document.pdf",
        "expected_section": "第一章 产品概述",
        "expected_keywords": ["保修期", "12 个月"],
    }
]
```

你要验证的不只是“有没有命中”，还要验证：
**命中的 parent 是否足够完整，能让 LLM 一次回答清楚。**

---

## 六、检索策略调优指南

### 6.1 现在调优检索，不能只看 Dense/BM25 权重

v1 的检索调优重点是：

- `SEMANTIC_WEIGHT`
- `FINAL_TOP_K`
- `SIMILARITY_THRESHOLD`

v2 之后，还要多看两件事：

- child 命中后能否正确聚合到 parent
- hydrated parent 是否过大或过多

### 6.2 Dense / Sparse 权重怎么调

当前默认思路仍然合理：

- 概念性、描述性问题：Dense 权重大一些
- 术语、数字、代码、版本号问题：BM25 权重大一些

一个实战策略是把评测问题分成两组：

```python
conceptual_questions = [...]
keyword_questions = [...]
```

然后分别观察 Recall，而不是把所有问题混在一起。

### 6.3 `MAX_HYDRATED_PARENTS` 比旧的 `FINAL_TOP_K` 更关键

旧系统只要决定“返回几个 chunk”。

新系统要决定：

- 召回多少 child
- 最终送多少 parent 给 LLM

如果 hydrated parents 太多：

- 上下文成本上升
- 章节之间互相干扰
- LLM 更容易答散

如果太少：

- 对比题、多条件题会丢信息

一个常见经验值是 `4~6` 个 parent。

### 6.4 相似度阈值现在只作用于 child 层

这点很重要。

阈值过滤掉的是“候选 child”，不是最终 parent。
所以阈值过高时，问题不是“少了一些块”，而是：
**整个 parent 可能根本进不了最终候选集合。**

调这个参数时建议看两类错误：

- 明显无关结果进入最终答案
- 明显相关段落完全消失

### 6.5 什么时候该加 Reranking

判断标准：

| 症状 | 优先动作 |
|------|---------|
| 根本召回不到相关内容 | 先看 chunking、query rewrite、BM25 |
| 能召回到，但最终前几条排序不对 | 再考虑 Reranking |
| 返回的 parent 太长、太杂 | 先调 parent size，不要先上 Reranking |

如果以后接入 Reranking，最佳位置通常是：

```text
Dense/BM25 child 召回
  → parent 聚合
    → 对 parent rerank
      → 送入 LLM
```

### 6.6 检索评估应该怎么做

建议至少维护 3 组样本：

1. 概念解释题
2. 精确术语题
3. 跨段落/跨页问题

指标分两层看：

- **Retrieval**
  - Recall@K
  - MRR
  - parent hit rate
- **Generation**
  - 答案完整性
  - 来源正确性
  - 幻觉率

---

## 七、面试高频问题与回答

### Q1：什么是 RAG？为什么用 RAG 而不是直接用 LLM？

**回答思路：**
RAG（检索增强生成）= 检索系统 + 大语言模型的结合。

核心问题是 LLM 有两个天然缺陷：
1. **知识截止**：训练数据有时间限制，不知道最新信息
2. **幻觉问题**：对不确定的内容会编造答案

RAG 的解决方案：不让 LLM "凭感觉"回答，而是先从知识库里检索相关内容，再让 LLM 基于这些内容生成答案。
类比：让医生看病，不是让他凭记忆说，而是先查病历再诊断。

---

### Q2：你们项目用的是什么检索策略？为什么选这个？

**回答思路：**
我们用的是**child 级混合检索 + parent 回填**。

完整说法最好分 3 层：

1. **Dense 检索**
   把 query 和 child chunks 编码成向量，在 Chroma 里做相似度搜索。
2. **Sparse 检索**
   用 BM25 在 child chunks 上做关键词匹配，补强术语、数字、版本号、代码片段。
3. **Parent Hydration**
   多个 child 命中后按 `parent_id` 聚合，再从 docstore 取回 parent 原文给 LLM。

这样做的原因是：

- child 更适合召回，精度高
- parent 更适合给 LLM，看起来完整
- Dense 和 Sparse 互补，能兼顾语义和精确术语

面试里如果只答“我们用混合检索”，还不够。更好的答法是：
**我们把召回层和阅读层拆开了，召回用 child，回答用 parent。**

---

### Q3：分块策略怎么选？影响是什么？

**回答思路：**
分块是 RAG 里最容易被低估、但影响最大的模块之一。

我会先讲清楚两个不同目标：

- **检索最优块大小**
- **回答最优上下文大小**

它们通常不是一个值，所以我们用了 `parent-child` 双层分块：

- child 小块负责召回
- parent 大块负责给 LLM 完整上下文

再往上一级，我们不是纯靠字符数切，而是：

1. 先按标题和结构切 section
2. 再按 token 预算切 parent
3. 再切 child
4. 用内容哈希生成稳定 ID

如果你这样答，面试官会觉得你理解的是“工程分块策略”，不是只会背 `chunk_size=500`。

---

### Q4：如何处理多轮对话中的代词问题？

**回答思路：**
这是 RAG 系统中容易被忽略的问题。用户说"它的原理是什么"，"它"指什么？

我们的解决方案是**查询改写（Query Rewriting）**：
在检索之前，用一个小模型（如 Claude Haiku）结合对话历史，把含代词的问题改写为完整独立的问题。

"LangGraph 是什么？" → "它的核心概念有哪些？" → 改写为 → "LangGraph 的核心概念有哪些？"

关键设计：用轻量模型做改写，主模型只用于最终回答生成，这样既解决了问题又控制了成本。

---

### Q5：为什么还要单独做 parent docstore？直接全放向量库不行吗？

**回答思路：**

可以全放，但不划算，也不稳定。

如果 parent 和 child 都进向量库，会有几个问题：

1. 索引成本翻倍
2. Dense 检索噪声上升
3. parent 本来就不是最适合召回的粒度

所以更合理的做法是：

- child 进向量库，负责“找”
- parent 进 docstore，负责“回填完整内容”

这其实是很多生产系统的常见分层：
**vector index 负责候选召回，docstore 负责源文档内容。**

---

### Q6：为什么你们要做 `doc_version` 和稳定 chunk ID？

**回答思路：**

因为生产里索引不是“建一次就结束”，而是会反复重建、增量更新和排查问题。

如果只靠 `chunk_index`：

- 文档前面插入一点内容，后面所有块序号都会变
- 很难判断“内容没变，只是位置变了”
- 灰度迁移和回归测试都很痛苦

所以我们用内容哈希生成：

- `doc_version`
- `parent_id`
- `child_id`

这样能保证：

- 同内容重建时 ID 稳定
- 内容变化时相关块 ID 才变化
- 调试时能跨版本追踪同一语义块

---

### Q7：向量数据库怎么选型？

**回答思路（按场景）：**

| 场景 | 选型 | 理由 |
|------|------|------|
| 开发/原型 | ChromaDB | 本地，无需服务，pip 即装 |
| 小规模生产（<100万块） | Qdrant | 性能好，支持过滤，有云版本 |
| 大规模生产（>1000万块） | Milvus / Weaviate | 分布式，水平扩展 |
| 已有 Postgres | pgvector | 减少组件，SQL 查询方便 |

我们用 ChromaDB 的原因：这是教学项目，优先降低学习门槛。
生产化时我会首选 Qdrant（性能好、有 payload 过滤、支持 Hybrid Search）。

---

### Q8：如何评估 RAG 系统的效果？

**回答思路：**

RAG 评估分两层：

**检索层（Retrieval）：**
- `Recall@K`：K个结果里有几个是相关的
- `MRR（Mean Reciprocal Rank）`：相关结果排在第几位
- `NDCG`：考虑排名顺序的精度指标

**生成层（Generation）：**
- `Faithfulness`：答案是否基于检索内容（防幻觉）
- `Answer Relevance`：答案是否回答了问题
- `Context Precision`：检索到的内容有多少是有用的

如果是 parent-child 架构，我还会额外看一个指标：

- `Parent Hit Rate`：最终送入 LLM 的 parent 是否真的包含目标答案

因为现在“检索命中一个 child”不等于“最终上下文已经足够完整”。

工具上可以用：

- 自建离线评估集
- `RAGAs`
- 人工 spot check

三者结合。

---

### Q9：如何防止 LLM 幻觉？

**回答思路：**

我们从三个层次防止幻觉：

1. **Prompt 层面**：System Prompt 明确规定"只能基于提供的上下文回答，如果找不到相关信息，明确说找不到，不得推测"
2. **检索层面**：相似度阈值过滤（score < 0.3 的结果不送入 LLM），宁可说"找不到"也不给低质量上下文
3. **验证层面**（进阶）：让另一个 LLM 检查答案中的每个事实是否能在检索结果中找到支撑

---

### Q10：如果面试官追问“为什么不直接用纯语义分块”？

**推荐答法：**

纯语义分块不是不能用，而是要看收益是否覆盖成本。

我会这样回答：

1. 纯 semantic chunking 在专业长文档里可能更准
2. 但它通常更慢、更贵，而且对脏文本/OCR 很敏感
3. 工程上我更倾向先做好结构切分、token 预算、parent-child 检索
4. 在评测证明收益明显之后，再考虑把 semantic boundary detection 加进 section 内部

这个回答比“语义分块更高级”要成熟得多，因为它体现了成本、稳定性和评测思维。

---

### Q11：如果问“你们怎么保证索引更新不会出错”？

**推荐答法：**

我会从幂等性和一致性两个角度回答：

- 幂等性：同一个 `doc_id` 更新时先删旧版本再写新版本
- 一致性：child vectorstore 和 parent docstore 必须同时成功
- 失败恢复：任何一步出错时按文档级回滚
- 版本化：用新的 collection/version 避免直接覆盖旧索引

这能体现你不只是会“做 RAG”，而是会“运维 RAG”。

---

### Q12：ChromaDB 的 Embedding 向量怎么存储的？

**回答思路：**

ChromaDB 底层用 SQLite 存元数据，用 HNSW（Hierarchical Navigable Small World）算法建向量索引。

HNSW 是一种图索引结构，搜索复杂度 O(log N)，远优于暴力搜索的 O(N)。
它通过多层图结构，上层稀疏（用于快速定位区域），下层密集（用于精确匹配）。

查询过程：
1. 从上层图的入口节点出发
2. 贪心搜索：每步走向最近邻居
3. 逐层下降到底层，得到候选集
4. 从候选集中取 Top-K

---

### 面试加分建议：怎么把回答讲得更像做过生产

面试官最怕听到的，是一连串概念名词但没有工程判断。

建议你回答任何 RAG 问题时都尽量带上这 4 个维度：

1. **为什么这样设计**
2. **替代方案是什么**
3. **代价和边界是什么**
4. **怎么验证它真的有效**

举例：

- 不要只说“我们用了 parent-child”
- 更好的说法是：
  - “因为召回最优粒度和回答最优粒度不同，所以我们让 child 负责召回，parent 负责最终上下文。这样会多一个 docstore，但能显著降低碎片化回答的问题。”

## 八、延伸阅读与进阶路径

### 8.1 当前项目还缺什么（进阶方向）

| 功能 | 现状 | 改进方向 |
|------|------|---------|
| Reranking | 无 | 集成 Cohere Rerank 或 BGE-Reranker |
| Section 识别增强 | 基础规则版 | 标题层级树、列表/表格/代码块专项切分 |
| 语义断点增强 | 未做 embedding 级切点 | 在 section 内增加 semantic boundary detection |
| 流式输出 | 无 | 使用 `chain.astream()` + Streamlit streaming |
| 评估框架 | 无 | 集成 RAGAs 自动评估 |
| 多模态 | 无 | 支持图片、表格提取 |
| 索引迁移 | 已有版本化基础 | 增加双读双写、灰度切换、自动回滚 |
| 文档更新通知 | 无 | Webhook 触发重新索引 |

### 8.2 推荐学习资源

**论文（按重要性排序）：**
- `REALM: Retrieval-Augmented Language Model Pre-Training`（RAG 奠基论文）
- `Dense Passage Retrieval for Open-Domain Question Answering`（DPR，向量检索基础）
- `Reciprocal Rank Fusion outperforms Condorcet`（RRF 算法原论文）

**工程实践：**
- LangChain 官方文档的 RAG 章节
- ChromaDB 官方文档（理解 Collection、Embedding Function、Distance Function）
- `RAGAs` 评估框架文档

### 8.3 学完本项目，下一步学什么

本系列课程路径：
- **02**：LangGraph 多智能体（Agent 能主动行动，不只是回答问题）
- **03**：工具调用 Agent（让 AI 能搜索网页、查数据库、执行代码）
- **04**：多模态 Agent（处理图片、音频）
- **05**：生产化部署（监控、成本控制、A/B 测试）

---

*文档更新日期：2026-04-16*  
*对应代码版本：01_RAG chunking v2 / parent-child retrieval*
