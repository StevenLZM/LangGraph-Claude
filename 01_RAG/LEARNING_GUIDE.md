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

### 1.2 完整数据流（必须背下来）

```
用户提问
  │
  ▼
[1] 查询改写 (chain.py)
    小模型(haiku)把"它的性能怎么样"改写成"LangGraph 的性能怎么样"
    解决多轮对话中代词指代问题
  │
  ▼
[2] 混合检索 (retriever.py)
    ├─ 语义检索：把问题向量化 → ChromaDB 相似度搜索 Top-6
    └─ BM25 检索：把问题分词 → jieba + rank-bm25 关键词匹配 Top-6
  │
  ▼
[3] RRF 融合排序 (retriever.py)
    Score = weight × Σ 1/(k + rank)
    语义权重 0.6 + BM25 权重 0.4 → 合并去重 → Top-4
  │
  ▼
[4] 相似度过滤 (vectorstore.py)
    score < 0.3 的块丢弃，防止"驴唇不对马嘴"的答案
  │
  ▼
[5] Prompt 组装 (chain.py)
    System Prompt + 检索到的上下文 + 历史对话 + 用户问题
  │
  ▼
[6] LLM 生成 (claude-sonnet)
    严格基于检索内容回答，禁止幻觉
  │
  ▼
[7] 提取来源并展示 (app.py)
    答案 + [文档名, 第X页]
```

---

### 1.3 技术栈总览

| 层次 | 组件 | 技术选型 | 说明 |
|------|------|---------|------|
| **文档解析** | `loader.py` | PyMuPDF / pypdf | PyMuPDF 更准确，pypdf 纯 Python 兜底 |
| **文本分块** | `chunker.py` | LangChain TextSplitter | 递归分块 + 句子感知两种策略 |
| **向量化** | `embedder.py` | DashScope / OpenAI / HuggingFace | 工厂模式，按 API Key 自动选择 |
| **向量存储** | `vectorstore.py` | ChromaDB | 本地持久化，支持元数据过滤 |
| **关键词检索** | `retriever.py` | jieba + rank-bm25 | 中文分词 + BM25 算法 |
| **混合排序** | `retriever.py` | RRF (Reciprocal Rank Fusion) | 业界标准混合排序算法 |
| **LLM 链** | `chain.py` | LangChain LCEL | 声明式 Pipeline 构建 |
| **多轮记忆** | `session.py` | 自研 SessionManager | FIFO 自动截断，会话隔离 |
| **UI** | `app.py` | Streamlit | 快速构建数据应用 |
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
CHUNK_SIZE = 500          # 每块最大字符数
CHUNK_OVERLAP = 50        # 相邻块重叠字符数（防止语义断裂）
SEMANTIC_TOP_K = 6        # 向量检索取前6个
BM25_TOP_K = 6            # BM25检索取前6个
FINAL_TOP_K = 4           # 最终给LLM的块数（太多会超出上下文窗口）
SEMANTIC_WEIGHT = 0.6     # 语义检索权重
SIMILARITY_THRESHOLD = 0.3  # 相似度门槛，低于此值丢弃
```

---

### 3.2 PDF 解析 `rag/loader.py`

**知识点：文档预处理是 RAG 精度的基础**

```python
# loader.py — 双引擎解析策略
def load_pdf(file_path: str) -> list[Document]:
    try:
        return _load_with_pymupdf(file_path)   # 主引擎：准确率高
    except Exception:
        return _load_with_pypdf(file_path)      # 备用：纯Python，无系统依赖
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

**知识点：分块是 RAG 效果最关键的超参数之一**

**策略一：递归分块（默认）**

```python
# chunker.py:30-55
separators = [
    "\n\n",  # 段落（最优先，语义完整）
    "\n",    # 换行
    "。", "！", "？",  # 中文句子
    ".", "!", "?",     # 英文句子
    "；", ";",
    "，", ",",
    " ",     # 单词
    "",      # 字符（最后手段）
]
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,      # 500字符
    chunk_overlap=chunk_overlap, # 50字符重叠
    separators=separators,
    length_function=len,        # 按字符数，非词数
)
```

**递归逻辑**：先尝试用 `\n\n`（段落）切分，如果某块仍超过 500 字符，再用 `\n` 切，以此类推，直到满足大小限制。

**策略二：句子感知分块**

```python
# chunker.py:70-100 — 确保块在句子边界结束
# 适合内容结构清晰的文档（论文、报告）
# 代价：块大小不均匀，可能有很小的块
```

**chunk_index 元数据（chunker.py:110）：**
```python
chunk.metadata["chunk_index"] = i  # 块在文档中的序号
# 作用：调试时追踪哪些块被检索到，评估分块质量
```

---

### 3.4 向量化 `rag/embedder.py`

**知识点：Embedding 是语义检索的核心**

Embedding 模型把文本转换为高维向量（如 1536 维），语义相似的文本在向量空间中距离近。

```python
# embedder.py — 工厂模式
def create_embeddings(config: LLMConfig) -> Embeddings:
    if config.dashscope_api_key:
        return DashScopeEmbeddings(model="text-embedding-v2")
    elif config.openai_api_key:
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        # 本地模型，无需 API Key，适合离线场景
        return HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
```

**批处理 + 重试（embedder.py:60-90）：**
```python
# 每批25个文档，避免 API 限流
# 指数退避重试（1s → 2s → 4s），最多3次
# 批间延迟 0.3s，保护 API 配额
```

**为什么不直接用一个 for 循环逐个 embed？**
大型文档可能有几百个块，逐个调用 API 极慢且容易触发限流。
批处理是生产系统的标准做法。

---

### 3.5 向量存储 `rag/vectorstore.py`

**知识点：幂等性是生产系统的必要特性**

```python
# vectorstore.py:45-70 — 幂等索引
def add_documents(chunks, doc_id, vectorstore):
    # 先删除同 doc_id 的旧版本
    _delete_by_doc_id(doc_id, vectorstore)
    # 再插入新版本
    vectorstore.add_documents(chunks)
```

**为什么需要幂等？**
用户上传同一文件两次（内容更新后），如果不删旧版本，向量库会有重复数据，
检索结果会出现相同内容的多个重复块，答案质量下降。

**相似度过滤（vectorstore.py:90-110）：**
```python
def similarity_search_with_threshold(query, k, threshold, vectorstore):
    # 带分数的搜索
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    # 过滤低分结果
    return [(doc, score) for doc, score in results if score >= threshold]
```

**ChromaDB 单例模式（vectorstore.py:20-30）：**
```python
_vectorstore_instance = None

def get_vectorstore(embeddings, persist_directory):
    global _vectorstore_instance
    if _vectorstore_instance is None:
        _vectorstore_instance = Chroma(...)  # 只初始化一次
    return _vectorstore_instance
```

原因：ChromaDB 初始化（加载索引文件）耗时，每次请求都初始化会很慢。

---

### 3.6 混合检索 `rag/retriever.py`

**知识点：这是本项目最有技术含量的模块**

#### 3.6.1 为什么要混合检索

| 检索方式 | 优点 | 缺点 | 适用场景 |
|---------|------|------|---------|
| 纯语义（向量） | 理解语义，同义词/近义词 | 对专有名词、代码效果差 | 概念性问题 |
| 纯关键词（BM25） | 精确匹配，对术语敏感 | 不理解语义，无法处理同义词 | 精确查找 |
| **混合** | 兼顾两者 | 实现复杂 | **生产场景默认选择** |

举例：用户问"模型推理延迟"
- 语义检索能找到"响应时间"、"吞吐量"相关的段落（语义相近）
- BM25 能精确找到含有"延迟"字样的段落
- 混合：两者都找到，最终排名更准

#### 3.6.2 BM25 原理

```
BM25(q, d) = Σ IDF(t) × [tf(t,d) × (k1+1)] / [tf(t,d) + k1 × (1-b+b×|d|/avgdl)]

其中：
- IDF(t)：词 t 的逆文档频率（稀有词权重高）
- tf(t,d)：词 t 在文档 d 中的频率
- k1=1.5, b=0.75：调节参数（可调优）
- |d|/avgdl：文档长度归一化
```

#### 3.6.3 RRF 融合算法

```python
# retriever.py:80-110
def _rrf_fusion(semantic_docs, bm25_docs, semantic_weight, bm25_weight, k=60):
    scores = {}
    
    # 语义检索得分：按排名加权
    for rank, doc in enumerate(semantic_docs):
        doc_id = doc.page_content[:50]  # 用内容前50字符作ID
        scores[doc_id] = scores.get(doc_id, 0) + semantic_weight / (k + rank + 1)
    
    # BM25 检索得分
    for rank, doc in enumerate(bm25_docs):
        doc_id = doc.page_content[:50]
        scores[doc_id] = scores.get(doc_id, 0) + bm25_weight / (k + rank + 1)
    
    # 按总分排序，取前 FINAL_TOP_K 个
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:FINAL_TOP_K]
```

**k=60 的含义**：平滑参数，防止排名第1的结果权重过高（避免"赢者通吃"）。
数值越大，排名差异对最终分数的影响越小。

---

### 3.7 RAG 链 `rag/chain.py`

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

**查询改写（chain.py:40-70）：**

```python
# 用小模型（haiku，便宜10倍）处理代词消解
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

---

### 3.8 会话记忆 `memory/session.py`

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

### 3.9 UI 主程序 `app.py`

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

### 4.1 先运行全部测试，建立整体感知

```bash
cd 01_RAG
python -m pytest tests/test_rag_pipeline.py -v 2>&1 | head -60
```

预期输出：
```
tests/test_rag_pipeline.py::TestDocumentLoader::test_load_pdf_returns_documents PASSED
tests/test_rag_pipeline.py::TestDocumentChunker::test_chunk_count PASSED
...
```

**不需要 API Key 的测试**（可以直接运行）：
- `TestDocumentLoader`（6个）：测试 PDF 解析
- `TestDocumentChunker`（7个）：测试分块逻辑
- `TestSessionMemory`（6个）：测试会话记忆
- `TestFilesystemClient`（4个）：测试文件管理

---

### 4.2 动手实验：逐步跑通完整流程

建议按如下顺序手动执行，在 Python REPL 中边运行边观察：

#### 实验1：理解文档加载

```python
# 在 01_RAG 目录下启动 Python
import sys
sys.path.insert(0, '.')

from rag.loader import load_pdf

# 加载一个 PDF
docs = load_pdf('data/documents/AI_Agent_技术白皮书.pdf')
print(f"页数：{len(docs)}")
print(f"\n第1页内容（前200字）：\n{docs[0].page_content[:200]}")
print(f"\n元数据：{docs[0].metadata}")
```

观察要点：
- `page_content` 是否干净（没有乱码、多余空行）
- `metadata` 里 `doc_id` 是否一致（同一文件所有页 doc_id 相同）

#### 实验2：理解分块效果

```python
from rag.chunker import chunk_documents

chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
print(f"页数：{len(docs)} → 块数：{len(chunks)}")
print(f"\n第一块（{len(chunks[0].page_content)}字符）：")
print(chunks[0].page_content)
print(f"\n第一块元数据：{chunks[0].metadata}")

# 查看相邻块的重叠
print("\n=== 观察重叠 ===")
print("块0末尾：", repr(chunks[0].page_content[-80:]))
print("块1开头：", repr(chunks[1].page_content[:80]))
```

观察要点：
- 块和块之间是否有50字符左右的重叠
- 块是否在合理的语义边界（如段落、句子）切割

#### 实验3：理解 Embedding

```python
from config import LLMConfig
from rag.embedder import create_embeddings

config = LLMConfig()
embeddings = create_embeddings(config)

# 单个文本向量化
vec = embeddings.embed_query("什么是 RAG？")
print(f"向量维度：{len(vec)}")
print(f"前5个值：{vec[:5]}")

# 两个语义相近的文本，向量应该相似
import numpy as np
vec1 = embeddings.embed_query("什么是检索增强生成")
vec2 = embeddings.embed_query("RAG 的全称和含义")
vec3 = embeddings.embed_query("今天天气怎么样")

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"\n语义相近的相似度：{cosine_sim(vec1, vec2):.4f}")  # 应该 > 0.8
print(f"语义无关的相似度：{cosine_sim(vec1, vec3):.4f}")  # 应该 < 0.3
```

**这个实验直观展示了向量检索的原理**。

#### 实验4：理解向量存储与检索

```python
from config import PathConfig, ChromaConfig, LLMConfig
from rag.vectorstore import get_vectorstore, add_documents, list_documents
from rag.embedder import create_embeddings

config = LLMConfig()
embeddings = create_embeddings(config)
path_config = PathConfig()

vs = get_vectorstore(embeddings, str(path_config.vectorstore_dir))

# 查看已有文档
docs_list = list_documents(vs)
print("已索引文档：")
for d in docs_list:
    print(f"  - {d['source']}：{d['total_chunks']}个块")

# 手动执行一次向量检索
results = vs.similarity_search_with_relevance_scores("RAG 检索增强生成", k=3)
for doc, score in results:
    print(f"\n相似度：{score:.4f}")
    print(f"来源：{doc.metadata.get('source')} 第{doc.metadata.get('page')}页")
    print(f"内容：{doc.page_content[:100]}...")
```

#### 实验5：理解混合检索

```python
from rag.retriever import create_hybrid_retriever
from config import RAGConfig

rag_config = RAGConfig()

# 假设你已经索引了文档，chunks 是所有块
retriever = create_hybrid_retriever(vs, chunks, rag_config)

# 执行检索
results = retriever.invoke("LangGraph 的节点和边如何定义")
print(f"检索到 {len(results)} 个块：")
for doc in results:
    print(f"\n来源：{doc.metadata.get('source')} p{doc.metadata.get('page')}")
    print(f"内容：{doc.page_content[:150]}...")
```

#### 实验6：完整 RAG 链（需要 API Key）

```python
from rag.chain import create_rag_chain
from memory.session import get_session_manager

manager = get_session_manager()
chain = create_rag_chain(vs, chunks, config)

# 第一轮对话
result1 = chain.invoke(
    {"question": "LangGraph 是什么？"},
    config={"configurable": {"session_id": "test_session"}}
)
print("回答1：", result1["answer"])
print("来源：", result1.get("sources", []))

# 第二轮（测试多轮对话感知）
result2 = chain.invoke(
    {"question": "它的核心概念有哪些？"},  # "它"指LangGraph
    config={"configurable": {"session_id": "test_session"}}
)
print("\n回答2：", result2["answer"])
# 关键：改写后的查询应该是 "LangGraph 的核心概念有哪些"
```

---

### 4.3 用测试文件学习断言技巧

打开 `tests/test_rag_pipeline.py` 重点看这些断言：

```python
# test_rag_pipeline.py:45-80 — 学习如何验证元数据完整性
def test_metadata_fields(self):
    docs = load_pdf(self.pdf_path)
    for doc in docs:
        assert "source" in doc.metadata, "缺少 source 字段"
        assert "page" in doc.metadata, "缺少 page 字段"
        assert "doc_id" in doc.metadata, "缺少 doc_id 字段"
        assert doc.metadata["page"] > 0, "页码必须为正数"

# test_rag_pipeline.py:110-130 — 学习如何验证分块质量
def test_chunk_size_respected(self):
    chunks = chunk_documents(self.docs, chunk_size=500, chunk_overlap=50)
    oversized = [c for c in chunks if len(c.page_content) > 500 * 1.5]
    # 允许10%的块轻微超出（因为在句子边界切割）
    assert len(oversized) / len(chunks) < 0.1, "超过10%的块大小超限"

# test_rag_pipeline.py:200-220 — 学习如何测试自动截断
def test_auto_trimming(self):
    manager = SessionManager(max_history=2)  # 最多保留2轮
    for i in range(5):  # 添加5轮对话
        manager.add_exchange("s1", f"问题{i}", f"回答{i}")
    messages = manager.get_or_create("s1").messages
    assert len(messages) == 4  # 只保留最新2轮（2×2=4条消息）
```

---

### 4.4 设计自己的测试用例（动手练习）

尝试编写以下测试，巩固理解：

```python
# 练习1：验证幂等索引
def test_idempotent_indexing():
    """相同文档上传两次，结果应和上传一次相同"""
    # 第一次索引
    add_documents(chunks, doc_id, vs)
    count1 = get_collection_stats(vs)["total_chunks"]
    
    # 第二次索引（相同 doc_id）
    add_documents(chunks, doc_id, vs)
    count2 = get_collection_stats(vs)["total_chunks"]
    
    assert count1 == count2, "幂等索引失败：出现重复数据"

# 练习2：验证相似度过滤
def test_similarity_threshold():
    """低相似度的结果应被过滤掉"""
    # 用完全不相关的查询
    results = similarity_search_with_threshold(
        "香蕉苹果西瓜水果",
        k=6, threshold=0.3, vectorstore=vs
    )
    # 知识库里全是 AI 文档，水果查询应该返回空
    assert len(results) == 0 or all(score >= 0.3 for _, score in results)

# 练习3：验证查询改写触发
def test_query_rewrite():
    """包含代词的查询应被改写"""
    rewritten = rewrite_query("它的实现原理是什么", "用户：LangGraph 是什么？\nAI：...")
    assert "LangGraph" in rewritten  # 代词应被替换为具体名词
```

---

## 五、分块策略调优指南

### 5.1 分块参数对效果的影响

```
chunk_size:  小 ←————————————————→ 大
             精确但上下文不足        上下文丰富但不精确

chunk_overlap: 小 ←——————→ 大
               可能错过跨边界信息   重复内容多，浪费存储和检索
```

### 5.2 不同内容类型的推荐参数

| 文档类型 | chunk_size | chunk_overlap | 策略 | 理由 |
|---------|-----------|--------------|------|------|
| 技术手册/文档 | 400-600 | 50-80 | 递归 | 结构清晰，段落语义独立 |
| 法律合同 | 800-1200 | 150-200 | 句子感知 | 条款上下文强相关，小块会丢失语义 |
| 学术论文 | 600-800 | 100 | 递归 | 摘要/结论/正文语义密度不同 |
| 新闻/短文 | 200-400 | 30-50 | 递归 | 单篇文章本身就是一个语义单元 |
| 代码文档 | 按函数切 | 0 | 自定义 | 代码块不应被截断 |
| 客服 FAQ | 按 Q&A 切 | 0 | 自定义 | 每条 Q&A 是原子单元 |

### 5.3 实际工作中如何调优分块

**第一步：建立评估数据集**

在正式调优前，必须有评估基准，否则"调优"只是在乱猜。

```python
# 评估数据集格式：问题 + 期望检索到的文档片段
eval_dataset = [
    {
        "question": "RRF 融合算法的参数 k 代表什么",
        "expected_source": "AI_Agent_技术白皮书.pdf",
        "expected_page": 4,
        "expected_keywords": ["平滑参数", "排名", "权重"]
    },
    ...
]
```

**第二步：量化评估指标**

```python
def evaluate_chunking(chunks, eval_dataset, vectorstore):
    hit_count = 0
    for item in eval_dataset:
        results = vectorstore.similarity_search(item["question"], k=4)
        # 检查期望内容是否在检索结果中
        hit = any(
            item["expected_source"] in doc.metadata.get("source", "") and
            any(kw in doc.page_content for kw in item["expected_keywords"])
            for doc in results
        )
        hit_count += hit
    return hit_count / len(eval_dataset)  # Recall@4
```

**第三步：系统性对比实验**

```python
# 对比不同分块参数，记录结果
experiments = [
    {"chunk_size": 300, "overlap": 30},
    {"chunk_size": 500, "overlap": 50},   # 当前默认
    {"chunk_size": 800, "overlap": 100},
    {"chunk_size": 500, "overlap": 100},  # 更大重叠
]

for params in experiments:
    chunks = chunk_documents(docs, **params)
    add_documents(chunks, doc_id, vs)
    score = evaluate_chunking(chunks, eval_dataset, vs)
    print(f"size={params['chunk_size']}, overlap={params['overlap']}: Recall={score:.2%}")
```

**实际工作经验（血泪教训）：**

1. **先别动 chunk_size**：文档质量（清洗是否干净）比分块参数影响大10倍
2. **overlap 宁大勿小**：overlap 太小是"省小钱花大钱"，多几个重复块成本可忽略
3. **混合文档类型分别建库**：技术文档和法律合同不要放同一个 collection
4. **小块（<200字）通常是噪声**：页码、标题、表格行等，需要在后处理阶段过滤

---

### 5.4 进阶：语义分块（Semantic Chunking）

当前项目用的是基于字符长度的机械切割，更高级的做法是**语义分块**：

```python
# 伪代码：检测相邻句子的语义相似度，在"主题切换点"切割
from langchain_experimental.text_splitter import SemanticChunker

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",  # 相似度跌破百分位时切割
    breakpoint_threshold_amount=95,           # 取相似度最低的5%作为切割点
)
```

**优点**：切割点在真正的语义边界，不会在句子中间切断
**缺点**：需要给每个句子做 Embedding，成本是普通分块的10-100倍，速度慢

**什么时候用？** 内容密集的专业文档（医学报告、法律条文），用户对精确度要求极高。

---

## 六、检索策略调优指南

### 6.1 混合权重调优

当前配置：`SEMANTIC_WEIGHT=0.6, BM25_WEIGHT=0.4`

**判断如何调整权重的方法：**

```python
# 准备两类测试问题
conceptual_questions = [
    "RAG 系统的工作原理",       # 语义理解型
    "为什么混合检索比单一检索好",
]

keyword_questions = [
    "SEMANTIC_TOP_K 的默认值是多少",  # 关键词精确匹配型
    "RRF 算法公式中 k 的取值",
]

# 分别测试 Recall
for weight in [0.3, 0.5, 0.6, 0.7, 0.9]:
    semantic_w = weight
    bm25_w = 1 - weight
    # 对两类问题分别评估
    ...
```

**经验规则：**
- 中文技术文档：语义 0.6 / BM25 0.4（当前配置是合理的）
- 英文学术论文：语义 0.7 / BM25 0.3（英文语义模型更强）
- 包含大量数字/代码：语义 0.4 / BM25 0.6（BM25 对精确术语更敏感）

### 6.2 Reranking 策略

**当前项目没有实现 Reranking，这是一个重要的调优方向。**

Reranking 的位置在 RRF 融合之后，对最终的 Top-K 再次精排：

```
RRF 融合 Top-10
    ↓ Reranking（Cross-Encoder）
精排后 Top-4 → 送入 LLM
```

**为什么要 Reranking？**

向量检索（Bi-Encoder）：query 和 doc 分别编码，速度快但精度有限
Cross-Encoder：把 query+doc 拼接输入，精度高但速度慢（不能对全库做）

两阶段策略：用 Bi-Encoder 粗筛（Top-50），用 Cross-Encoder 精排（Top-4）

**在当前项目中添加 Reranking：**

```python
# 方案A：使用 Cohere Rerank（商业 API，效果最好）
from langchain_cohere import CohereRerank

reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=4)
reranked_docs = reranker.compress_documents(retrieved_docs, query)

# 方案B：使用本地 Cross-Encoder（免费，中文效果不如 Cohere）
from sentence_transformers import CrossEncoder
model = CrossEncoder("BAAI/bge-reranker-v2-m3")
scores = model.predict([(query, doc.page_content) for doc in retrieved_docs])
sorted_docs = [doc for _, doc in sorted(zip(scores, retrieved_docs), reverse=True)][:4]
```

**实际工作中什么时候加 Reranking？**

| 情况 | 建议 |
|------|------|
| 召回率 OK，精确率低（用户说"答案不相关"）| 加 Reranking |
| 召回率低（根本找不到相关内容）| 先优化分块和检索，Reranking 帮不了 |
| 文档库 < 1万块，响应速度要求低 | 值得尝试 |
| 文档库 > 10万块，P99 < 2s | 谨慎，Reranking 增加200-500ms |

### 6.3 相似度阈值调优

```
threshold: 低 ←—————————————→ 高
0.1：几乎不过滤，很多不相关结果  0.7：过滤太严，容易"找不到"
```

**自动化找最优阈值：**

```python
# 准备两类样本
relevant_samples = [...]   # 应该被检索到的
irrelevant_samples = [...] # 不应该被检索到的

best_f1 = 0
best_threshold = 0.3

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    tp = sum(1 for q, expected_source in relevant_samples
             if any(d.metadata["source"] == expected_source
                    for d, s in similarity_search_with_threshold(q, threshold=threshold)))
    fp = sum(1 for q in irrelevant_samples
             if similarity_search_with_threshold(q, threshold=threshold))
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / len(relevant_samples)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    if f1 > best_f1:
        best_f1, best_threshold = f1, threshold

print(f"最优阈值：{best_threshold}，F1={best_f1:.3f}")
```

### 6.4 Top-K 参数调优

**FINAL_TOP_K 的影响：**

```
FINAL_TOP_K: 小 ←—————→ 大
             Token 少，答案可能不全  Token 多，成本高，上下文干扰也多
```

**实际工作中的做法：**

```python
# 先用大 K（如10）看 LLM 能否生成好答案
# 再逐步减小 K，观察答案质量何时开始下降
# 找到质量和成本的平衡点

# 动态 K：根据问题类型调整
def get_top_k(question: str) -> int:
    if any(kw in question for kw in ["对比", "比较", "区别"]):
        return 8  # 比较类问题需要更多上下文
    elif any(kw in question for kw in ["是什么", "定义"]):
        return 3  # 定义类问题一块就够
    else:
        return 4  # 默认
```

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
我们用的是**混合检索（Hybrid Retrieval）**，结合向量语义检索和 BM25 关键词检索。

- 向量检索（Dense）：把文本编码成高维向量，用余弦相似度匹配。能理解语义和同义词，但对专有名词、代码、数字不敏感。
- BM25（Sparse）：基于词频统计的传统算法。对精确关键词匹配效果好，但不理解语义。

用 **RRF（倒数排名融合）** 算法把两路结果合并：
`score = Σ weight_i / (k + rank_i)`，权重语义 0.6，BM25 0.4。

实测比单一方法提升约 15-20% 的召回率，特别是在"用户用自然语言提问，答案里是专业术语"的场景。

---

### Q3：分块策略怎么选？影响是什么？

**回答思路：**
分块是 RAG 中最关键、最容易被忽视的环节。

核心矛盾：块太小 → 上下文不足，LLM 无法给出完整答案；块太大 → 检索时匹配噪声多，相关内容被淹没。

我们的策略：
- 默认用**递归字符分块**，chunk_size=500，overlap=50
- overlap 确保语义不在块边界断裂
- 分隔符优先级：段落 > 句子 > 词 > 字符

调优方式：建立评估数据集（问题-期望答案对），系统性测试不同参数组合，用 Recall@K 作为量化指标。

---

### Q4：如何处理多轮对话中的代词问题？

**回答思路：**
这是 RAG 系统中容易被忽略的问题。用户说"它的原理是什么"，"它"指什么？

我们的解决方案是**查询改写（Query Rewriting）**：
在检索之前，用一个小模型（如 Claude Haiku）结合对话历史，把含代词的问题改写为完整独立的问题。

"LangGraph 是什么？" → "它的核心概念有哪些？" → 改写为 → "LangGraph 的核心概念有哪些？"

关键设计：用小模型（比 Sonnet 便宜10倍）做改写，主模型只用于最终回答生成，这样既解决了问题又控制了成本。

---

### Q5：向量数据库怎么选型？

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

### Q6：如何评估 RAG 系统的效果？

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

工具：`RAGAs` 框架可以自动化评估这两层指标，用 LLM 打分。

---

### Q7：如何防止 LLM 幻觉？

**回答思路：**

我们从三个层次防止幻觉：

1. **Prompt 层面**：System Prompt 明确规定"只能基于提供的上下文回答，如果找不到相关信息，明确说找不到，不得推测"
2. **检索层面**：相似度阈值过滤（score < 0.3 的结果不送入 LLM），宁可说"找不到"也不给低质量上下文
3. **验证层面**（进阶）：让另一个 LLM 检查答案中的每个事实是否能在检索结果中找到支撑

---

### Q8：ChromaDB 的 Embedding 向量怎么存储的？

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

## 八、延伸阅读与进阶路径

### 8.1 当前项目还缺什么（进阶方向）

| 功能 | 现状 | 改进方向 |
|------|------|---------|
| Reranking | 无 | 集成 Cohere Rerank 或 BGE-Reranker |
| 语义分块 | 无 | 集成 SemanticChunker |
| 流式输出 | 无 | 使用 `chain.astream()` + Streamlit streaming |
| 评估框架 | 无 | 集成 RAGAs 自动评估 |
| 多模态 | 无 | 支持图片、表格提取 |
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

*文档生成日期：2026-04-13*  
*对应代码版本：01_RAG（参见 README.md）*
