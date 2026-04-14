# 工程设计：RAG 知识库问答系统

> 对应 PRD：01_rag_knowledge_base.md

---

## 一、整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
│   [上传PDF]  [提问输入框]  [历史对话]  [来源展示]  [配置面板]     │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP / WebSocket
┌───────────────────────────▼─────────────────────────────────────┐
│                      应用核心层                                   │
│  ┌─────────────────┐   ┌────────────────┐   ┌────────────────┐  │
│  │  DocumentLoader  │   │  RAG Chain     │   │MemoryManager  │  │
│  │  (PDF解析+分块)  │   │  (检索+生成)   │   │(对话记忆管理)  │  │
│  └────────┬────────┘   └───────┬────────┘   └───────┬────────┘  │
└───────────┼────────────────────┼────────────────────┼───────────┘
            │                    │                    │
┌───────────▼────────────────────▼────────────────────▼───────────┐
│                       基础设施层                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  ChromaDB    │  │  OpenAI API  │  │   MCP File Server    │   │
│  │ (向量存储)   │  │ (Embed+Chat) │  │  (文件读取/管理)     │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、详细流程设计

### 2.1 文档摄入流程（Ingestion Pipeline）

```
PDF文件
  │
  ▼
┌─────────────────────────────────────────────┐
│  Step 1: 文档解析                            │
│  · PyMuPDF 提取文本                          │
│  · 保留 metadata: {filename, page, total_pages} │
│  · 检测并跳过图片/表格页（可选 OCR 扩展）    │
└───────────────────────┬─────────────────────┘
                        │ raw_text + metadata
                        ▼
┌─────────────────────────────────────────────┐
│  Step 2: 文本清洗                            │
│  · 去除页眉页脚（正则匹配页码格式）           │
│  · 合并跨页断句（末尾无标点则拼接下页首行）   │
│  · 统一空白符处理                            │
└───────────────────────┬─────────────────────┘
                        │ cleaned_text
                        ▼
┌─────────────────────────────────────────────┐
│  Step 3: 语义分块 (Semantic Chunking)        │
│                                              │
│  策略A: RecursiveCharacterTextSplitter       │
│    chunk_size=500, overlap=50               │
│    separators=["。","！","？","\n\n","\n"]   │
│                                              │
│  策略B: SemanticChunker (语义相似度分块)     │
│    基于 embedding 相似度决定分块边界         │
│    适合逻辑连贯性强的文档                    │
└───────────────────────┬─────────────────────┘
                        │ chunks[]
                        ▼
┌─────────────────────────────────────────────┐
│  Step 4: 向量化 & 存储                       │
│  · 批量调用 text-embedding-3-small           │
│  · 每批 100 个 chunk，避免超限               │
│  · 存入 ChromaDB，collection 以文档ID命名    │
│  · metadata 写入: source, page, chunk_index  │
└─────────────────────────────────────────────┘
```

### 2.2 问答流程（Query Pipeline）

```
用户问题
  │
  ▼
┌─────────────────────────────────────────────┐
│  Step 1: 问题改写 (Query Rewriting)          │
│  · 多轮对话时，结合历史将指代词还原          │
│  · "它的价格？" → "iPhone 15 的价格？"      │
│  · 使用小模型执行，节省成本                  │
└───────────────────────┬─────────────────────┘
                        │ standalone_question
                        ▼
┌─────────────────────────────────────────────┐
│  Step 2: 混合检索 (Hybrid Retrieval)         │
│                                              │
│  ┌───────────────┐    ┌────────────────────┐ │
│  │ 语义检索       │    │  关键词检索 (BM25) │ │
│  │ Top-K=6       │    │  Top-K=6          │ │
│  └───────┬───────┘    └──────────┬─────────┘ │
│          └──────────┬────────────┘           │
│                     ▼                        │
│             RRF 融合排序                      │
│         (Reciprocal Rank Fusion)             │
│             最终 Top-K=4                     │
└───────────────────────┬─────────────────────┘
                        │ relevant_docs[]
                        ▼
┌─────────────────────────────────────────────┐
│  Step 3: 上下文压缩 (Context Compression)    │
│  · LLMChainExtractor 从每个 chunk 中         │
│    仅提取与问题相关的句子                     │
│  · 减少无关信息噪声，节省 token               │
└───────────────────────┬─────────────────────┘
                        │ compressed_context
                        ▼
┌─────────────────────────────────────────────┐
│  Step 4: 生成回答                            │
│  Prompt 结构:                                │
│  ┌─────────────────────────────────────┐    │
│  │ System: 你是知识库问答助手...        │    │
│  │ Context: {compressed_context}       │    │
│  │ History: {chat_history}             │    │
│  │ Human: {question}                   │    │
│  │ 规则: 仅基于Context回答，无关则拒绝  │    │
│  └─────────────────────────────────────┘    │
└───────────────────────┬─────────────────────┘
                        │ answer + source_docs
                        ▼
┌─────────────────────────────────────────────┐
│  Step 5: 答案后处理                          │
│  · 提取 source_docs 的 filename + page       │
│  · 格式化为 [来源: 手册.pdf, 第3页]          │
│  · 写入对话记忆（Human/AI消息对）            │
└─────────────────────────────────────────────┘
```

---

## 三、MCP 集成设计

### 3.1 使用的 MCP Server

```yaml
# .mcp.json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data/documents"],
      "description": "文件系统访问，用于读取和管理上传的PDF文件"
    }
  }
}
```

### 3.2 MCP 工具调用场景

| MCP 工具 | 调用时机 | 用途 |
|----------|----------|------|
| `filesystem/read_file` | 用户上传后 | 读取 PDF 二进制内容 |
| `filesystem/list_directory` | 初始化时 | 扫描已有文档列表 |
| `filesystem/delete_file` | 用户删除文档 | 清理原始文件 |

### 3.3 MCP 调用代码

```python
# 通过 MCP 读取文件（Claude Desktop / Claude Code 环境）
async def read_document_via_mcp(file_path: str) -> bytes:
    """使用 MCP filesystem server 读取文档"""
    result = await mcp_client.call_tool(
        "filesystem",
        "read_file",
        {"path": file_path}
    )
    return result.content
```

---

## 四、LangChain 组件设计

### 4.1 LCEL Chain 构建

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的知识库问答助手。
请严格基于以下检索到的文档内容回答问题。
如果文档中没有相关信息，请明确告知用户"文档中未找到相关信息"，不要猜测。

检索到的文档内容：
{context}
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# 问题改写链（多轮对话场景）
question_rewriter = (
    ChatPromptTemplate.from_messages([
        ("system", "根据对话历史，将用户问题改写为独立完整的问题（不改变原意）"),
        MessagesPlaceholder("chat_history"),
        ("human", "改写此问题为独立问题：{question}")
    ])
    | small_llm
    | StrOutputParser()
)

# 文档格式化函数
def format_docs(docs: list) -> str:
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[文档{i+1} | 来源:{source} | 第{page}页]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

# 主 RAG 链
rag_chain = (
    RunnablePassthrough.assign(
        standalone_question=question_rewriter  # 改写问题
    )
    | RunnablePassthrough.assign(
        docs=lambda x: hybrid_retriever.get_relevant_documents(x["standalone_question"]),
    )
    | RunnablePassthrough.assign(
        context=lambda x: format_docs(x["docs"])
    )
    | {
        "answer": prompt | llm | StrOutputParser(),
        "sources": lambda x: x["docs"]
    }
)
```

### 4.2 混合检索器

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

def build_hybrid_retriever(vectorstore: Chroma, documents: list):
    # 语义检索器
    semantic_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )
    
    # 关键词检索器（BM25）
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 6
    
    # RRF 融合
    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.6, 0.4]  # 语义检索权重更高
    )
    
    return ensemble_retriever
```

### 4.3 对话记忆管理

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 每个 session_id 对应独立的历史
store: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 带记忆的 RAG 链
chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
    output_messages_key="answer"
)
```

---

## 五、Prompt 工程设计

### 5.1 System Prompt

```
你是一个专业的企业知识库问答助手，基于提供的文档内容回答问题。

【行为准则】
1. 仅使用"检索到的文档内容"中的信息作答
2. 如信息不在文档中，回答："根据现有文档，未找到该问题的相关信息"
3. 回答需引用来源：在相关句子后标注 [来源: 文档名, 第X页]
4. 保持专业、简洁，避免重复文档原文
5. 支持追问，理解上下文中的代词指代

【禁止行为】
- 不基于通用知识补充文档未提及的信息
- 不猜测或推断文档中未明确说明的内容
```

### 5.2 问题改写 Prompt

```
你的任务是将对话中的最新问题改写为独立完整的问题。

改写规则：
- 将代词还原为具体名词（"它" → 具体实体名）
- 保留原问题的意图，不增加也不减少信息
- 如果最新问题本身就是独立完整的，直接返回原问题

对话历史：{chat_history}
最新问题：{question}

独立问题：
```

---

## 六、目录结构

```
01_rag_knowledge_base/
├── app.py                    # Streamlit 主程序
├── rag/
│   ├── __init__.py
│   ├── loader.py             # PDF 解析与清洗
│   ├── chunker.py            # 文本分块策略
│   ├── embedder.py           # Embedding 封装
│   ├── vectorstore.py        # ChromaDB 管理
│   ├── retriever.py          # 混合检索器
│   └── chain.py              # LCEL RAG 链
├── memory/
│   └── session.py            # 会话记忆管理
├── mcp/
│   └── filesystem_client.py  # MCP 文件系统调用
├── config.py                 # 配置（chunk_size, top_k 等）
├── .env.example
└── requirements.txt
```

---

## 七、关键配置参数

```python
# config.py
RAG_CONFIG = {
    # 文档分块
    "chunk_size": 500,
    "chunk_overlap": 50,
    
    # 检索
    "semantic_top_k": 6,
    "bm25_top_k": 6,
    "final_top_k": 4,
    "semantic_weight": 0.6,
    
    # 模型
    "embedding_model": "text-embedding-3-small",
    "chat_model": "claude-sonnet-4-6",
    "rewrite_model": "claude-haiku-4-5-20251001",  # 小模型做改写，省成本
    
    # 对话
    "max_history_messages": 10,  # 最多保留10轮历史
    
    # 质量控制
    "similarity_threshold": 0.3,  # 低于此值的检索结果丢弃
}
```

---

## 八、测试方案

```python
# tests/test_rag.py

# 测试1: 文档内容直接问答
assert "第3页" in answer_sources("产品的保修期是多久？")

# 测试2: 跨页问题
assert len(retrieve("安装步骤")) >= 2  # 多页内容都被召回

# 测试3: 超出文档范围
response = ask("股票今天涨了吗？")
assert "未找到" in response or "不在文档" in response

# 测试4: 多轮代词消解
ask("iPhone 15 有哪些颜色？")
response = ask("它的价格是多少？")
assert "iPhone 15" in get_rewritten_question()

# 测试5: 相似度阈值过滤
results = retriever.get_relevant_documents("完全不相关的内容xyz")
assert all(r.metadata["score"] > 0.3 for r in results)
```
