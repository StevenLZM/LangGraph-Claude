"""
rag/chain.py — RAG Chain 构建（LCEL）
支持：DashScope (千问) > Anthropic > OpenAI
流程：问题改写 → 混合检索 → 上下文构建 → LLM 生成
"""
from __future__ import annotations
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document

from config import llm_config, DASHSCOPE_BASE_URL
from rag.vectorstore import get_vectorstore, similarity_search_with_threshold


# ────────────────────────────────────────────────────────────────
# LLM 工厂（优先 DashScope）
# ────────────────────────────────────────────────────────────────
def _get_llm(model_name: str | None = None):
    """获取对话 LLM，按 DashScope > Anthropic > OpenAI 优先级"""
    provider = llm_config.provider()
    model = model_name or llm_config.CHAT_MODEL

    if provider == "dashscope":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=0.3,
            openai_api_key=llm_config.DASHSCOPE_API_KEY,
            openai_api_base=DASHSCOPE_BASE_URL,
            max_retries=3,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=0.3,
            api_key=llm_config.ANTHROPIC_API_KEY,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=0.3,
            api_key=llm_config.OPENAI_API_KEY,
        )
    else:
        raise EnvironmentError(
            "未配置可用的 API Key，请在 .env 中设置 DASHSCOPE_API_KEY"
        )


def _get_rewrite_llm():
    """问题改写用轻量模型（省 Token 成本）"""
    return _get_llm(llm_config.REWRITE_MODEL)


# ────────────────────────────────────────────────────────────────
# Prompt 模板
# ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是一个专业的企业知识库问答助手。

【行为准则】
1. 严格基于提供的"检索到的文档内容"回答问题
2. 如果文档中没有相关信息，明确告知：「根据当前知识库，未找到该问题的相关信息」
3. 回答时引用来源，在相关句子后标注 [来源: 文档名, 第X页]
4. 保持专业、简洁，避免直接复述文档原文
5. 支持追问，理解对话中的代词指代

【禁止行为】
- 不补充文档未提及的通用知识
- 不猜测或推断文档中未明确说明的内容
- 不编造数据或参考来源

检索到的文档内容：
{context}"""

REWRITE_PROMPT = """将对话中的最新用户问题改写为独立完整的问题。

规则：
1. 将代词还原为具体名词（"它" → 实际名称）
2. 保留原问题意图，不增删信息
3. 若问题已独立完整，直接返回原问题

对话历史：
{chat_history}

最新问题：{question}

独立问题（仅输出问题，不加其他文字）："""


# ────────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────────
def format_docs_for_context(docs: list[Document]) -> str:
    """格式化检索结果为 LLM Context"""
    if not docs:
        return "（无相关文档内容）"

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "?")
        score = doc.metadata.get("similarity_score", "")
        score_str = f" | 相似度: {score}" if score else ""
        parts.append(
            f"【文档{i} | 来源: {source} | 第{page}页{score_str}】\n"
            f"{doc.page_content}"
        )

    return ("\n\n" + "─" * 40 + "\n\n").join(parts)


# ────────────────────────────────────────────────────────────────
# Chain 构建
# ────────────────────────────────────────────────────────────────
def create_rag_chain():
    """构建完整 RAG LCEL Chain（问题改写 → 检索 → 生成）"""
    llm = _get_llm()
    rewrite_llm = _get_rewrite_llm()
    vs = get_vectorstore()

    # Step 1: 问题改写
    question_rewriter = (
        ChatPromptTemplate.from_template(REWRITE_PROMPT)
        | rewrite_llm
        | StrOutputParser()
        | (lambda x: x.strip())
    )

    # Step 2: 检索函数
    def retrieve_docs(input_dict: dict) -> list[Document]:
        q = input_dict.get("standalone_question") or input_dict.get("question", "")
        if not q:
            return []
        return similarity_search_with_threshold(query=q, k=4, vectorstore=vs)

    # Step 3: RAG Prompt
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

    # Step 4: 完整 LCEL Chain
    rag_chain = (
        RunnablePassthrough.assign(
            standalone_question=question_rewriter
        )
        | RunnablePassthrough.assign(
            docs=RunnableLambda(retrieve_docs)
        )
        | RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: format_docs_for_context(x["docs"]))
        )
        | {
            "answer": rag_prompt | llm | StrOutputParser(),
            "sources": RunnableLambda(lambda x: x["docs"]),
        }
    )

    return rag_chain


def create_chain_with_history():
    """创建带对话记忆的 RAG Chain"""
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_community.chat_message_histories import ChatMessageHistory

    rag_chain = create_rag_chain()
    session_store: dict[str, ChatMessageHistory] = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in session_store:
            session_store[session_id] = ChatMessageHistory()
        return session_store[session_id]

    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return chain_with_history, get_session_history
