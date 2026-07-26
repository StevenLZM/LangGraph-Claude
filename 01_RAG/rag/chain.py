"""One-pass RAG execution with seven-stage retrieval and evidence citations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

from config import DASHSCOPE_BASE_URL, DEEPSEEK_BASE_URL, llm_config
from rag.context_assembler import render_evidence_block
from rag.query_rewriter import rewrite_query
from rag.retrieval_trace import EvaluationTrace
from rag.retriever import retrieve_with_trace


EVIDENCE_INSUFFICIENT_ANSWER = "根据当前知识库，未找到该问题的相关信息。"
INVALID_CITATION_SAFE_ANSWER = (
    "根据当前知识库，无法生成带有有效证据引用的可靠回答。"
)

SYSTEM_PROMPT = """你是专业的企业知识库问答助手。

规则：
1. 只能依据“检索到的证据”回答，不得补充证据外的业务事实。
2. 每个事实性结论必须在句末引用一个或多个证据编号，例如 [S1] 或 [S1][S2]。
3. 只能引用实际提供的 [Sx]，不得编造引用。
4. 证据不足时回答：「根据当前知识库，未找到该问题的相关信息。」
5. 回答末尾列出实际使用的来源，格式为“来源：[S1] 文档名，第 X 页”。

检索到的证据：
{context}"""


@dataclass(frozen=True)
class RagExecution:
    answer: str
    source_documents: tuple[Document, ...]
    trace: EvaluationTrace
    generation_latency_ms: float


def _get_llm(model_name: str | None = None):
    """Get the configured chat model in project priority order."""
    provider = llm_config.provider()
    model = model_name or llm_config.CHAT_MODEL
    if provider == "deepseek":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=0.3,
            api_key=llm_config.DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            max_retries=3,
        )
    if provider == "dashscope":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=0.3,
            openai_api_key=llm_config.DASHSCOPE_API_KEY,
            openai_api_base=DASHSCOPE_BASE_URL,
            max_retries=3,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            temperature=0.3,
            api_key=llm_config.ANTHROPIC_API_KEY,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=0.3,
            api_key=llm_config.OPENAI_API_KEY,
        )
    raise EnvironmentError(
        "未配置可用的 API Key，请在 .env 中设置 DEEPSEEK_API_KEY"
    )


def _get_rewrite_llm():
    return _get_llm(llm_config.REWRITE_MODEL)


def format_docs_for_context(docs: Sequence[Document]) -> str:
    if not docs:
        return "（无相关文档内容）"
    prepared: list[Document] = []
    for index, document in enumerate(docs, start=1):
        metadata = dict(document.metadata)
        metadata.setdefault("evidence_id", f"S{index}")
        prepared.append(
            Document(
                page_content=document.page_content,
                metadata=metadata,
            )
        )
    return ("\n\n" + "─" * 40 + "\n\n").join(
        render_evidence_block(document)
        for document in prepared
    )


def generate_answer_from_documents(
    *,
    question: str,
    documents: Sequence[Document],
    chat_history: Sequence[Any],
) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )
    chain = prompt | _get_llm() | StrOutputParser()
    return str(
        chain.invoke(
            {
                "context": format_docs_for_context(documents),
                "chat_history": list(chat_history),
                "question": question,
            }
        )
    )


def run_rag_with_trace(
    question: str,
    *,
    chat_history: Sequence[Any] = (),
    auth_context: dict[str, Any] | None = None,
) -> RagExecution:
    history_text = _serialize_history(chat_history)
    rewrite = rewrite_query(
        question=question,
        chat_history=history_text,
    )
    rewritten_query = str(rewrite.get("rewritten_query") or question)
    retrieval = retrieve_with_trace(
        rewritten_query,
        retrieval_context={
            "original_query": question,
            "time_intent": rewrite.get("time_intent"),
            "auth_context": dict(auth_context or {}),
        },
    )
    documents = retrieval.final_documents
    if not documents:
        return RagExecution(
            answer=EVIDENCE_INSUFFICIENT_ANSWER,
            source_documents=documents,
            trace=retrieval.trace,
            generation_latency_ms=0.0,
        )

    started = perf_counter()
    answer = generate_answer_from_documents(
        question=question,
        documents=documents,
        chat_history=chat_history,
    )
    if not citations_are_valid(answer, documents):
        allowed = ", ".join(
            f"[{document.metadata['evidence_id']}]"
            for document in documents
        )
        answer = generate_answer_from_documents(
            question=(
                f"{question}\n\n上一次回答的证据引用无效。"
                f"只能使用这些引用：{allowed}；每个事实结论必须引用。"
            ),
            documents=documents,
            chat_history=chat_history,
        )
        if not citations_are_valid(answer, documents):
            answer = INVALID_CITATION_SAFE_ANSWER
    generation_latency_ms = (perf_counter() - started) * 1000
    return RagExecution(
        answer=answer,
        source_documents=documents,
        trace=retrieval.trace,
        generation_latency_ms=generation_latency_ms,
    )


def citations_are_valid(
    answer: str,
    documents: Sequence[Document],
) -> bool:
    allowed = {
        str(document.metadata.get("evidence_id") or "")
        for document in documents
    }
    allowed.discard("")
    cited = {
        f"S{match}"
        for match in re.findall(r"\[S(\d+)\]", str(answer))
    }
    return bool(cited) and cited.issubset(allowed)


def create_rag_chain():
    """Create a UI-compatible Runnable around the one-pass execution."""

    def invoke(input_dict: dict[str, Any]) -> dict[str, Any]:
        execution = run_rag_with_trace(
            str(input_dict["question"]),
            chat_history=input_dict.get("chat_history") or (),
            auth_context=input_dict.get("auth_context") or {},
        )
        return {
            "answer": execution.answer,
            "sources": list(execution.source_documents),
            "trace": execution.trace,
            "generation_latency_ms": execution.generation_latency_ms,
        }

    return RunnableLambda(invoke)


def create_chain_with_history():
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory

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


def _serialize_history(chat_history: Sequence[Any]) -> str:
    if not chat_history:
        return ""
    try:
        return "\n".join(
            f"{getattr(message, 'type', 'msg')}: "
            f"{getattr(message, 'content', '')}"
            for message in chat_history
        )
    except Exception:
        return str(chat_history)
