from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.intent import (
    extract_order_id,
    is_faq_query,
    is_human_transfer_request,
    is_logistics_query,
    is_memory_recall_query,
    is_order_query,
    is_preference_statement,
    is_product_query,
    is_refund_request,
)


@dataclass(frozen=True)
class RetrievalDecision:
    intent: str
    needs_memory: bool
    needs_faq: bool
    needs_business_context: bool
    memory_filters: dict[str, Any] = field(default_factory=dict)
    external_sources: list[str] = field(default_factory=list)
    query_rewrite: str = ""
    confidence: float = 0.0
    decision_source: str = "rules"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rule_based_retrieval_decision(message: str, *, decision_source: str = "rules") -> RetrievalDecision:
    query = " ".join(message.split())
    order_id = extract_order_id(query)
    external_sources: list[str] = []
    memory_filters: dict[str, Any] = {}

    intent = "fallback"
    needs_memory = False
    needs_faq = False
    needs_business_context = False
    confidence = 0.62

    if is_human_transfer_request(query):
        intent = "human_transfer"
        needs_business_context = bool(order_id)
        confidence = 0.9
    elif is_memory_recall_query(query):
        intent = "memory_recall"
        needs_memory = True
        memory_filters = {"category": ["delivery_preference", "preference", "profile", "complaint"]}
        confidence = 0.88
    elif is_preference_statement(query):
        intent = "memory_write"
        needs_memory = False
        confidence = 0.84
    elif is_faq_query(query):
        intent = "faq"
        needs_faq = True
        confidence = 0.86
    elif is_refund_request(query):
        intent = "refund"
        needs_business_context = bool(order_id)
        needs_memory = True
        memory_filters = {"category": ["complaint", "preference"]}
        confidence = 0.82
    elif is_logistics_query(query):
        intent = "logistics"
        needs_business_context = bool(order_id)
        needs_memory = True
        memory_filters = {"category": ["delivery_preference", "preference"]}
        confidence = 0.86
    elif order_id and is_order_query(query):
        intent = "order"
        needs_business_context = True
        confidence = 0.86
    elif is_product_query(query):
        intent = "product"
        needs_business_context = True
        needs_memory = True
        memory_filters = {"category": ["preference"]}
        confidence = 0.78

    if needs_memory:
        external_sources.append("long_term_memory")
    if needs_faq:
        external_sources.append("faq_rag")
    if needs_business_context:
        external_sources.append("business_tools")

    return RetrievalDecision(
        intent=intent,
        needs_memory=needs_memory,
        needs_faq=needs_faq,
        needs_business_context=needs_business_context,
        memory_filters=memory_filters,
        external_sources=external_sources,
        query_rewrite=query,
        confidence=confidence,
        decision_source=decision_source,
    )


async def decide_retrieval(
    message: str,
    *,
    llm: Any | None = None,
    mode: str = "rules",
) -> RetrievalDecision:
    if mode.casefold() != "llm" or llm is None:
        return rule_based_retrieval_decision(message)

    try:
        response = await llm.ainvoke(_decision_prompt(message))
        raw_content = str(getattr(response, "content", response))
        return _decision_from_json(raw_content, original_query=message)
    except Exception as exc:
        fallback = rule_based_retrieval_decision(message, decision_source="rules_fallback")
        return RetrievalDecision(**{**fallback.to_dict(), "error": str(exc)})


def _decision_prompt(message: str) -> list[object]:
    return [
        SystemMessage(
            content=(
                "你是客服系统的检索决策器。只输出 JSON，不要输出解释。"
                "字段：intent, needs_memory, needs_faq, needs_business_context, "
                "memory_filters, external_sources, query_rewrite, confidence。"
                "不要执行业务操作，只判断需要哪些上下文。"
            )
        ),
        HumanMessage(content=f"用户问题：{message}"),
    ]


def _decision_from_json(raw_content: str, *, original_query: str) -> RetrievalDecision:
    payload = json.loads(_extract_json_object(raw_content))
    external_sources = [str(item) for item in payload.get("external_sources", [])]
    memory_filters = payload.get("memory_filters") or {}
    if not isinstance(memory_filters, dict):
        memory_filters = {}
    confidence = float(payload.get("confidence", 0.0))
    return RetrievalDecision(
        intent=str(payload.get("intent") or "fallback"),
        needs_memory=bool(payload.get("needs_memory", False)),
        needs_faq=bool(payload.get("needs_faq", False)),
        needs_business_context=bool(payload.get("needs_business_context", False)),
        memory_filters=memory_filters,
        external_sources=external_sources,
        query_rewrite=str(payload.get("query_rewrite") or original_query),
        confidence=max(0.0, min(1.0, confidence)),
        decision_source="llm",
    )


def _extract_json_object(raw_content: str) -> str:
    text = raw_content.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("retrieval decision did not contain a JSON object")
    return text[start : end + 1]
