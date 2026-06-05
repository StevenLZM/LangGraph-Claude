from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.retrieval import decide_retrieval, rule_based_retrieval_decision


class JSONDecisionLLM:
    async def ainvoke(self, messages):
        del messages
        return SimpleNamespace(
            content=(
                '{"intent":"memory_recall","needs_memory":true,'
                '"needs_faq":false,"needs_business_context":false,'
                '"memory_filters":{"category":["delivery_preference"]},'
                '"external_sources":["long_term_memory"],'
                '"query_rewrite":"配送偏好","confidence":0.91}'
            )
        )


@pytest.mark.asyncio
async def test_llm_retrieval_decision_parses_structured_json():
    decision = await decide_retrieval(
        "你记得我的配送偏好吗？",
        llm=JSONDecisionLLM(),
        mode="llm",
    )

    assert decision.intent == "memory_recall"
    assert decision.needs_memory is True
    assert decision.needs_faq is False
    assert decision.memory_filters == {"category": ["delivery_preference"]}
    assert decision.external_sources == ["long_term_memory"]
    assert decision.query_rewrite == "配送偏好"
    assert decision.confidence == 0.91
    assert decision.decision_source == "llm"


@pytest.mark.asyncio
async def test_llm_retrieval_decision_falls_back_to_rules_on_invalid_json():
    class InvalidLLM:
        async def ainvoke(self, messages):
            del messages
            return "not-json"

    decision = await decide_retrieval(
        "帮我查一下物流 ORD123456",
        llm=InvalidLLM(),
        mode="llm",
    )

    assert decision.intent == "logistics"
    assert decision.needs_business_context is True
    assert decision.decision_source == "rules_fallback"


def test_rule_based_retrieval_decision_routes_memory_faq_and_business_context():
    memory = rule_based_retrieval_decision("你记得我的配送偏好吗？")
    faq = rule_based_retrieval_decision("退货政策是什么？")
    logistics = rule_based_retrieval_decision("我的订单 ORD123456 到哪了？")

    assert memory.needs_memory is True
    assert memory.needs_faq is False
    assert "long_term_memory" in memory.external_sources

    assert faq.needs_faq is True
    assert "faq_rag" in faq.external_sources

    assert logistics.needs_business_context is True
    assert logistics.intent == "logistics"
