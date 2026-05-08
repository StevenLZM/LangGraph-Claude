from __future__ import annotations

from types import SimpleNamespace

from rag.faq_tool import FAQRAGResult, FAQRAGTool


def test_faq_rag_tool_returns_no_match_when_backend_is_unavailable():
    tool = FAQRAGTool(backend_loader=lambda: None)

    result = tool.search("退换货政策是什么？")

    assert result == FAQRAGResult(
        matched=False,
        answer="根据当前知识库，未找到该问题的相关信息。",
        sources=[],
        backend="unavailable",
        error="knowledge base backend is not configured",
    )


def test_faq_rag_tool_formats_answer_and_sources_from_backend_documents():
    docs = [
        SimpleNamespace(
            page_content="签收后 7 天内支持无理由退货，商品需保持完好。",
            metadata={"source": "售后政策.pdf", "page_range": "3-4"},
        ),
        SimpleNamespace(
            page_content="退款审核通常需要 1-3 个工作日。",
            metadata={"source": "退款说明.pdf", "page": 2},
        ),
    ]
    backend = SimpleNamespace(invoke=lambda query: docs)
    tool = FAQRAGTool(backend_loader=lambda: backend)

    result = tool.search("退款和退货政策")

    assert result.matched is True
    assert "签收后 7 天内支持无理由退货" in result.answer
    assert "退款审核通常需要 1-3 个工作日" in result.answer
    assert result.sources == ["售后政策.pdf#3-4", "退款说明.pdf#2"]
    assert result.backend == "01_RAG"


def test_customer_service_routes_faq_questions_to_rag_tool(monkeypatch):
    import agent.service as service

    monkeypatch.setattr(
        service,
        "faq_rag_tool",
        FAQRAGTool(
            backend_loader=lambda: SimpleNamespace(
                invoke=lambda query: [
                    SimpleNamespace(
                        page_content="会员退货政策：签收 7 天内可申请退货。",
                        metadata={"source": "FAQ.md", "page": 1},
                    )
                ]
            )
        ),
    )

    decision = service.handle_customer_message("请问会员退货政策是什么？")

    assert decision.tool_name == "faq_rag"
    assert "会员退货政策" in decision.answer
    assert decision.order_context == {
        "rag_matched": True,
        "rag_sources": ["FAQ.md#1"],
        "rag_backend": "01_RAG",
    }
