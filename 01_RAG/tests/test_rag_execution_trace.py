from langchain_core.documents import Document

from rag.retrieval_trace import EvaluationTrace
from rag.retriever import RetrievalExecution


def _final_documents():
    return [
        Document(
            page_content="产品保修期为 12 个月。",
            metadata={
                "doc_id": "doc-1",
                "parent_id": "parent-1",
                "source": "manual.pdf",
                "page_range": "3",
                "evidence_id": "S1",
            },
        )
    ]


def _retrieval_execution():
    return RetrievalExecution(
        final_documents=tuple(_final_documents()),
        trace=EvaluationTrace(
            trace_id="trace-1",
            original_query="保修期多久？",
            rewritten_query="产品保修期多久？",
            metadata={},
            stages={},
        ),
    )


def _patch_rewrite(monkeypatch, chain):
    monkeypatch.setattr(
        chain,
        "rewrite_query",
        lambda **kwargs: {
            "rewritten_query": "产品保修期多久？",
            "time_intent": {"type": "none"},
        },
    )


def test_run_rag_with_trace_retrieves_once_and_generates_from_final_context(
    monkeypatch,
):
    import rag.chain as chain

    _patch_rewrite(monkeypatch, chain)
    calls = {"retrieve": 0, "generate": 0}
    captured = {}

    def fake_retrieve(*args, **kwargs):
        calls["retrieve"] += 1
        return _retrieval_execution()

    def fake_generate(*, question, documents, chat_history):
        calls["generate"] += 1
        captured["documents"] = list(documents)
        return "保修期为 12 个月。[S1]"

    monkeypatch.setattr(chain, "retrieve_with_trace", fake_retrieve)
    monkeypatch.setattr(chain, "generate_answer_from_documents", fake_generate)

    execution = chain.run_rag_with_trace("保修期多久？")

    assert calls == {"retrieve": 1, "generate": 1}
    assert captured["documents"] == _final_documents()
    assert execution.source_documents == tuple(_final_documents())
    assert execution.answer.endswith("[S1]")


def test_invalid_citation_triggers_one_constrained_retry(monkeypatch):
    import rag.chain as chain

    _patch_rewrite(monkeypatch, chain)
    monkeypatch.setattr(
        chain,
        "retrieve_with_trace",
        lambda *args, **kwargs: _retrieval_execution(),
    )
    answers = iter(
        [
            "保修期为 12 个月。",
            "保修期为 12 个月。[S1]",
        ]
    )
    calls = []

    def fake_generate(*, question, documents, chat_history):
        calls.append(question)
        return next(answers)

    monkeypatch.setattr(chain, "generate_answer_from_documents", fake_generate)

    execution = chain.run_rag_with_trace("保修期多久？")

    assert execution.answer.endswith("[S1]")
    assert len(calls) == 2
    assert "只能使用" in calls[1]


def test_second_invalid_citation_returns_safe_answer(monkeypatch):
    import rag.chain as chain

    _patch_rewrite(monkeypatch, chain)
    monkeypatch.setattr(
        chain,
        "retrieve_with_trace",
        lambda *args, **kwargs: _retrieval_execution(),
    )
    monkeypatch.setattr(
        chain,
        "generate_answer_from_documents",
        lambda **kwargs: "引用了不存在的来源。[S9]",
    )

    execution = chain.run_rag_with_trace("保修期多久？")

    assert execution.answer == chain.INVALID_CITATION_SAFE_ANSWER


def test_empty_final_context_returns_evidence_insufficient_without_generation(
    monkeypatch,
):
    import rag.chain as chain

    _patch_rewrite(monkeypatch, chain)
    monkeypatch.setattr(
        chain,
        "retrieve_with_trace",
        lambda *args, **kwargs: RetrievalExecution(
            final_documents=(),
            trace=_retrieval_execution().trace,
        ),
    )

    def fail_generation(**kwargs):
        raise AssertionError("empty context must not call the LLM")

    monkeypatch.setattr(
        chain,
        "generate_answer_from_documents",
        fail_generation,
    )

    execution = chain.run_rag_with_trace("不存在的问题")

    assert execution.answer == chain.EVIDENCE_INSUFFICIENT_ANSWER
