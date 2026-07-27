from contextlib import contextmanager

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

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
            metadata={
                "metadata_filter": {"$and": [{"tenant_id": "tenant-1"}]},
            },
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


class _RecordedSpan:
    def __init__(self, name, inputs, tags, metadata):
        self.name = name
        self.inputs = inputs
        self.tags = tags
        self.metadata = metadata
        self.outputs = None

    def end(self, *, outputs):
        self.outputs = outputs


def _record_trace_spans(monkeypatch, chain):
    spans = []

    @contextmanager
    def fake_rag_tracing_context(**kwargs):
        yield None

    @contextmanager
    def fake_trace_span(name, *, inputs=None, tags=(), metadata=None, **kwargs):
        span = _RecordedSpan(name, inputs, tags, metadata)
        spans.append(span)
        yield span

    monkeypatch.setattr(chain, "trace_span", fake_trace_span)
    monkeypatch.setattr(chain, "rag_tracing_context", fake_rag_tracing_context)
    return spans


def _generation_attempt_spans(spans):
    return [span for span in spans if span.name == "generation.attempt"]


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


def test_rag_execution_records_root_rewrite_retrieval_and_generation_spans(
    monkeypatch,
):
    """Removing a request boundary would hide the audited one-pass execution."""
    import rag.chain as chain

    _patch_rewrite(monkeypatch, chain)
    spans = _record_trace_spans(monkeypatch, chain)
    retrieval = _retrieval_execution()
    monkeypatch.setattr(chain, "retrieve_with_trace", lambda *args, **kwargs: retrieval)
    monkeypatch.setattr(
        chain,
        "generate_answer_from_documents",
        lambda **kwargs: "保修期为 12 个月。[S1]",
    )
    history = [HumanMessage(content="上一轮问题")]

    execution = chain.run_rag_with_trace(
        "保修期多久？",
        chat_history=history,
        auth_context={"tenant_id": "tenant-1"},
        trace_metadata={"session_id": "session-1"},
        trace_tags=("01-rag",),
    )

    assert [span.name for span in spans] == [
        "rag.request",
        "query.rewrite",
        "retrieval",
        "generation",
        "generation.attempt",
    ]
    assert spans[0].inputs == {
        "question": "保修期多久？",
        "chat_history": history,
        "auth_context": {"tenant_id": "tenant-1"},
    }
    assert spans[0].tags == ("01-rag",)
    assert spans[0].metadata == {"session_id": "session-1"}
    assert spans[1].outputs["rewritten_query"] == "产品保修期多久？"
    assert spans[1].outputs["time_intent"] == {"type": "none"}
    assert spans[2].inputs == {
        "rewritten_query": "产品保修期多久？",
        "time_intent": {"type": "none"},
        "auth_context": {"tenant_id": "tenant-1"},
    }
    assert spans[2].outputs["metadata_filter"] == retrieval.trace.metadata[
        "metadata_filter"
    ]
    assert spans[2].outputs["documents"] == (
        {
            "page_content": "产品保修期为 12 个月。",
            "metadata": _final_documents()[0].metadata,
        },
    )
    assert spans[2].outputs["evaluation_trace"]["trace_id"] == "trace-1"
    assert spans[3].inputs == {
        "question": "保修期多久？",
        "chat_history": history,
        "documents": retrieval.final_documents,
        "evidence_context": chain.format_docs_for_context(retrieval.final_documents),
        "system_prompt": chain.SYSTEM_PROMPT,
    }
    assert spans[3].outputs["answer"] == execution.answer
    assert _generation_attempt_spans(spans)[0].inputs == {
        "question": "保修期多久？",
        "chat_history": history,
        "documents": retrieval.final_documents,
        "evidence_context": chain.format_docs_for_context(retrieval.final_documents),
        "system_prompt": chain.SYSTEM_PROMPT,
    }
    assert spans[0].outputs["answer"] == execution.answer
    assert spans[0].outputs["documents"] == spans[2].outputs["documents"]
    assert spans[0].outputs["evaluation_trace"]["trace_id"] == "trace-1"
    assert spans[0].outputs["generation_latency_ms"] == execution.generation_latency_ms


def test_empty_final_context_records_skipped_generation_span_without_calling_llm(
    monkeypatch,
):
    """Treating an evidence miss as generation would conceal that no LLM ran."""
    import rag.chain as chain

    _patch_rewrite(monkeypatch, chain)
    spans = _record_trace_spans(monkeypatch, chain)
    monkeypatch.setattr(
        chain,
        "retrieve_with_trace",
        lambda *args, **kwargs: RetrievalExecution(
            final_documents=(), trace=_retrieval_execution().trace
        ),
    )

    def fail_generation(**kwargs):
        raise AssertionError("empty context must not call the LLM")

    monkeypatch.setattr(chain, "generate_answer_from_documents", fail_generation)

    execution = chain.run_rag_with_trace("不存在的问题")

    assert execution.answer == chain.EVIDENCE_INSUFFICIENT_ANSWER
    assert [span.name for span in spans] == [
        "rag.request",
        "query.rewrite",
        "retrieval",
        "generation",
    ]
    assert spans[-1].outputs == {"skipped": True, "reason": "no_evidence"}


def test_generation_span_records_two_attempts_without_extra_retry(monkeypatch):
    """Wrapping retries must not introduce an additional answer-model invocation."""
    import rag.chain as chain

    _patch_rewrite(monkeypatch, chain)
    spans = _record_trace_spans(monkeypatch, chain)
    monkeypatch.setattr(
        chain,
        "retrieve_with_trace",
        lambda *args, **kwargs: _retrieval_execution(),
    )
    answers = iter(["保修期为 12 个月。", "保修期为 12 个月。[S1]"])
    calls = []

    def fake_generate(*, question, documents, chat_history):
        calls.append(question)
        return next(answers)

    monkeypatch.setattr(chain, "generate_answer_from_documents", fake_generate)

    execution = chain.run_rag_with_trace("保修期多久？")

    assert len(calls) == 2
    generation_span = next(span for span in spans if span.name == "generation")
    assert generation_span.outputs == {"answer": execution.answer, "attempts": 2}
    assert [span.inputs for span in _generation_attempt_spans(spans)] == [
        {
            "question": "保修期多久？",
            "chat_history": (),
            "documents": _retrieval_execution().final_documents,
            "evidence_context": chain.format_docs_for_context(
                _retrieval_execution().final_documents
            ),
            "system_prompt": chain.SYSTEM_PROMPT,
        },
        {
            "question": (
                "保修期多久？\n\n上一次回答的证据引用无效。"
                "只能使用这些引用：[S1]；每个事实结论必须引用。"
            ),
            "chat_history": (),
            "documents": _retrieval_execution().final_documents,
            "evidence_context": chain.format_docs_for_context(
                _retrieval_execution().final_documents
            ),
            "system_prompt": chain.SYSTEM_PROMPT,
        },
    ]
