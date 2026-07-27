from contextlib import contextmanager

import pytest
from langchain_core.documents import Document

from rag.retrieval_trace import REQUIRED_EVAL_STAGES
from rag.retriever import (
    RetrievalPipelineComponents,
    build_retrieval_metadata_filter,
    retrieve_with_trace,
)


def _child(child_id: str, **scores: float) -> Document:
    return Document(
        page_content=f"child {child_id}",
        metadata={
            "doc_id": "doc-1",
            "parent_id": f"parent-{child_id}",
            "child_id": child_id,
            "source": "manual.pdf",
            "page": 3,
            "page_range": "3",
            **scores,
        },
    )


def _parent(parent_id: str, **scores: float) -> Document:
    return Document(
        page_content=f"parent {parent_id}",
        metadata={
            "doc_id": "doc-1",
            "parent_id": parent_id,
            "source": "manual.pdf",
            "page": 3,
            "page_range": "3",
            **scores,
        },
    )


def _complete_fake_components() -> RetrievalPipelineComponents:
    bm25_docs = [_child("bm25")]
    dense_docs = [_child("dense")]
    rrf_docs = [_child("rrf")]
    reranked_docs = [_child("reranked")]
    business_docs = [_child("business")]
    diversified_docs = [_parent("diverse")]
    final_docs = [_parent("final")]
    return RetrievalPipelineComponents(
        bm25=lambda query, context: bm25_docs,
        dense=lambda query, context: dense_docs,
        rrf=lambda bm25, dense: rrf_docs,
        cross_encoder=lambda query, docs: reranked_docs,
        business_fusion=lambda query, docs, context: business_docs,
        diversify_parents=lambda docs: diversified_docs,
        assemble_context=lambda docs: final_docs,
    )


class _RecordedSpan:
    def __init__(self, name, inputs):
        self.name = name
        self.inputs = inputs
        self.outputs = None


def _record_stage_spans(monkeypatch):
    from rag import retriever

    spans = []

    @contextmanager
    def fake_trace_span(name, *, inputs=None, **kwargs):
        span = _RecordedSpan(name, inputs)
        spans.append(span)
        yield span

    def fake_end_trace_span(span, *, outputs):
        span.outputs = outputs

    monkeypatch.setattr(retriever, "trace_span", fake_trace_span)
    monkeypatch.setattr(retriever, "end_trace_span", fake_end_trace_span)
    return spans


def test_retrieval_records_all_required_langsmith_stage_spans(monkeypatch):
    """Dropping a stage wrapper would leave the production trace unauditable."""
    spans = _record_stage_spans(monkeypatch)
    outputs = {
        "bm25": [_child("bm25", bm25_score=12.5)],
        "dense": [_child("dense", dense_score=0.82)],
        "rrf": [_child("rrf", rrf_score=0.031)],
        "cross_encoder": [_child("reranked", rerank_score=0.91)],
        "business_fusion": [_child("business", business_score=1.04)],
        "diversify_parents": [_parent("diverse", mmr_score=0.77)],
        "assemble_context": [_parent("final", context_score=0.77)],
    }
    components = RetrievalPipelineComponents(
        bm25=lambda query, context: outputs["bm25"],
        dense=lambda query, context: outputs["dense"],
        rrf=lambda bm25, dense: outputs["rrf"],
        cross_encoder=lambda query, docs: outputs["cross_encoder"],
        business_fusion=lambda query, docs, context: outputs["business_fusion"],
        diversify_parents=lambda docs: outputs["diversify_parents"],
        assemble_context=lambda docs: outputs["assemble_context"],
    )

    execution = retrieve_with_trace(
        "保修期",
        retrieval_context={"tenant_id": "tenant-1"},
        components=components,
    )

    assert tuple(span.name for span in spans) == REQUIRED_EVAL_STAGES
    assert execution.trace.require_complete() is None
    expected_filter = {"$and": [{"tenant_id": "tenant-1"}]}
    expected_stage_contract = (
        ("bm25_child", outputs["bm25"], 50, "bm25_score"),
        ("dense_child", outputs["dense"], 50, "dense_score"),
        ("rrf_child", outputs["rrf"], 80, "rrf_score"),
        (
            "cross_encoder_child",
            outputs["cross_encoder"],
            15,
            "rerank_score",
        ),
        (
            "business_fused_child",
            outputs["business_fusion"],
            15,
            "business_score",
        ),
        (
            "diversified_parent",
            outputs["diversify_parents"],
            8,
            "mmr_score",
        ),
        (
            "final_context_parent",
            outputs["assemble_context"],
            6,
            "context_score",
        ),
    )
    for span, (name, documents, configured_k, score_type) in zip(
        spans, expected_stage_contract
    ):
        assert span.name == name
        assert span.inputs["query"] == "保修期"
        assert span.inputs["metadata_filter"] == expected_filter
        assert "documents" in span.inputs
        assert span.outputs["documents"] == [
            {
                "page_content": document.page_content,
                "metadata": dict(document.metadata),
            }
            for document in documents
        ]
        assert span.outputs["configured_k"] == configured_k
        assert span.outputs["score_type"] == score_type
        assert isinstance(span.outputs["latency_ms"], float)
        assert span.outputs["latency_ms"] >= 0

    assert spans[0].inputs["documents"] == []
    assert spans[1].inputs["documents"] == []
    assert spans[2].inputs["documents"] == {
        "bm25_child": spans[0].outputs["documents"],
        "dense_child": spans[1].outputs["documents"],
    }
    for previous, current in zip(spans[2:-1], spans[3:]):
        assert current.inputs["documents"] == previous.outputs["documents"]


def test_tracing_executes_each_pipeline_callable_exactly_once(monkeypatch):
    """Replaying a callable to populate a span would change ranking and latency."""
    _record_stage_spans(monkeypatch)
    calls = {
        "bm25": 0,
        "dense": 0,
        "rrf": 0,
        "cross_encoder": 0,
        "business_fusion": 0,
        "diversify_parents": 0,
        "assemble_context": 0,
    }

    def once(name, result):
        def call(*args):
            calls[name] += 1
            return result

        return call

    components = RetrievalPipelineComponents(
        bm25=once("bm25", [_child("bm25")]),
        dense=once("dense", [_child("dense")]),
        rrf=once("rrf", [_child("rrf")]),
        cross_encoder=once("cross_encoder", [_child("reranked")]),
        business_fusion=once("business_fusion", [_child("business")]),
        diversify_parents=once(
            "diversify_parents", [_parent("diversified")]
        ),
        assemble_context=once("assemble_context", [_parent("final")]),
    )

    retrieve_with_trace(
        "保修期",
        retrieval_context={},
        components=components,
    )

    assert calls == {name: 1 for name in calls}


def test_stage_latency_measures_business_call_not_trace_setup(monkeypatch):
    """Including trace transport setup would corrupt evaluation-stage latency."""
    from rag import retriever

    clock = iter((10.0, 20.0, 21.0))
    recorded_outputs = []

    @contextmanager
    def trace_with_slow_setup(name, *, inputs):
        retriever.perf_counter()
        yield object()

    monkeypatch.setattr(retriever, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(retriever, "trace_span", trace_with_slow_setup)
    monkeypatch.setattr(
        retriever,
        "end_trace_span",
        lambda span, *, outputs: recorded_outputs.append(outputs),
    )

    result, latency = retriever._run_traced_stage(
        name="bm25_child",
        function=lambda: [_child("bm25")],
        args=(),
        inputs={},
        configured_k=50,
        score_type="bm25_score",
    )

    assert result == [_child("bm25")]
    assert latency == 1000.0


@pytest.mark.parametrize("failing_branch", ["bm25", "dense"])
def test_failed_parallel_branch_records_error_without_replaying_pipeline(
    monkeypatch, failing_branch
):
    """A degraded recall span must describe the one failure and preserve the other path."""
    spans = _record_stage_spans(monkeypatch)
    calls = {
        "bm25": 0,
        "dense": 0,
        "rrf": 0,
        "cross_encoder": 0,
        "business_fusion": 0,
        "diversify_parents": 0,
        "assemble_context": 0,
    }

    def recall(name):
        calls[name] += 1
        if name == failing_branch:
            raise RuntimeError(f"{name} unavailable")
        return [_child(name)]

    def next_stage(name, result):
        def call(*args):
            calls[name] += 1
            return result

        return call

    components = RetrievalPipelineComponents(
        bm25=lambda query, context: recall("bm25"),
        dense=lambda query, context: recall("dense"),
        rrf=next_stage("rrf", [_child("rrf")]),
        cross_encoder=next_stage(
            "cross_encoder", [_child("reranked")]
        ),
        business_fusion=next_stage(
            "business_fusion", [_child("business")]
        ),
        diversify_parents=next_stage(
            "diversify_parents", [_parent("diversified")]
        ),
        assemble_context=next_stage(
            "assemble_context", [_parent("final")]
        ),
    )

    execution = retrieve_with_trace(
        "保修期",
        retrieval_context={},
        components=components,
    )

    failed_span = next(
        span for span in spans if span.name == f"{failing_branch}_child"
    )
    assert failed_span.outputs["documents"] == []
    assert failed_span.outputs["error"] == f"{failing_branch} unavailable"
    assert failed_span.outputs["latency_ms"] >= 0
    assert calls == {name: 1 for name in calls}
    assert execution.trace.metadata["degraded_stages"] == [
        f"{failing_branch}_child"
    ]
    assert tuple(execution.trace.stages) == REQUIRED_EVAL_STAGES


def test_retrieve_with_trace_records_seven_real_stage_outputs():
    execution = retrieve_with_trace(
        "保修期",
        retrieval_context={"visibility": "public"},
        components=_complete_fake_components(),
    )

    assert tuple(execution.trace.stages) == REQUIRED_EVAL_STAGES
    assert execution.trace.stages["bm25_child"].configured_k == 50
    assert execution.trace.stages["rrf_child"].configured_k == 80
    assert execution.trace.stages["final_context_parent"].configured_k == 6
    assert [
        doc.metadata["parent_id"]
        for doc in execution.final_documents
    ] == ["final"]
    assert all(
        candidate.level == "child"
        for candidate in execution.trace.stages[
            "cross_encoder_child"
        ].candidates
    )
    assert all(
        candidate.level == "parent"
        for candidate in execution.trace.stages[
            "diversified_parent"
        ].candidates
    )


def test_official_trace_rejects_missing_business_stage():
    execution = retrieve_with_trace(
        "保修期",
        retrieval_context={"visibility": "public"},
        components=_complete_fake_components(),
    )
    del execution.trace.stages["business_fused_child"]

    with pytest.raises(ValueError, match="business_fused_child"):
        execution.trace.require_complete()


def test_empty_pipeline_records_empty_stages_without_padding():
    empty = RetrievalPipelineComponents(
        bm25=lambda query, context: [],
        dense=lambda query, context: [],
        rrf=lambda bm25, dense: [],
        cross_encoder=lambda query, docs: [],
        business_fusion=lambda query, docs, context: [],
        diversify_parents=lambda docs: [],
        assemble_context=lambda docs: [],
    )

    execution = retrieve_with_trace(
        "不存在的问题",
        retrieval_context={},
        components=empty,
    )

    assert tuple(execution.trace.stages) == REQUIRED_EVAL_STAGES
    assert all(
        stage.candidates == ()
        for stage in execution.trace.stages.values()
    )
    assert execution.final_documents == ()


def test_auth_and_time_filters_are_combined_for_both_routes():
    metadata_filter = build_retrieval_metadata_filter(
        {
            "auth_context": {
                "tenant_id": "tenant-1",
                "principals": ["user-1", "role-reader"],
                "visibility": "internal",
            },
            "time_intent": {
                "type": "range",
                "field": "doc_date",
                "range": {"gte": 20240101, "lte": 20241231},
            },
        }
    )

    assert metadata_filter == {
        "$and": [
            {"tenant_id": "tenant-1"},
            {"acl_principals": {"$in": ["user-1", "role-reader"]}},
            {"visibility": "internal"},
            {"has_doc_date": True},
            {"doc_date_min": {"$lte": 20241231}},
            {"doc_date_max": {"$gte": 20240101}},
        ]
    }


def test_get_hybrid_retriever_checks_count_without_scanning_children(monkeypatch):
    from rag import retriever

    class FakeStore:
        def count_children(self, *, status):
            assert status == "active"
            return 1

        def get_all_child_documents(self):
            raise AssertionError("must never scan every Child")

    monkeypatch.setattr(retriever, "get_vectorstore", lambda: FakeStore())

    result = retriever.get_hybrid_retriever(
        time_intent={"type": "none"},
    )

    assert result is not None
    assert result.time_intent == {"type": "none"}
