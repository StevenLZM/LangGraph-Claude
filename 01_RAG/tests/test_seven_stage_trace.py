import pytest
from langchain_core.documents import Document

from rag.retrieval_trace import REQUIRED_EVAL_STAGES
from rag.retriever import (
    RetrievalPipelineComponents,
    build_retrieval_metadata_filter,
    retrieve_with_trace,
)


def _child(child_id: str) -> Document:
    return Document(
        page_content=f"child {child_id}",
        metadata={
            "doc_id": "doc-1",
            "parent_id": f"parent-{child_id}",
            "child_id": child_id,
            "source": "manual.pdf",
            "page_range": "3",
        },
    )


def _parent(parent_id: str) -> Document:
    return Document(
        page_content=f"parent {parent_id}",
        metadata={
            "doc_id": "doc-1",
            "parent_id": parent_id,
            "source": "manual.pdf",
            "page_range": "3",
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
