import pytest
from langchain_core.documents import Document

from rag.retrieval_trace import (
    REQUIRED_EVAL_STAGES,
    EvaluationTrace,
    StageRecorder,
)


def test_stage_recorder_preserves_rank_and_identity():
    recorder = StageRecorder(trace_id="trace-1", original_query="保修期")
    docs = [
        Document(
            page_content="保修期为 12 个月",
            metadata={
                "doc_id": "doc-1",
                "parent_id": "parent-1",
                "child_id": "child-1",
                "source": "manual.pdf",
                "page_range": "3",
                "section_path": "保修政策",
                "rrf_score": 0.02,
            },
        )
    ]

    stage = recorder.record(
        name="rrf_child",
        documents=docs,
        configured_k=80,
        score_type="rrf_score",
        latency_ms=12.5,
    )

    assert stage.candidates[0].rank == 1
    assert stage.candidates[0].child_id == "child-1"
    assert stage.candidates[0].score == 0.02
    assert recorder.trace.stages["rrf_child"] == stage


def test_trace_rejects_duplicate_stage_name():
    recorder = StageRecorder(trace_id="trace-1", original_query="q")
    recorder.record(
        name="bm25_child",
        documents=[],
        configured_k=50,
        score_type="bm25_score",
        latency_ms=1.0,
    )

    with pytest.raises(ValueError, match="重复阶段"):
        recorder.record(
            name="bm25_child",
            documents=[],
            configured_k=50,
            score_type="bm25_score",
            latency_ms=1.0,
        )


def test_trace_lists_missing_required_stages():
    trace = EvaluationTrace(
        trace_id="trace-1",
        original_query="q",
        rewritten_query="q",
        metadata={},
        stages={},
    )

    assert trace.missing_required_stages() == list(REQUIRED_EVAL_STAGES)


def test_child_stage_requires_child_id():
    recorder = StageRecorder(trace_id="trace-1", original_query="q")
    docs = [
        Document(
            page_content="content",
            metadata={
                "doc_id": "doc-1",
                "parent_id": "parent-1",
                "source": "manual.pdf",
                "page_range": "3",
            },
        )
    ]

    with pytest.raises(ValueError, match="child_id"):
        recorder.record(
            name="dense_child",
            documents=docs,
            configured_k=50,
            score_type="dense_score",
            latency_ms=1.0,
        )


def test_parent_stage_requires_parent_id():
    recorder = StageRecorder(trace_id="trace-1", original_query="q")
    docs = [
        Document(
            page_content="content",
            metadata={
                "doc_id": "doc-1",
                "source": "manual.pdf",
                "page_range": "3",
            },
        )
    ]

    with pytest.raises(ValueError, match="parent_id"):
        recorder.record(
            name="final_context_parent",
            documents=docs,
            configured_k=6,
            score_type="final_order",
            latency_ms=1.0,
        )


def test_trace_require_complete_names_missing_stage():
    trace = EvaluationTrace(
        trace_id="trace-1",
        original_query="q",
        rewritten_query="q",
        metadata={},
        stages={},
    )

    with pytest.raises(ValueError, match="bm25_child"):
        trace.require_complete()
