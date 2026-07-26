import pytest
from langchain_core.documents import Document


class FakeParentStore:
    def __init__(self, parents):
        self.parents = {
            parent.metadata["parent_id"]: parent
            for parent in parents
        }

    def get_parents(self, parent_ids):
        return {
            parent_id: self.parents[parent_id]
            for parent_id in parent_ids
            if parent_id in self.parents
        }


def _parent(parent_id, doc_id, content):
    return Document(
        page_content=content,
        metadata={
            "parent_id": parent_id,
            "doc_id": doc_id,
            "source": f"{doc_id}.pdf",
            "page_range": "1",
        },
    )


def _child(
    child_id,
    parent_id,
    doc_id,
    score,
    vector,
):
    return Document(
        page_content=child_id,
        metadata={
            "child_id": child_id,
            "parent_id": parent_id,
            "doc_id": doc_id,
            "business_score": score,
            "retrieval_embedding": vector,
        },
    )


def test_parent_score_uses_best_child_plus_capped_evidence_bonus():
    from rag.postprocessor import diversify_parent_candidates

    parents = [_parent("p1", "doc-1", "parent content")]
    children = [
        _child("c1", "p1", "doc-1", 0.8, [1.0, 0.0]),
        _child("c2", "p1", "doc-1", 0.7, [1.0, 0.0]),
    ]

    result = diversify_parent_candidates(
        children,
        parent_docstore=FakeParentStore(parents),
        top_k=8,
    )

    assert result[0].metadata["parent_score"] == pytest.approx(0.82)
    assert result[0].metadata["matched_child_ids"] == ["c1", "c2"]
    assert result[0].metadata["representative_child_id"] == "c1"


def test_exact_duplicate_parent_keeps_higher_scored_parent():
    from rag.postprocessor import diversify_parent_candidates

    parents = [
        _parent("p1", "doc-1", "完全相同的父块"),
        _parent("p2", "doc-2", "完全相同的父块"),
    ]
    children = [
        _child("c1", "p1", "doc-1", 0.9, [1.0, 0.0]),
        _child("c2", "p2", "doc-2", 0.8, [0.0, 1.0]),
    ]

    result = diversify_parent_candidates(
        children,
        parent_docstore=FakeParentStore(parents),
    )

    assert [doc.metadata["parent_id"] for doc in result] == ["p1"]


def test_near_duplicate_parent_uses_representative_child_vectors():
    from rag.postprocessor import diversify_parent_candidates

    parents = [
        _parent("p1", "doc-1", "版本一内容"),
        _parent("p2", "doc-2", "版本二内容"),
    ]
    children = [
        _child("c1", "p1", "doc-1", 0.9, [1.0, 0.0]),
        _child("c2", "p2", "doc-2", 0.8, [0.99, 0.01]),
    ]

    result = diversify_parent_candidates(
        children,
        parent_docstore=FakeParentStore(parents),
        similarity_threshold=0.92,
    )

    assert [doc.metadata["parent_id"] for doc in result] == ["p1"]


def test_document_quota_keeps_at_most_two_parents_per_doc():
    from rag.postprocessor import diversify_parent_candidates

    parents = [
        _parent(f"p{index}", "doc-1", f"content {index}")
        for index in range(3)
    ]
    children = [
        _child(
            f"c{index}",
            f"p{index}",
            "doc-1",
            1.0 - index * 0.1,
            [float(index == vector_index) for vector_index in range(3)],
        )
        for index in range(3)
    ]

    result = diversify_parent_candidates(
        children,
        parent_docstore=FakeParentStore(parents),
        max_parents_per_document=2,
        similarity_threshold=0.99,
    )

    assert len(result) == 2
    assert {doc.metadata["doc_id"] for doc in result} == {"doc-1"}


def test_mmr_prefers_a_diverse_source_over_near_duplicate_relevance():
    from rag.postprocessor import diversify_parent_candidates

    parents = [
        _parent("p-a", "doc-a", "alpha"),
        _parent("p-b", "doc-b", "beta"),
        _parent("p-c", "doc-c", "gamma"),
    ]
    children = [
        _child("c-a", "p-a", "doc-a", 1.0, [1.0, 0.0]),
        _child("c-b", "p-b", "doc-b", 0.95, [0.9, 0.43589]),
        _child("c-c", "p-c", "doc-c", 0.8, [0.0, 1.0]),
    ]

    result = diversify_parent_candidates(
        children,
        parent_docstore=FakeParentStore(parents),
        top_k=2,
        similarity_threshold=0.92,
        mmr_lambda=0.70,
    )

    assert [doc.metadata["parent_id"] for doc in result] == ["p-a", "p-c"]
