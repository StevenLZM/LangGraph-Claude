import pytest
from langchain_core.documents import Document


def _documents():
    return [
        Document(
            page_content="best",
            metadata={
                "child_id": "best",
                "rerank_score": 2.0,
                "rrf_score": 0.02,
                "authority_score": 1.0,
                "doc_date_max": 20240201,
                "version_rank": 2,
            },
        ),
        Document(
            page_content="other",
            metadata={
                "child_id": "other",
                "rerank_score": 0.0,
                "rrf_score": 0.01,
                "authority_score": 0.0,
                "doc_date_max": 20230101,
                "version_rank": 1,
            },
        ),
    ]


def test_business_fusion_uses_approved_weighted_formula_for_latest():
    from rag.business_fusion import fuse_business_features

    result = fuse_business_features(
        "最新版本是什么？",
        _documents(),
        time_intent={"type": "latest", "field": "doc_date"},
    )

    assert result[0].metadata["child_id"] == "best"
    assert result[0].metadata["business_score"] == pytest.approx(1.0)
    assert result[1].metadata["business_score"] == pytest.approx(0.0)


def test_business_fusion_disables_freshness_for_ordinary_query():
    from rag.business_fusion import fuse_business_features

    result = fuse_business_features(
        "版本是什么？",
        _documents(),
        time_intent={"type": "none"},
    )

    assert result[0].metadata["business_features"]["freshness"] == 0.0
    assert result[0].metadata["business_score"] == pytest.approx(0.95)


def test_business_fusion_clamps_authority_and_preserves_stable_ties():
    from rag.business_fusion import fuse_business_features

    documents = [
        Document(
            page_content=child_id,
            metadata={
                "child_id": child_id,
                "rerank_score": 1.0,
                "rrf_score": 1.0,
                "authority_score": authority,
            },
        )
        for child_id, authority in (("first", 2.0), ("second", 2.0))
    ]

    result = fuse_business_features(
        "q",
        documents,
        time_intent=None,
    )

    assert [doc.metadata["child_id"] for doc in result] == ["first", "second"]
    assert all(
        doc.metadata["business_features"]["authority"] == 1.0
        for doc in result
    )
