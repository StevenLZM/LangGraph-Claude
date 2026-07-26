from langchain_core.documents import Document


def _parent(index, *, parent_id=None, content=None):
    return Document(
        page_content=content or f"第 {index} 个父块内容。",
        metadata={
            "parent_id": parent_id or f"parent-{index}",
            "doc_id": f"doc-{index}",
            "source": f"doc-{index}.pdf",
            "page_range": str(index),
            "parent_score": 1.0 - index * 0.01,
        },
    )


def test_context_assembler_limits_documents_and_assigns_stable_evidence_ids():
    from rag.context_assembler import assemble_final_context

    result = assemble_final_context(
        [_parent(index) for index in range(7)],
        max_documents=6,
        token_budget=1000,
    )

    assert len(result) == 6
    assert [doc.metadata["evidence_id"] for doc in result] == [
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
    ]


def test_context_assembler_deduplicates_parent_id():
    from rag.context_assembler import assemble_final_context

    result = assemble_final_context(
        [
            _parent(1, parent_id="same"),
            _parent(2, parent_id="same"),
            _parent(3),
        ],
        max_documents=6,
        token_budget=1000,
    )

    assert [doc.metadata["parent_id"] for doc in result] == ["same", "parent-3"]


def test_context_assembler_never_exceeds_token_budget():
    from rag.context_assembler import assemble_final_context

    result = assemble_final_context(
        [_parent(index, content="保修说明。" * 20) for index in range(3)],
        max_documents=6,
        token_budget=45,
    )

    assert result
    assert sum(doc.metadata["context_token_count"] for doc in result) <= 45


def test_context_assembler_truncates_oversized_first_parent_at_boundary():
    from rag.context_assembler import assemble_final_context

    original = _parent(
        1,
        content=("第一段说明很长。" * 20) + "\n\n" + ("第二段。" * 20),
    )

    result = assemble_final_context(
        [original],
        max_documents=6,
        token_budget=40,
    )

    assert len(result) == 1
    assert len(result[0].page_content) < len(original.page_content)
    assert result[0].page_content.endswith(("。", "\n"))
    assert result[0].metadata["context_truncated"] is True
    assert result[0].metadata["parent_id"] == "parent-1"
    assert result[0].metadata["evidence_id"] == "S1"
