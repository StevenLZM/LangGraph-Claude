from pathlib import Path
import sys

from langchain_core.documents import Document


sys.path.insert(0, str(Path(__file__).parent.parent))


def test_chunk_documents_returns_hierarchical_result():
    from rag.chunker import ChunkingResult, chunk_documents

    docs = [
        Document(
            page_content=(
                "第一章 产品概述\n\n"
                "RAG 系统用于检索增强生成。它支持知识库问答和多轮对话。\n\n"
                "第二章 技术规格\n\n"
                "系统要求 Python 3.9+，内存 8GB+，磁盘空间 10GB+。"
            ),
            metadata={
                "source": "spec.pdf",
                "file_path": "/tmp/spec.pdf",
                "page": 1,
                "total_pages": 1,
                "doc_id": "doc-001",
            },
        )
    ]

    result = chunk_documents(docs)

    assert isinstance(result, ChunkingResult)
    assert result.parents, "应生成 parent chunks"
    assert result.children, "应生成 child chunks"
    assert result.stats["total_parents"] == len(result.parents)
    assert result.stats["total_children"] == len(result.children)
    assert all("parent_id" in child.metadata for child in result.children)
    assert all("doc_version" in parent.metadata for parent in result.parents)


def test_list_documents_includes_parent_and_child_counts():
    from rag.vectorstore import list_documents

    class MockVectorStore:
        def get(self, include=None):
            return {
                "ids": ["c-1", "c-2"],
                "metadatas": [
                    {
                        "doc_id": "doc-001",
                        "source": "spec.pdf",
                        "total_pages": 3,
                        "child_id": "c-1",
                        "doc_version": "v1",
                    },
                    {
                        "doc_id": "doc-001",
                        "source": "spec.pdf",
                        "total_pages": 3,
                        "child_id": "c-2",
                        "doc_version": "v1",
                    },
                ],
            }

    class MockParentDocstore:
        def list_documents(self):
            return [
                {
                    "doc_id": "doc-001",
                    "parent_count": 1,
                    "doc_version": "v1",
                }
            ]

    docs = list_documents(MockVectorStore(), MockParentDocstore())

    assert docs == [
        {
            "doc_id": "doc-001",
            "source": "spec.pdf",
            "total_pages": 3,
            "total_chunks": 2,
            "child_count": 2,
            "parent_count": 1,
            "doc_version": "v1",
            "pages": [],
        }
    ]


def test_parent_child_hybrid_retriever_hydrates_and_deduplicates():
    from rag.retriever import ParentChildHybridRetriever

    class FakeEnsemble:
        def invoke(self, query):
            assert query == "保修期"
            return [
                Document(
                    page_content="child 1",
                    metadata={
                        "doc_id": "doc-001",
                        "parent_id": "p-1",
                        "child_id": "c-1",
                        "page_range": "1",
                        "section_path": "第一章 产品概述",
                    },
                ),
                Document(
                    page_content="child 2",
                    metadata={
                        "doc_id": "doc-001",
                        "parent_id": "p-1",
                        "child_id": "c-2",
                        "page_range": "1-2",
                        "section_path": "第一章 产品概述",
                    },
                ),
                Document(
                    page_content="child 3",
                    metadata={
                        "doc_id": "doc-001",
                        "parent_id": "p-2",
                        "child_id": "c-3",
                        "page_range": "3",
                        "section_path": "第二章 技术规格",
                    },
                ),
            ]

    class FakeParentDocstore:
        def get_parents(self, parent_ids):
            return {
                "p-1": Document(
                    page_content="父块一：产品保修期为 12 个月。",
                    metadata={
                        "parent_id": "p-1",
                        "doc_id": "doc-001",
                        "source": "spec.pdf",
                        "section_path": "第一章 产品概述",
                        "page_range": "1-2",
                    },
                ),
                "p-2": Document(
                    page_content="父块二：系统要求 Python 3.9+。",
                    metadata={
                        "parent_id": "p-2",
                        "doc_id": "doc-001",
                        "source": "spec.pdf",
                        "section_path": "第二章 技术规格",
                        "page_range": "3",
                    },
                ),
            }

    retriever = ParentChildHybridRetriever(FakeEnsemble(), FakeParentDocstore())
    results = retriever.invoke("保修期")

    assert len(results) == 2
    assert results[0].page_content == "父块一：产品保修期为 12 个月。"
    assert results[0].metadata["matched_child_ids"] == ["c-1", "c-2"]
    assert results[0].metadata["section_path"] == "第一章 产品概述"


def test_parent_child_hybrid_retriever_supports_simple_invoke_without_config():
    from rag.retriever import ParentChildHybridRetriever

    class FakeEnsemble:
        def invoke(self, query):
            assert query == "保修期"
            return [
                Document(
                    page_content="child",
                    metadata={
                        "doc_id": "doc-001",
                        "parent_id": "p-1",
                        "child_id": "c-1",
                        "section_path": "第一章 产品概述",
                    },
                )
            ]

    class FakeParentDocstore:
        def get_parents(self, parent_ids):
            return {
                "p-1": Document(
                    page_content="父块：产品保修期为 12 个月。",
                    metadata={
                        "parent_id": "p-1",
                        "doc_id": "doc-001",
                        "source": "spec.pdf",
                        "section_path": "第一章 产品概述",
                    },
                )
            }

    results = ParentChildHybridRetriever(FakeEnsemble(), FakeParentDocstore()).invoke("保修期")

    assert len(results) == 1
    assert results[0].metadata["parent_id"] == "p-1"


def test_format_docs_for_context_uses_page_range_for_parent_docs():
    from rag.chain import format_docs_for_context

    docs = [
        Document(
            page_content="父块一：产品保修期为 12 个月。",
            metadata={
                "source": "spec.pdf",
                "page_range": "1-2",
                "section_path": "第一章 产品概述",
                "best_child_score": 0.82,
            },
        )
    ]

    context = format_docs_for_context(docs)

    assert "第1-2页" in context
    assert "章节: 第一章 产品概述" in context
