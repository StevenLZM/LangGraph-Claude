from pathlib import Path
import sys

import pytest
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


def test_parent_docstore_deletes_only_requested_document_version(tmp_path):
    from rag.docstore import ParentDocStore

    store = ParentDocStore(tmp_path / "parents.sqlite")
    store.upsert_parents(
        [
            Document(
                page_content="v1 parent",
                metadata={
                    "parent_id": "parent-v1",
                    "doc_id": "doc-1",
                    "doc_version": "v1",
                    "source": "manual.pdf",
                },
            ),
            Document(
                page_content="v2 parent",
                metadata={
                    "parent_id": "parent-v2",
                    "doc_id": "doc-1",
                    "doc_version": "v2",
                    "source": "manual.pdf",
                },
            ),
        ]
    )

    assert store.delete_document_version("doc-1", "v2") == 1
    assert set(store.get_parents(["parent-v1", "parent-v2"])) == {"parent-v1"}


def test_parent_docstore_deletes_versions_except_active(tmp_path):
    from rag.docstore import ParentDocStore

    store = ParentDocStore(tmp_path / "parents.sqlite")
    store.upsert_parents(
        [
            Document(
                page_content=version,
                metadata={
                    "parent_id": f"parent-{version}",
                    "doc_id": "doc-1",
                    "doc_version": version,
                },
            )
            for version in ("v1", "v2")
        ]
    )

    assert store.delete_versions_except("doc-1", "v2") == 1
    assert set(store.get_parents(["parent-v1", "parent-v2"])) == {"parent-v2"}


def test_add_documents_activates_new_version_before_removing_old_parents(monkeypatch):
    from rag.chunker import ChunkingResult
    from rag.vectorstore import add_documents

    calls = []

    class FakeStore:
        def bulk_stage_children(self, documents, vectors, ingest_run_id):
            calls.append("bulk staging children")
            assert len(documents) == len(vectors) == 1
            return 1

        def activate_version(self, doc_id, doc_version, ingest_run_id):
            calls.append("activate new version")

        def deactivate_other_versions(self, doc_id, active_doc_version):
            calls.append("deactivate old versions")

        def delete_ingest_run(self, ingest_run_id):
            calls.append("cleanup staging")

    class FakeDocStore:
        def upsert_parents(self, parents):
            calls.append("upsert parents")

        def delete_versions_except(self, doc_id, active_doc_version):
            calls.append("delete old parent versions")

        def delete_document_version(self, doc_id, doc_version):
            calls.append("delete new parent version")

    class FakeEmbeddings:
        pass

    monkeypatch.setattr(
        "rag.vectorstore.get_embeddings",
        lambda: FakeEmbeddings(),
    )
    monkeypatch.setattr(
        "rag.vectorstore.embed_with_retry",
        lambda embeddings, texts: [[0.1, 0.2, 0.3] for _ in texts],
    )
    chunks = ChunkingResult(
        parents=[
            Document(
                page_content="parent",
                metadata={
                    "doc_id": "doc-1",
                    "doc_version": "v2",
                    "parent_id": "parent-1",
                },
            )
        ],
        children=[
            Document(
                page_content="child",
                metadata={
                    "doc_id": "doc-1",
                    "doc_version": "v2",
                    "parent_id": "parent-1",
                    "child_id": "child-1",
                },
            )
        ],
        stats={},
    )

    assert add_documents(
        chunks,
        "doc-1",
        vectorstore=FakeStore(),
        parent_docstore=FakeDocStore(),
        ingest_run_id="run-1",
    ) == 1
    assert calls == [
        "upsert parents",
        "bulk staging children",
        "activate new version",
        "deactivate old versions",
        "delete old parent versions",
    ]


def test_add_documents_cleans_only_new_version_when_bulk_fails(monkeypatch):
    from rag.chunker import ChunkingResult
    from rag.vectorstore import add_documents

    calls = []

    class FakeStore:
        def bulk_stage_children(self, documents, vectors, ingest_run_id):
            raise RuntimeError("bulk failed")

        def activate_version(self, *args):
            calls.append("activated")

        def deactivate_other_versions(self, *args):
            calls.append("deactivated old")

        def delete_ingest_run(self, ingest_run_id):
            calls.append(("cleanup staging", ingest_run_id))

    class FakeDocStore:
        def upsert_parents(self, parents):
            calls.append("upsert parents")

        def delete_document_version(self, doc_id, doc_version):
            calls.append(("delete new parent version", doc_id, doc_version))

        def delete_versions_except(self, *args):
            calls.append("deleted old parents")

    monkeypatch.setattr(
        "rag.vectorstore.get_embeddings",
        lambda: object(),
    )
    monkeypatch.setattr(
        "rag.vectorstore.embed_with_retry",
        lambda embeddings, texts: [[0.1, 0.2, 0.3]],
    )
    chunks = ChunkingResult(
        parents=[
            Document(
                page_content="parent",
                metadata={
                    "doc_id": "doc-1",
                    "doc_version": "v2",
                    "parent_id": "parent-1",
                },
            )
        ],
        children=[
            Document(
                page_content="child",
                metadata={
                    "doc_id": "doc-1",
                    "doc_version": "v2",
                    "parent_id": "parent-1",
                    "child_id": "child-1",
                },
            )
        ],
        stats={},
    )

    with pytest.raises(RuntimeError, match="bulk failed"):
        add_documents(
            chunks,
            "doc-1",
            vectorstore=FakeStore(),
            parent_docstore=FakeDocStore(),
            ingest_run_id="run-failed",
        )

    assert calls == [
        "upsert parents",
        ("cleanup staging", "run-failed"),
        ("delete new parent version", "doc-1", "v2"),
    ]


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
