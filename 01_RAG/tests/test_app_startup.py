from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).parent.parent))


def test_safe_collection_stats_returns_error_state_instead_of_raising(monkeypatch):
    import app

    def fail_stats():
        raise RuntimeError("Open local milvus failed")

    app.st.session_state.vectorstore = object()
    monkeypatch.setattr(app, "get_collection_stats", lambda _vs: fail_stats())

    stats = app._safe_collection_stats()

    assert stats["total_children"] == 0
    assert stats["total_parents"] == 0
    assert "Open local milvus failed" in stats["error"]


def test_refresh_indexed_docs_uses_docstore_without_opening_vectorstore(monkeypatch):
    import app

    class FakeDocStore:
        def list_documents(self):
            return [{
                "doc_id": "doc-1",
                "doc_version": "v1",
                "source": "demo.pdf",
                "parent_count": 2,
            }]

    def fail_vectorstore():
        raise AssertionError("startup should not open Milvus")

    app.st.session_state.vectorstore = None
    monkeypatch.setattr(app, "get_vectorstore", fail_vectorstore)
    monkeypatch.setattr(app, "get_parent_docstore", lambda: FakeDocStore())

    app.refresh_indexed_docs()

    assert app.st.session_state.indexed_docs == [{
        "doc_id": "doc-1",
        "doc_version": "v1",
        "source": "demo.pdf",
        "parent_count": 2,
        "total_pages": 0,
        "total_chunks": 2,
        "child_count": 0,
    }]


def test_safe_collection_stats_reports_lock_without_opening_vectorstore(monkeypatch):
    import app

    class FakeDocStore:
        def count(self):
            return 2

    def fail_stats(_vs):
        raise AssertionError("startup should not open Milvus")

    app.st.session_state.vectorstore = None
    monkeypatch.setattr(app, "get_collection_stats", fail_stats)
    monkeypatch.setattr(app, "get_parent_docstore", lambda: FakeDocStore())
    monkeypatch.setattr(app, "_detect_milvus_lock", lambda: "Milvus Lite 数据库被其他进程占用")

    stats = app._safe_collection_stats()

    assert stats["total_parents"] == 2
    assert stats["total_children"] == 0
    assert "Milvus Lite" in stats["error"]


def test_get_or_build_chain_handles_vectorstore_startup_failure(monkeypatch):
    import app

    def fail_chain():
        raise RuntimeError("Open local milvus failed")

    app.st.session_state.chain = None
    monkeypatch.setattr(app, "create_chain_with_history", fail_chain)

    chain, get_history = app.get_or_build_chain()

    assert chain is None
    assert get_history is None
