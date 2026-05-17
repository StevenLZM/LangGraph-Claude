# 01_RAG Milvus Lite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 01_RAG ChromaDB vector store with Milvus Lite while keeping the existing RAG API and retrieval behavior.

**Architecture:** `config.py` owns Milvus Lite settings. `rag/vectorstore.py` preserves the current public functions and adapts LangChain Milvus initialization, metadata filters, delete/list/stats, and threshold search.

**Tech Stack:** Python, LangChain, `langchain-milvus`, `pymilvus`, Milvus Lite, pytest.

---

### File Structure

- Modify `01_RAG/config.py`: replace Chroma config export with Milvus Lite config.
- Modify `01_RAG/rag/vectorstore.py`: initialize `langchain_milvus.Milvus`, translate metadata filters, and keep the existing public API.
- Modify `01_RAG/requirements.txt`: replace Chroma dependencies with Milvus dependencies.
- Modify `01_RAG/.env.example`: document `MILVUS_URI`.
- Modify `01_RAG/README.md` and `01_RAG/app.py`: update Chroma wording to Milvus Lite.
- Modify tests in `01_RAG/tests/test_rag_pipeline.py`: add config and initialization tests using mocks.

### Task 1: Milvus Config and Dependency Surface

**Files:**
- Test: `01_RAG/tests/test_rag_pipeline.py`
- Modify: `01_RAG/config.py`
- Modify: `01_RAG/requirements.txt`
- Modify: `01_RAG/.env.example`

- [ ] **Step 1: Write the failing config tests**

```python
def test_milvus_config_defaults_to_local_lite_file(self):
    from config import MilvusConfig

    assert MilvusConfig.COLLECTION_NAME == "rag_knowledge_base_v2_children"
    assert MilvusConfig.URI.endswith("data/milvus.db")


def test_milvus_config_uses_existing_vectorstore_env_for_backwards_compat(self, monkeypatch):
    import importlib
    import config as cfg

    monkeypatch.setenv("VECTORSTORE_DIR", "data/custom_vectors")
    monkeypatch.delenv("MILVUS_URI", raising=False)

    reloaded = importlib.reload(cfg)

    assert reloaded.MilvusConfig.URI.endswith("data/custom_vectors/milvus.db")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd 01_RAG && pytest tests/test_rag_pipeline.py::TestConfig::test_milvus_config_defaults_to_local_lite_file tests/test_rag_pipeline.py::TestConfig::test_milvus_config_uses_existing_vectorstore_env_for_backwards_compat -v`

Expected: FAIL because `MilvusConfig` does not exist.

- [ ] **Step 3: Implement minimal config**

Add `MilvusConfig`, export `milvus_config`, keep `PathConfig.VECTORSTORE_DIR` for backward-compatible local path selection, and update dependencies/env docs.

- [ ] **Step 4: Run test to verify it passes**

Run the same pytest command and expect PASS.

### Task 2: Vectorstore Initialization and Filter Translation

**Files:**
- Test: `01_RAG/tests/test_rag_pipeline.py`
- Modify: `01_RAG/rag/vectorstore.py`

- [ ] **Step 1: Write failing tests**

```python
def test_get_vectorstore_initializes_milvus_with_lite_uri(self, monkeypatch):
    import rag.vectorstore as vectorstore

    captured = {}

    class FakeMilvus:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(vectorstore, "Milvus", FakeMilvus)
    monkeypatch.setattr(vectorstore, "get_embeddings", lambda: "embeddings")
    vectorstore._vectorstore_instance = None

    vectorstore.get_vectorstore(reset=True)

    assert captured["embedding_function"] == "embeddings"
    assert captured["collection_name"] == vectorstore.milvus_config.COLLECTION_NAME
    assert captured["connection_args"] == {"uri": vectorstore.milvus_config.URI}


def test_milvus_filter_translation_supports_date_and_and_filters(self):
    from rag.vectorstore import _to_milvus_filter

    assert _to_milvus_filter({"doc_id": "abc"}) == 'doc_id == "abc"'
    assert _to_milvus_filter({"upload_date": {"$gte": 20240101, "$lte": 20241231}}) == (
        "upload_date >= 20240101 and upload_date <= 20241231"
    )
    assert _to_milvus_filter({
        "$and": [
            {"has_doc_date": True},
            {"doc_date_min": {"$lte": 20241231}},
            {"doc_date_max": {"$gte": 20240101}},
        ]
    }) == "has_doc_date == true and doc_date_min <= 20241231 and doc_date_max >= 20240101"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd 01_RAG && pytest tests/test_rag_pipeline.py::TestVectorStore::test_get_vectorstore_initializes_milvus_with_lite_uri tests/test_rag_pipeline.py::TestVectorStore::test_milvus_filter_translation_supports_date_and_and_filters -v`

Expected: FAIL because `Milvus` and `_to_milvus_filter` are not implemented.

- [ ] **Step 3: Implement vectorstore Milvus adapter**

Replace the Chroma import and initialization with LangChain Milvus. Convert metadata dictionaries to Milvus boolean expressions before calling `get`, `as_retriever`, and threshold search.

- [ ] **Step 4: Run tests to verify they pass**

Run the same pytest command and expect PASS.

### Task 3: Documentation and UI Wording

**Files:**
- Modify: `01_RAG/README.md`
- Modify: `01_RAG/app.py`
- Modify: `docs/superpowers/specs/2026-05-17-01-rag-milvus-lite-design.md`

- [ ] **Step 1: Update wording**

Replace ChromaDB wording with Milvus Lite where it describes the active implementation. Keep historical design docs unchanged.

- [ ] **Step 2: Verify no active Chroma dependency remains**

Run: `rg -n "langchain-chroma|chromadb|ChromaDB|Chroma|chroma_config" 01_RAG/config.py 01_RAG/rag 01_RAG/requirements.txt 01_RAG/.env.example 01_RAG/README.md 01_RAG/app.py`

Expected: no production dependency references remain, except old design documents outside the command scope.

### Task 4: Verification

**Files:**
- Existing test suite.

- [ ] **Step 1: Run vectorstore/config tests**

Run: `cd 01_RAG && pytest tests/test_rag_pipeline.py::TestVectorStore tests/test_rag_pipeline.py::TestConfig -v`

Expected: PASS.

- [ ] **Step 2: Run non-slow project tests**

Run: `cd 01_RAG && pytest tests/ -v -k "not slow"`

Expected: PASS or report dependency/environment failures with exact output.
