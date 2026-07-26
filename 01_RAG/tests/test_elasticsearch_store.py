from types import SimpleNamespace

import pytest
from langchain_core.documents import Document


def _config(**overrides):
    values = {
        "URL": "http://127.0.0.1:9200",
        "PHYSICAL_INDEX": "rag-child-chunks-v1",
        "READ_ALIAS": "rag-child-chunks-read",
        "WRITE_ALIAS": "rag-child-chunks-write",
        "NUMBER_OF_SHARDS": 1,
        "NUMBER_OF_REPLICAS": 0,
        "VERIFY_CERTS": False,
        "USERNAME": "",
        "PASSWORD": "",
        "API_KEY": "",
        "CA_CERTS": "",
        "REQUEST_TIMEOUT": 30,
        "MAX_RETRIES": 3,
        "BULK_CHUNK_SIZE": 500,
        "EMBEDDING_DIMS": 0,
        "DENSE_NUM_CANDIDATES": 300,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_create_client_uses_local_url_without_auth(monkeypatch):
    from rag import elasticsearch_store

    captured = {}

    class FakeElasticsearch:
        def __init__(self, hosts, **kwargs):
            captured["hosts"] = hosts
            captured.update(kwargs)

    monkeypatch.setattr(elasticsearch_store, "Elasticsearch", FakeElasticsearch)

    elasticsearch_store.create_elasticsearch_client(config=_config())

    assert captured["hosts"] == ["http://127.0.0.1:9200"]
    assert "basic_auth" not in captured
    assert "api_key" not in captured
    assert captured["verify_certs"] is False


def test_create_client_prefers_api_key_over_basic_auth(monkeypatch):
    from rag import elasticsearch_store

    captured = {}

    class FakeElasticsearch:
        def __init__(self, hosts, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(elasticsearch_store, "Elasticsearch", FakeElasticsearch)

    elasticsearch_store.create_elasticsearch_client(
        config=_config(API_KEY="api-key", USERNAME="user", PASSWORD="password")
    )

    assert captured["api_key"] == "api-key"
    assert "basic_auth" not in captured


class _CreateIndices:
    def __init__(self):
        self.created = None

    def exists(self, *, index):
        return False

    def create(self, *, index, body):
        self.created = (index, body)
        return {"acknowledged": True}


class _CreateClient:
    def __init__(self):
        self.indices = _CreateIndices()


def test_ensure_index_creates_versioned_index_and_aliases():
    from rag.elasticsearch_store import ElasticsearchChildStore

    client = _CreateClient()
    store = ElasticsearchChildStore(client=client, config=_config())

    store.ensure_index(vector_dims=3)

    index, body = client.indices.created
    assert index == "rag-child-chunks-v1"
    assert body["settings"]["number_of_replicas"] == 0
    assert body["mappings"]["properties"]["content"]["analyzer"] == "cjk"
    assert body["mappings"]["properties"]["embedding"]["dims"] == 3
    assert body["mappings"]["properties"]["embedding"]["similarity"] == "cosine"
    assert body["aliases"]["rag-child-chunks-read"] == {}
    assert body["aliases"]["rag-child-chunks-write"]["is_write_index"] is True


def test_ensure_index_rejects_configured_embedding_dimension_mismatch():
    from rag.elasticsearch_store import ElasticsearchChildStore

    store = ElasticsearchChildStore(
        client=_CreateClient(),
        config=_config(EMBEDDING_DIMS=4),
    )

    with pytest.raises(ValueError, match="向量维度"):
        store.ensure_index(vector_dims=3)


class _ExistingIndices:
    def exists(self, *, index):
        return True

    def get_mapping(self, *, index):
        return {
            "rag-child-chunks-v1": {
                "mappings": {
                    "properties": {
                        "embedding": {"type": "dense_vector", "dims": 3}
                    }
                }
            }
        }

    def update_aliases(self, *, body):
        return {"acknowledged": True}


class _ExistingClient:
    def __init__(self):
        self.indices = _ExistingIndices()


def test_ensure_index_rejects_existing_mapping_dimension_mismatch():
    from rag.elasticsearch_store import ElasticsearchChildStore

    store = ElasticsearchChildStore(client=_ExistingClient(), config=_config())

    with pytest.raises(ValueError, match="向量维度"):
        store.ensure_index(vector_dims=4)


def test_build_es_filters_supports_date_range_and_status():
    from rag.elasticsearch_store import build_es_filters

    filters = build_es_filters(
        {
            "$and": [
                {"status": "active"},
                {"has_doc_date": True},
                {"doc_date_min": {"$lte": 20241231}},
                {"doc_date_max": {"$gte": 20240101}},
            ]
        }
    )

    assert {"term": {"status": "active"}} in filters
    assert {"term": {"has_doc_date": True}} in filters
    assert {"range": {"doc_date_min": {"lte": 20241231}}} in filters
    assert {"range": {"doc_date_max": {"gte": 20240101}}} in filters


def test_build_es_filters_rejects_unknown_operator():
    from rag.elasticsearch_store import build_es_filters

    with pytest.raises(ValueError, match="操作符"):
        build_es_filters({"doc_id": {"$regex": ".*"}})


def test_build_es_filters_rejects_unknown_field():
    from rag.elasticsearch_store import build_es_filters

    with pytest.raises(ValueError, match="字段"):
        build_es_filters({"password": "secret"})


def test_document_conversion_preserves_stable_identity_and_score():
    from rag.elasticsearch_store import document_to_source, hit_to_document

    doc = Document(
        page_content="保修期为 12 个月",
        metadata={
            "doc_id": "doc-1",
            "doc_version": "v2",
            "parent_id": "parent-1",
            "child_id": "child-1",
            "source": "manual.pdf",
            "page_range": "3",
        },
    )

    source = document_to_source(
        doc,
        [0.1, 0.2, 0.3],
        storage_id="doc-1:v2:child-1",
        status="active",
        ingest_run_id="run-1",
    )
    restored = hit_to_document(
        {"_id": source["storage_id"], "_score": 4.2, "_source": source},
        retrieval_source="bm25",
    )

    assert source["embedding"] == [0.1, 0.2, 0.3]
    assert restored.page_content == doc.page_content
    assert restored.metadata["child_id"] == "child-1"
    assert restored.metadata["retrieval_source"] == "bm25"
    assert restored.metadata["es_score"] == 4.2
