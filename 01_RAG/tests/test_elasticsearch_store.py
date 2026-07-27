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
        self.create_calls = 0

    def exists(self, *, index):
        return False

    def create(self, *, index, body):
        self.create_calls += 1
        self.created = (index, body)
        return {"acknowledged": True}


class _CreateClient:
    def __init__(self):
        self.indices = _CreateIndices()


def test_ensure_index_creates_versioned_index_and_aliases():
    from rag.elasticsearch_store import ElasticsearchChildStore

    client = _CreateClient()
    store = ElasticsearchChildStore(client=client, config=_config())

    result = store.ensure_index(vector_dims=3)

    assert result.status == "created"
    assert result.physical_index == "rag-child-chunks-v1"
    assert result.vector_dims == 3
    assert result.read_alias == "rag-child-chunks-read"
    assert result.write_alias == "rag-child-chunks-write"

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
    def __init__(
        self,
        *,
        mapping=None,
        aliases=None,
    ):
        self.mapping = mapping or {
            "mappings": {
                "properties": {
                    "embedding": {"type": "dense_vector", "dims": 3}
                }
            }
        }
        self.aliases = aliases or {
            "rag-child-chunks-read": {"rag-child-chunks-v1": {}},
            "rag-child-chunks-write": {
                "rag-child-chunks-v1": {"is_write_index": True}
            },
        }
        self.create_calls = 0
        self.update_alias_calls = []

    def exists(self, *, index):
        return True

    def get_mapping(self, *, index):
        return {index: self.mapping}

    def exists_alias(self, *, name):
        return bool(self.aliases.get(name))

    def get_alias(self, *, name):
        return {
            index: {"aliases": {name: attributes}}
            for index, attributes in self.aliases.get(name, {}).items()
        }

    def update_aliases(self, *, body):
        self.update_alias_calls.append(body)
        for action in body["actions"]:
            add = action["add"]
            self.aliases.setdefault(add["alias"], {})[add["index"]] = {
                key: value
                for key, value in add.items()
                if key not in {"alias", "index"}
            }
        return {"acknowledged": True}


class _ExistingClient:
    def __init__(self, **kwargs):
        self.indices = _ExistingIndices(**kwargs)


def test_ensure_index_rejects_existing_mapping_dimension_mismatch():
    from rag.elasticsearch_store import ElasticsearchChildStore

    store = ElasticsearchChildStore(client=_ExistingClient(), config=_config())

    with pytest.raises(ValueError, match="向量维度"):
        store.ensure_index(vector_dims=4)


def test_ensure_index_is_noop_when_mapping_and_aliases_match():
    from rag.elasticsearch_store import ElasticsearchChildStore

    client = _ExistingClient()
    store = ElasticsearchChildStore(client=client, config=_config())

    result = store.ensure_index(vector_dims=3)

    assert result.status == "already_initialized"
    assert client.indices.create_calls == 0
    assert client.indices.update_alias_calls == []


@pytest.mark.parametrize(
    ("missing_alias", "expected_action"),
    [
        (
            "rag-child-chunks-read",
            {
                "add": {
                    "index": "rag-child-chunks-v1",
                    "alias": "rag-child-chunks-read",
                }
            },
        ),
        (
            "rag-child-chunks-write",
            {
                "add": {
                    "index": "rag-child-chunks-v1",
                    "alias": "rag-child-chunks-write",
                    "is_write_index": True,
                }
            },
        ),
    ],
)
def test_ensure_index_repairs_only_missing_alias(missing_alias, expected_action):
    from rag.elasticsearch_store import ElasticsearchChildStore

    aliases = {
        "rag-child-chunks-read": {"rag-child-chunks-v1": {}},
        "rag-child-chunks-write": {
            "rag-child-chunks-v1": {"is_write_index": True}
        },
    }
    aliases.pop(missing_alias)
    client = _ExistingClient(aliases=aliases)
    store = ElasticsearchChildStore(client=client, config=_config())

    result = store.ensure_index(vector_dims=3)

    assert result.status == "aliases_repaired"
    assert client.indices.update_alias_calls == [{"actions": [expected_action]}]


def test_ensure_index_rejects_repair_when_alias_is_still_missing_after_update():
    from rag.elasticsearch_store import ElasticsearchChildStore

    class MissingAliasAfterUpdateIndices(_ExistingIndices):
        def update_aliases(self, *, body):
            response = super().update_aliases(body=body)
            self.aliases.pop("rag-child-chunks-write")
            return response

    class MissingAliasAfterUpdateClient:
        def __init__(self):
            self.indices = MissingAliasAfterUpdateIndices(
                aliases={
                    "rag-child-chunks-read": {"rag-child-chunks-v1": {}},
                }
            )

    store = ElasticsearchChildStore(
        client=MissingAliasAfterUpdateClient(),
        config=_config(),
    )

    with pytest.raises(RuntimeError, match="别名修复后仍缺失"):
        store.ensure_index(vector_dims=3)


@pytest.mark.parametrize(
    ("alias_name", "other_index"),
    [
        ("rag-child-chunks-read", "another-index"),
        ("rag-child-chunks-write", "another-index"),
    ],
)
def test_ensure_index_rejects_alias_bound_to_another_index(
    alias_name,
    other_index,
):
    from rag.elasticsearch_store import ElasticsearchChildStore

    aliases = {
        "rag-child-chunks-read": {"rag-child-chunks-v1": {}},
        "rag-child-chunks-write": {
            "rag-child-chunks-v1": {"is_write_index": True}
        },
    }
    aliases[alias_name] = {other_index: {}}
    client = _ExistingClient(aliases=aliases)
    store = ElasticsearchChildStore(client=client, config=_config())

    with pytest.raises(ValueError, match="别名冲突"):
        store.ensure_index(vector_dims=3)

    assert client.indices.update_alias_calls == []


@pytest.mark.parametrize(
    "mapping",
    [
        {"mappings": {"properties": {}}},
        {
            "mappings": {
                "properties": {
                    "embedding": {"type": "keyword", "dims": 3}
                }
            }
        },
    ],
)
def test_ensure_index_rejects_missing_or_non_vector_embedding_mapping(mapping):
    from rag.elasticsearch_store import ElasticsearchChildStore

    client = _ExistingClient(mapping=mapping)
    store = ElasticsearchChildStore(client=client, config=_config())

    with pytest.raises(ValueError, match="embedding"):
        store.ensure_index(vector_dims=3)

    assert client.indices.update_alias_calls == []


def test_ensure_index_rejects_existing_mapping_dimension_mismatch_before_alias_update():
    from rag.elasticsearch_store import ElasticsearchChildStore

    client = _ExistingClient(
        mapping={
            "mappings": {
                "properties": {
                    "embedding": {"type": "dense_vector", "dims": 4}
                }
            }
        }
    )
    store = ElasticsearchChildStore(client=client, config=_config())

    with pytest.raises(ValueError, match="向量维度"):
        store.ensure_index(vector_dims=3)

    assert client.indices.update_alias_calls == []


def test_ensure_index_rejects_write_alias_without_write_index_marker():
    from rag.elasticsearch_store import ElasticsearchChildStore

    aliases = {
        "rag-child-chunks-read": {"rag-child-chunks-v1": {}},
        "rag-child-chunks-write": {"rag-child-chunks-v1": {}},
    }
    client = _ExistingClient(aliases=aliases)
    store = ElasticsearchChildStore(client=client, config=_config())

    with pytest.raises(ValueError, match="rag-child-chunks-write"):
        store.ensure_index(vector_dims=3)

    assert client.indices.update_alias_calls == []


class _ConcurrentCreateIndices(_ExistingIndices):
    def __init__(self):
        super().__init__()
        self.exists_calls = 0

    def exists(self, *, index):
        self.exists_calls += 1
        return self.exists_calls > 1

    def create(self, *, index, body):
        self.create_calls += 1
        raise RuntimeError("resource_already_exists_exception")


class _ConcurrentCreateClient:
    def __init__(self):
        self.indices = _ConcurrentCreateIndices()


def test_ensure_index_recovers_from_concurrent_create_race():
    from rag.elasticsearch_store import ElasticsearchChildStore

    client = _ConcurrentCreateClient()
    store = ElasticsearchChildStore(client=client, config=_config())

    result = store.ensure_index(vector_dims=3)

    assert result.status == "already_initialized"
    assert client.indices.create_calls == 1
    assert client.indices.update_alias_calls == []


def test_ensure_index_reraises_non_concurrent_create_error():
    from rag.elasticsearch_store import ElasticsearchChildStore

    class FailingIndices(_CreateIndices):
        def create(self, *, index, body):
            raise RuntimeError("cluster unavailable")

    client = _CreateClient()
    client.indices = FailingIndices()
    store = ElasticsearchChildStore(client=client, config=_config())

    with pytest.raises(RuntimeError, match="cluster unavailable"):
        store.ensure_index(vector_dims=3)


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
    assert restored.metadata["retrieval_embedding"] == [0.1, 0.2, 0.3]


def test_storage_id_includes_document_version():
    from rag.elasticsearch_store import build_storage_id

    assert build_storage_id(
        doc_id="doc-1",
        doc_version="v2",
        child_id="child-9",
    ) == "doc-1:v2:child-9"
    assert build_storage_id(
        doc_id="doc-1",
        doc_version="v3",
        child_id="child-9",
    ) != build_storage_id(
        doc_id="doc-1",
        doc_version="v2",
        child_id="child-9",
    )


def test_bulk_stage_children_rejects_partial_bulk_failure():
    from rag.elasticsearch_store import ElasticsearchChildStore

    class FakeIndices:
        def exists(self, *, index):
            return True

        def get_mapping(self, *, index):
            return {
                index: {
                    "mappings": {
                        "properties": {
                            "embedding": {"type": "dense_vector", "dims": 3}
                        }
                    }
                }
            }

        def exists_alias(self, *, name):
            return True

        def get_alias(self, *, name):
            attributes = (
                {"is_write_index": True}
                if name == "rag-child-chunks-write"
                else {}
            )
            return {"rag-child-chunks-v1": {"aliases": {name: attributes}}}

        def update_aliases(self, *, body):
            return {"acknowledged": True}

    class FakeClient:
        indices = FakeIndices()

        def bulk(self, **kwargs):
            return {
                "errors": True,
                "items": [
                    {
                        "index": {
                            "_id": "doc-1:v1:child-1",
                            "status": 400,
                            "error": {"reason": "mapping rejected"},
                        }
                    }
                ],
            }

    child = Document(
        page_content="child",
        metadata={
            "doc_id": "doc-1",
            "doc_version": "v1",
            "parent_id": "parent-1",
            "child_id": "child-1",
        },
    )
    store = ElasticsearchChildStore(client=FakeClient(), config=_config())

    with pytest.raises(RuntimeError, match="mapping rejected"):
        store.bulk_stage_children([child], [[0.1, 0.2, 0.3]], "run-1")
