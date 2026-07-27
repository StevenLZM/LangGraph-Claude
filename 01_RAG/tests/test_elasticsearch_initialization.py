from types import SimpleNamespace

import pytest
from elastic_transport import ApiResponseMeta, HttpHeaders, NodeConfig, ObjectApiResponse

from rag.elasticsearch_store import IndexInitializationResult
from rag.init_elasticsearch import (
    initialize_elasticsearch,
    main,
    validate_initialization_config,
)


PHYSICAL_INDEX = "rag-child-chunks-v1"
READ_ALIAS = "rag-child-chunks-read"
WRITE_ALIAS = "rag-child-chunks-write"


class FakeIndices:
    def __init__(
        self,
        *,
        vector_dims=1024,
        aliases=None,
        global_aliases=None,
        alias_responses=None,
    ):
        self.vector_dims = vector_dims
        self.aliases = aliases or {
            READ_ALIAS: {},
            WRITE_ALIAS: {"is_write_index": True},
        }
        self.global_aliases = global_aliases or {
            alias_name: {PHYSICAL_INDEX: attributes}
            for alias_name, attributes in self.aliases.items()
        }
        self.alias_responses = alias_responses or {}
        self.read_indices = []

    def get_mapping(self, *, index):
        self.read_indices.append(("mapping", index))
        return {
            index: {
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "dense_vector",
                            "dims": self.vector_dims,
                        }
                    }
                }
            }
        }

    def get_alias(self, *, name):
        self.read_indices.append(("alias", name))
        if name in self.alias_responses:
            return self.alias_responses[name]
        return {
            target_index: {"aliases": {name: attributes}}
            for target_index, attributes in self.global_aliases.get(name, {}).items()
        }


class FakeClient:
    def __init__(
        self,
        *,
        ping_result=True,
        vector_dims=1024,
        aliases=None,
        global_aliases=None,
        alias_responses=None,
    ):
        self.ping_result = ping_result
        self.indices = FakeIndices(
            vector_dims=vector_dims,
            aliases=aliases,
            global_aliases=global_aliases,
            alias_responses=alias_responses,
        )
        self.ping_calls = 0
        self.count_indices = []

    def ping(self):
        self.ping_calls += 1
        return self.ping_result

    def count(self, *, index):
        self.count_indices.append(index)
        return {"count": 0}


class FakeStore:
    def __init__(self, *, status="created", client=None, config=None):
        self.ensure_calls = []
        self.status = status
        self.client = client
        self.config = config or SimpleNamespace(
            PHYSICAL_INDEX=PHYSICAL_INDEX,
            READ_ALIAS=READ_ALIAS,
            WRITE_ALIAS=WRITE_ALIAS,
        )

    def ensure_index(self, vector_dims):
        self.ensure_calls.append(vector_dims)
        return IndexInitializationResult(
            status=self.status,
            physical_index=self.config.PHYSICAL_INDEX,
            vector_dims=vector_dims,
            read_alias=self.config.READ_ALIAS,
            write_alias=self.config.WRITE_ALIAS,
        )


def _object_api_response(body):
    return ObjectApiResponse(
        body=body,
        meta=ApiResponseMeta(
            status=200,
            http_version="1.1",
            headers=HttpHeaders(),
            duration=0.0,
            node=NodeConfig(scheme="http", host="localhost", port=9200),
        ),
    )


@pytest.mark.parametrize(
    ("api_key", "model", "dims", "message"),
    [
        ("", "qwen3.7-text-embedding", 1024, "DASHSCOPE_API_KEY"),
        ("你的真实百炼APIKey", "qwen3.7-text-embedding", 1024, "DASHSCOPE_API_KEY"),
        ("sk-valid-value", "text-embedding-v3", 1024, "EMBEDDING_MODEL"),
        ("sk-valid-value", "qwen3.7-text-embedding", 768, "ES_EMBEDDING_DIMS"),
    ],
)
def test_validate_initialization_config_rejects_invalid_values(
    api_key,
    model,
    dims,
    message,
):
    """Rejecting a missing, placeholder, incompatible model, or wrong dimensions."""
    with pytest.raises(ValueError, match=message):
        validate_initialization_config(
            api_key=api_key,
            embedding_model=model,
            embedding_dims=dims,
        )


@pytest.mark.parametrize("placeholder", ["your-api-key", "replace-me"])
def test_validate_initialization_config_rejects_common_key_placeholders(placeholder):
    """Rejecting placeholder keys prevents a harmless-looking empty index setup."""
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        validate_initialization_config(
            api_key=placeholder,
            embedding_model="qwen3.7-text-embedding",
            embedding_dims=1024,
        )


def test_initialize_stops_before_index_write_when_ping_fails():
    """A failed cluster health check cannot reach the write-capable store."""
    client = FakeClient(ping_result=False)
    store = FakeStore()

    with pytest.raises(ConnectionError, match="Elasticsearch"):
        initialize_elasticsearch(
            client=client,
            store=store,
            api_key="sk-valid-value",
            embedding_model="qwen3.7-text-embedding",
            embedding_dims=1024,
        )

    assert store.ensure_calls == []


@pytest.mark.parametrize(
    ("config_attribute", "invalid_target", "message"),
    [
        ("PHYSICAL_INDEX", "wrong-child-index", "ES_PHYSICAL_INDEX"),
        ("READ_ALIAS", "wrong-child-read", "ES_INDEX_READ_ALIAS"),
        ("WRITE_ALIAS", "wrong-child-write", "ES_INDEX_WRITE_ALIAS"),
    ],
)
def test_initialize_rejects_noncanonical_targets_before_ping_or_index_write(
    config_attribute,
    invalid_target,
    message,
):
    """A configurable target must not redirect the formal initializer write."""
    config_values = {
        "PHYSICAL_INDEX": PHYSICAL_INDEX,
        "READ_ALIAS": READ_ALIAS,
        "WRITE_ALIAS": WRITE_ALIAS,
    }
    config_values[config_attribute] = invalid_target
    client = FakeClient()
    store = FakeStore(config=SimpleNamespace(**config_values))

    with pytest.raises(ValueError, match=message):
        initialize_elasticsearch(
            client=client,
            store=store,
            api_key="sk-valid-value",
            embedding_model="qwen3.7-text-embedding",
            embedding_dims=1024,
        )

    assert client.ping_calls == 0
    assert store.ensure_calls == []
    assert client.indices.read_indices == []
    assert client.count_indices == []


def test_initialize_returns_verified_empty_index_summary():
    """A successful setup reports only the independently re-read empty index state."""
    client = FakeClient()
    store = FakeStore()

    summary = initialize_elasticsearch(
        client=client,
        store=store,
        api_key="sk-valid-value",
        embedding_model="qwen3.7-text-embedding",
        embedding_dims=1024,
    )

    assert summary.status == "created"
    assert summary.physical_index == PHYSICAL_INDEX
    assert summary.vector_dims == 1024
    assert summary.document_count == 0
    assert summary.read_alias == READ_ALIAS
    assert summary.write_alias == WRITE_ALIAS
    assert client.indices.read_indices == [
        ("mapping", PHYSICAL_INDEX),
        ("alias", READ_ALIAS),
        ("alias", WRITE_ALIAS),
    ]
    assert client.count_indices == [PHYSICAL_INDEX]


def test_initialize_accepts_object_api_response_alias_readback():
    """Elasticsearch 8 transport object responses retain the full alias target map."""
    client = FakeClient(
        alias_responses={
            READ_ALIAS: _object_api_response(
                {PHYSICAL_INDEX: {"aliases": {READ_ALIAS: {}}}}
            ),
            WRITE_ALIAS: _object_api_response(
                {
                    PHYSICAL_INDEX: {
                        "aliases": {WRITE_ALIAS: {"is_write_index": True}}
                    }
                }
            ),
        }
    )
    store = FakeStore()

    summary = initialize_elasticsearch(
        client=client,
        store=store,
        api_key="sk-valid-value",
        embedding_model="qwen3.7-text-embedding",
        embedding_dims=1024,
    )

    assert summary.status == "created"


def test_initialize_rejects_non_dict_object_api_response_alias_readback():
    """A transport response whose body is not an alias object cannot validate."""
    client = FakeClient(
        alias_responses={
            READ_ALIAS: _object_api_response([]),
            WRITE_ALIAS: _object_api_response(
                {
                    PHYSICAL_INDEX: {
                        "aliases": {WRITE_ALIAS: {"is_write_index": True}}
                    }
                }
            ),
        }
    )
    store = FakeStore()

    with pytest.raises(ValueError, match="read alias"):
        initialize_elasticsearch(
            client=client,
            store=store,
            api_key="sk-valid-value",
            embedding_model="qwen3.7-text-embedding",
            embedding_dims=1024,
        )


@pytest.mark.parametrize("status", ["created", "aliases_repaired", "already_initialized"])
def test_initialize_rereads_every_successful_initialization_status(status):
    """A future shortcut that trusts ensure_index results would lose verification."""
    client = FakeClient()
    store = FakeStore(status=status)

    summary = initialize_elasticsearch(
        client=client,
        store=store,
        api_key="sk-valid-value",
        embedding_model="qwen3.7-text-embedding",
        embedding_dims=1024,
    )

    assert summary.status == status
    assert client.indices.read_indices == [
        ("mapping", PHYSICAL_INDEX),
        ("alias", READ_ALIAS),
        ("alias", WRITE_ALIAS),
    ]
    assert client.count_indices == [PHYSICAL_INDEX]


@pytest.mark.parametrize(
    ("vector_dims", "aliases", "message"),
    [
        (768, None, "embedding"),
        (1024, {READ_ALIAS: {}, WRITE_ALIAS: {}}, WRITE_ALIAS),
    ],
)
def test_initialize_rejects_inconsistent_readback(vector_dims, aliases, message):
    """A stale mapping or write alias must fail instead of producing a trusted summary."""
    client = FakeClient(vector_dims=vector_dims, aliases=aliases)
    store = FakeStore()

    with pytest.raises(ValueError, match=message):
        initialize_elasticsearch(
            client=client,
            store=store,
            api_key="sk-valid-value",
            embedding_model="qwen3.7-text-embedding",
            embedding_dims=1024,
        )


def test_initialize_rejects_global_read_alias_with_an_extra_target():
    """Physical-index readback alone must not hide a multi-target read alias."""
    client = FakeClient(
        global_aliases={
            READ_ALIAS: {
                PHYSICAL_INDEX: {},
                "another-index": {},
            },
            WRITE_ALIAS: {PHYSICAL_INDEX: {"is_write_index": True}},
        }
    )
    store = FakeStore()

    with pytest.raises(ValueError, match="read alias"):
        initialize_elasticsearch(
            client=client,
            store=store,
            api_key="sk-valid-value",
            embedding_model="qwen3.7-text-embedding",
            embedding_dims=1024,
        )


def test_main_prints_a_safe_initialization_summary(monkeypatch, capsys, caplog):
    """The CLI summary must not surface the configured DashScope credential."""
    from rag import init_elasticsearch

    unique_key = "sk-unique-secret-value"
    client = FakeClient()
    store = FakeStore(status="already_initialized", client=client)
    monkeypatch.setenv("DASHSCOPE_API_KEY", unique_key)
    monkeypatch.setattr(init_elasticsearch.llm_config, "DASHSCOPE_API_KEY", unique_key)
    monkeypatch.setattr(init_elasticsearch.llm_config, "EMBEDDING_MODEL", "qwen3.7-text-embedding")
    monkeypatch.setattr(init_elasticsearch.elasticsearch_config, "EMBEDDING_DIMS", 1024)
    monkeypatch.setattr(init_elasticsearch, "create_elasticsearch_client", lambda: client)
    monkeypatch.setattr(init_elasticsearch, "ElasticsearchChildStore", lambda *, client: store)

    assert main() == 0

    captured = capsys.readouterr()
    assert "status=already_initialized" in captured.out
    assert "documents=0" in captured.out
    assert unique_key not in captured.out
    assert unique_key not in captured.err
    assert unique_key not in caplog.text


def test_main_reports_known_validation_reason_without_unknown_exception_text(
    monkeypatch,
    capsys,
    caplog,
):
    """CLI failures distinguish known validation from arbitrary secret-bearing errors."""
    from rag import init_elasticsearch

    monkeypatch.setattr(
        init_elasticsearch,
        "initialize_from_environment",
        lambda: (_ for _ in ()).throw(
            ValueError("EMBEDDING_MODEL must be qwen3.7-text-embedding")
        ),
    )

    assert main() == 1

    known_failure = capsys.readouterr()
    assert "EMBEDDING_MODEL must be qwen3.7-text-embedding" in known_failure.err

    unique_secret = "do-not-print-this-unknown-exception-secret"
    monkeypatch.setattr(
        init_elasticsearch,
        "initialize_from_environment",
        lambda: (_ for _ in ()).throw(RuntimeError(unique_secret)),
    )

    assert main() == 1

    unknown_failure = capsys.readouterr()
    combined_output = (
        known_failure.out
        + known_failure.err
        + unknown_failure.out
        + unknown_failure.err
        + caplog.text
    )
    assert unique_secret not in combined_output
