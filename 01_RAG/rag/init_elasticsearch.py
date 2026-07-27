"""Explicit, safe initialization for the RAG Elasticsearch child index."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Mapping

from config import elasticsearch_config, llm_config
from rag.elasticsearch_store import (
    ElasticsearchChildStore,
    InitializationStatus,
    create_elasticsearch_client,
)


REQUIRED_EMBEDDING_MODEL = "qwen3.7-text-embedding"
REQUIRED_EMBEDDING_DIMS = 1024
REQUIRED_PHYSICAL_INDEX = "rag-child-chunks-v1"
REQUIRED_READ_ALIAS = "rag-child-chunks-read"
REQUIRED_WRITE_ALIAS = "rag-child-chunks-write"
_KEY_PLACEHOLDERS = frozenset(
    value.casefold()
    for value in (
        "你的真实百炼APIKey",
        "your-api-key",
        "replace-me",
    )
)


@dataclass(frozen=True)
class ElasticsearchInitializationSummary:
    status: InitializationStatus
    physical_index: str
    vector_dims: int
    read_alias: str
    write_alias: str
    document_count: int


def validate_initialization_config(
    *,
    api_key: str,
    embedding_model: str,
    embedding_dims: int,
) -> None:
    """Validate the fixed embedding contract without exposing credentials."""
    if not api_key.strip() or api_key.strip().casefold() in _KEY_PLACEHOLDERS:
        raise ValueError("DASHSCOPE_API_KEY must be configured with a real key")
    if embedding_model != REQUIRED_EMBEDDING_MODEL:
        raise ValueError(
            "EMBEDDING_MODEL must be " f"{REQUIRED_EMBEDDING_MODEL}"
        )
    if embedding_dims != REQUIRED_EMBEDDING_DIMS:
        raise ValueError(
            "ES_EMBEDDING_DIMS must be " f"{REQUIRED_EMBEDDING_DIMS}"
        )


def initialize_elasticsearch(
    *,
    client: Any,
    store: ElasticsearchChildStore,
    api_key: str,
    embedding_model: str,
    embedding_dims: int,
) -> ElasticsearchInitializationSummary:
    """Initialize the index then independently verify its usable empty state."""
    validate_initialization_config(
        api_key=api_key,
        embedding_model=embedding_model,
        embedding_dims=embedding_dims,
    )
    _validate_initialization_targets(store)
    if not client.ping():
        raise ConnectionError("Elasticsearch ping failed")

    result = store.ensure_index(REQUIRED_EMBEDDING_DIMS)
    _validate_mapping(
        client.indices.get_mapping(index=REQUIRED_PHYSICAL_INDEX),
        physical_index=REQUIRED_PHYSICAL_INDEX,
        vector_dims=REQUIRED_EMBEDDING_DIMS,
    )
    _validate_aliases(
        client.indices.get_alias(index=REQUIRED_PHYSICAL_INDEX),
        physical_index=REQUIRED_PHYSICAL_INDEX,
        read_alias=REQUIRED_READ_ALIAS,
        write_alias=REQUIRED_WRITE_ALIAS,
    )
    count_response = client.count(index=REQUIRED_PHYSICAL_INDEX)
    document_count = _document_count(count_response)
    return ElasticsearchInitializationSummary(
        status=result.status,
        physical_index=REQUIRED_PHYSICAL_INDEX,
        vector_dims=REQUIRED_EMBEDDING_DIMS,
        read_alias=REQUIRED_READ_ALIAS,
        write_alias=REQUIRED_WRITE_ALIAS,
        document_count=document_count,
    )


def initialize_from_environment() -> ElasticsearchInitializationSummary:
    """Use the existing configuration singleton to initialize Elasticsearch."""
    client = create_elasticsearch_client()
    store = ElasticsearchChildStore(client=client)
    return initialize_elasticsearch(
        client=client,
        store=store,
        api_key=llm_config.DASHSCOPE_API_KEY,
        embedding_model=llm_config.EMBEDDING_MODEL,
        embedding_dims=elasticsearch_config.EMBEDDING_DIMS,
    )


def main() -> int:
    """Run initialization and print a non-sensitive, one-line result summary."""
    try:
        summary = initialize_from_environment()
    except Exception as exc:
        print(
            f"Elasticsearch initialization failed ({type(exc).__name__}).",
            file=sys.stderr,
        )
        return 1

    print(
        " ".join(
            (
                f"status={summary.status}",
                f"index={summary.physical_index}",
                f"dims={summary.vector_dims}",
                f"read_alias={summary.read_alias}",
                f"write_alias={summary.write_alias}",
                f"documents={summary.document_count}",
            )
        )
    )
    return 0


def _validate_initialization_targets(store: ElasticsearchChildStore) -> None:
    expected_targets = (
        ("PHYSICAL_INDEX", "ES_PHYSICAL_INDEX", REQUIRED_PHYSICAL_INDEX),
        ("READ_ALIAS", "ES_INDEX_READ_ALIAS", REQUIRED_READ_ALIAS),
        ("WRITE_ALIAS", "ES_INDEX_WRITE_ALIAS", REQUIRED_WRITE_ALIAS),
    )
    for attribute, field_name, expected_value in expected_targets:
        if getattr(store.config, attribute, None) != expected_value:
            raise ValueError(f"{field_name} must be {expected_value}")


def _validate_mapping(
    mapping: Mapping[str, Any],
    *,
    physical_index: str,
    vector_dims: int,
) -> None:
    try:
        embedding = mapping[physical_index]["mappings"]["properties"]["embedding"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Elasticsearch mapping is missing embedding") from exc
    if not isinstance(embedding, Mapping) or embedding.get("type") != "dense_vector":
        raise ValueError("Elasticsearch mapping embedding must be dense_vector")
    try:
        actual_dims = int(embedding["dims"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Elasticsearch mapping embedding dims are invalid") from exc
    if actual_dims != vector_dims:
        raise ValueError("Elasticsearch mapping embedding dims do not match")


def _validate_aliases(
    aliases: Mapping[str, Any],
    *,
    physical_index: str,
    read_alias: str,
    write_alias: str,
) -> None:
    try:
        index_aliases = aliases[physical_index]["aliases"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Elasticsearch aliases are missing") from exc
    if not isinstance(index_aliases, Mapping) or read_alias not in index_aliases:
        raise ValueError(f"Elasticsearch read alias is missing: {read_alias}")
    try:
        write_attributes = index_aliases[write_alias]
    except KeyError as exc:
        raise ValueError(f"Elasticsearch write alias is missing: {write_alias}") from exc
    if not isinstance(write_attributes, Mapping) or write_attributes.get(
        "is_write_index"
    ) is not True:
        raise ValueError(f"Elasticsearch write alias must be writable: {write_alias}")


def _document_count(response: Mapping[str, Any]) -> int:
    try:
        count = int(response["count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Elasticsearch count response is invalid") from exc
    if count < 0:
        raise ValueError("Elasticsearch document count is invalid")
    return count


if __name__ == "__main__":
    raise SystemExit(main())
