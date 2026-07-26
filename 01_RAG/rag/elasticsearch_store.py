from __future__ import annotations

from typing import Any, Mapping, Sequence

from langchain_core.documents import Document

from config import ElasticsearchConfig, elasticsearch_config

try:
    from elasticsearch import Elasticsearch
except ImportError:  # pragma: no cover - exercised by the explicit factory error
    Elasticsearch = None  # type: ignore[assignment]


FILTERABLE_FIELDS = frozenset(
    {
        "storage_id",
        "child_id",
        "parent_id",
        "doc_id",
        "doc_version",
        "source",
        "file_name",
        "page_number",
        "page_start",
        "page_end",
        "chunk_index",
        "section_path",
        "tenant_id",
        "acl_principals",
        "visibility",
        "status",
        "ingest_run_id",
        "document_type",
        "language",
        "tags",
        "created_at",
        "updated_at",
        "valid_from",
        "valid_to",
        "upload_date",
        "doc_date_min",
        "doc_date_max",
        "has_doc_date",
        "version",
        "version_rank",
        "authority_score",
        "content_hash",
        "embedding_model",
        "embedding_version",
    }
)

SOURCE_METADATA_FIELDS = (
    "child_id",
    "parent_id",
    "doc_id",
    "doc_version",
    "source",
    "file_name",
    "page_number",
    "page_range",
    "page_start",
    "page_end",
    "chunk_index",
    "section_path",
    "tenant_id",
    "acl_principals",
    "visibility",
    "document_type",
    "language",
    "tags",
    "created_at",
    "updated_at",
    "valid_from",
    "valid_to",
    "upload_date",
    "doc_date_min",
    "doc_date_max",
    "has_doc_date",
    "version",
    "version_rank",
    "authority_score",
    "content_hash",
    "embedding_model",
    "embedding_version",
)


def create_elasticsearch_client(
    *,
    config: ElasticsearchConfig = elasticsearch_config,
) -> Any:
    if Elasticsearch is None:
        raise ImportError(
            "缺少 Elasticsearch Python Client；请安装 elasticsearch>=8.19,<9"
        )

    kwargs: dict[str, Any] = {
        "verify_certs": config.VERIFY_CERTS,
        "request_timeout": config.REQUEST_TIMEOUT,
        "max_retries": config.MAX_RETRIES,
        "retry_on_timeout": True,
    }
    if config.API_KEY:
        kwargs["api_key"] = config.API_KEY
    elif config.USERNAME and config.PASSWORD:
        kwargs["basic_auth"] = (config.USERNAME, config.PASSWORD)
    if config.CA_CERTS:
        kwargs["ca_certs"] = config.CA_CERTS
    return Elasticsearch([config.URL], **kwargs)


class ElasticsearchChildStore:
    def __init__(
        self,
        *,
        client: Any | None = None,
        config: ElasticsearchConfig = elasticsearch_config,
    ) -> None:
        self.client = client or create_elasticsearch_client(config=config)
        self.config = config

    def ensure_index(self, vector_dims: int) -> None:
        self._validate_vector_dims(vector_dims)
        index = self.config.PHYSICAL_INDEX
        if self.client.indices.exists(index=index):
            mapping = self.client.indices.get_mapping(index=index)
            actual_dims = int(
                mapping[index]["mappings"]["properties"]["embedding"]["dims"]
            )
            if actual_dims != vector_dims:
                raise ValueError(
                    f"向量维度不一致: index={actual_dims}, actual={vector_dims}"
                )
            self.client.indices.update_aliases(
                body={
                    "actions": [
                        {
                            "add": {
                                "index": index,
                                "alias": self.config.READ_ALIAS,
                            }
                        },
                        {
                            "add": {
                                "index": index,
                                "alias": self.config.WRITE_ALIAS,
                                "is_write_index": True,
                            }
                        },
                    ]
                }
            )
            return

        body = _index_definition(
            vector_dims=vector_dims,
            read_alias=self.config.READ_ALIAS,
            write_alias=self.config.WRITE_ALIAS,
            shards=self.config.NUMBER_OF_SHARDS,
            replicas=self.config.NUMBER_OF_REPLICAS,
        )
        response = self.client.indices.create(index=index, body=body)
        if response.get("acknowledged") is False:
            raise RuntimeError(f"Elasticsearch 索引创建未确认: {index}")

    def _validate_vector_dims(self, vector_dims: int) -> None:
        if vector_dims <= 0:
            raise ValueError("向量维度必须大于 0")
        configured = int(self.config.EMBEDDING_DIMS)
        if configured and configured != vector_dims:
            raise ValueError(
                f"向量维度与 ES_EMBEDDING_DIMS 不一致: "
                f"configured={configured}, actual={vector_dims}"
            )


def build_es_filters(metadata_filter: Mapping[str, Any] | None) -> list[dict]:
    if not metadata_filter:
        return []
    if not isinstance(metadata_filter, Mapping):
        raise ValueError("metadata_filter 必须是对象")

    clauses: list[dict] = []
    for field, condition in metadata_filter.items():
        if field == "$and":
            if not _is_sequence(condition):
                raise ValueError("$and 必须是条件数组")
            for item in condition:
                clauses.extend(build_es_filters(item))
            continue
        if field == "$or":
            if not _is_sequence(condition):
                raise ValueError("$or 必须是条件数组")
            should = [
                clause
                for item in condition
                for clause in build_es_filters(item)
            ]
            clauses.append(
                {
                    "bool": {
                        "should": should,
                        "minimum_should_match": 1,
                    }
                }
            )
            continue
        if field not in FILTERABLE_FIELDS:
            raise ValueError(f"不允许过滤字段: {field}")
        clauses.extend(_field_filters(field, condition))
    return clauses


def document_to_source(
    doc: Document,
    vector: Sequence[float],
    *,
    storage_id: str,
    status: str,
    ingest_run_id: str,
) -> dict[str, Any]:
    metadata = dict(doc.metadata or {})
    source: dict[str, Any] = {
        "storage_id": str(storage_id),
        "content": str(doc.page_content or ""),
        "embedding": [float(value) for value in vector],
        "status": str(status),
        "ingest_run_id": str(ingest_run_id),
    }
    for field in SOURCE_METADATA_FIELDS:
        if field in metadata and metadata[field] is not None:
            source[field] = metadata[field]
    if "page_number" not in source and metadata.get("page") is not None:
        source["page_number"] = metadata["page"]
    return source


def hit_to_document(hit: Mapping[str, Any], retrieval_source: str) -> Document:
    source = dict(hit.get("_source") or {})
    metadata = {
        key: value
        for key, value in source.items()
        if key not in {"content", "embedding"}
    }
    metadata["storage_id"] = str(
        source.get("storage_id") or hit.get("_id") or ""
    )
    metadata["retrieval_source"] = str(retrieval_source)
    metadata["es_score"] = hit.get("_score")
    if "page" not in metadata and "page_number" in metadata:
        metadata["page"] = metadata["page_number"]
    return Document(
        page_content=str(source.get("content") or ""),
        metadata=metadata,
    )


def _field_filters(field: str, condition: Any) -> list[dict]:
    if not isinstance(condition, Mapping):
        return [{"term": {field: condition}}]

    clauses: list[dict] = []
    range_values: dict[str, Any] = {}
    for operator, value in condition.items():
        if operator in {"$gt", "$gte", "$lt", "$lte"}:
            range_values[operator[1:]] = value
        elif operator == "$in":
            if not _is_sequence(value):
                raise ValueError(f"{field} 的 $in 必须是数组")
            clauses.append({"terms": {field: list(value)}})
        elif operator == "$ne":
            clauses.append(
                {"bool": {"must_not": [{"term": {field: value}}]}}
            )
        elif operator == "$eq":
            clauses.append({"term": {field: value}})
        else:
            raise ValueError(f"不支持的过滤操作符: {operator}")
    if range_values:
        clauses.append({"range": {field: range_values}})
    return clauses


def _index_definition(
    *,
    vector_dims: int,
    read_alias: str,
    write_alias: str,
    shards: int,
    replicas: int,
) -> dict[str, Any]:
    keyword_fields = {
        field: {"type": "keyword"}
        for field in (
            "storage_id",
            "child_id",
            "parent_id",
            "doc_id",
            "doc_version",
            "source",
            "file_name",
            "page_range",
            "section_path",
            "tenant_id",
            "acl_principals",
            "visibility",
            "status",
            "ingest_run_id",
            "document_type",
            "language",
            "tags",
            "version",
            "content_hash",
            "embedding_model",
            "embedding_version",
        )
    }
    properties: dict[str, Any] = {
        **keyword_fields,
        "content": {"type": "text", "analyzer": "cjk"},
        "embedding": {
            "type": "dense_vector",
            "dims": vector_dims,
            "index": True,
            "similarity": "cosine",
        },
        **{
            field: {"type": "integer"}
            for field in (
                "page_number",
                "page_start",
                "page_end",
                "chunk_index",
                "upload_date",
                "doc_date_min",
                "doc_date_max",
                "version_rank",
            )
        },
        "has_doc_date": {"type": "boolean"},
        "authority_score": {"type": "float"},
        **{
            field: {"type": "date"}
            for field in ("created_at", "updated_at", "valid_from", "valid_to")
        },
    }
    return {
        "settings": {
            "number_of_shards": shards,
            "number_of_replicas": replicas,
        },
        "mappings": {
            "dynamic": "strict",
            "properties": properties,
        },
        "aliases": {
            read_alias: {},
            write_alias: {"is_write_index": True},
        },
    }


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )
