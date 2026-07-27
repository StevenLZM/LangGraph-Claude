from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

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

InitializationStatus = Literal[
    "created",
    "aliases_repaired",
    "already_initialized",
]


@dataclass(frozen=True)
class IndexInitializationResult:
    status: InitializationStatus
    physical_index: str
    vector_dims: int
    read_alias: str
    write_alias: str


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

    def ensure_index(self, vector_dims: int) -> IndexInitializationResult:
        self._validate_vector_dims(vector_dims)
        index = self.config.PHYSICAL_INDEX
        if not self.client.indices.exists(index=index):
            self._reject_existing_aliases_before_create()
            try:
                self._create_index(vector_dims)
            except Exception as exc:
                if not _is_resource_already_exists(exc):
                    raise
            else:
                self._require_complete_aliases(self._read_aliases())
                return self._initialization_result("created", vector_dims)

        return self._validate_existing_index(vector_dims)

    def _create_index(self, vector_dims: int) -> None:
        index = self.config.PHYSICAL_INDEX
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

    def _validate_existing_index(
        self,
        vector_dims: int,
    ) -> IndexInitializationResult:
        mapping = self.client.indices.get_mapping(index=self.config.PHYSICAL_INDEX)
        actual_dims = self._mapping_vector_dims(mapping)
        if actual_dims != vector_dims:
            raise ValueError(
                f"向量维度不一致: index={actual_dims}, actual={vector_dims}"
            )

        aliases = self._read_aliases()
        actions = self._alias_actions_for_missing_aliases(aliases)
        if not actions:
            return self._initialization_result("already_initialized", vector_dims)

        self.client.indices.update_aliases(body={"actions": actions})
        remaining_actions = self._alias_actions_for_missing_aliases(
            self._read_aliases()
        )
        if remaining_actions:
            missing_aliases = ", ".join(
                action["add"]["alias"] for action in remaining_actions
            )
            raise RuntimeError(
                f"Elasticsearch 别名修复后仍缺失: {missing_aliases}"
            )
        return self._initialization_result("aliases_repaired", vector_dims)

    def _mapping_vector_dims(self, mapping: Mapping[str, Any]) -> int:
        try:
            embedding = mapping[self.config.PHYSICAL_INDEX]["mappings"][
                "properties"
            ]["embedding"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Elasticsearch mapping 缺少 embedding 字段") from exc
        if not isinstance(embedding, Mapping) or embedding.get("type") != "dense_vector":
            raise ValueError("Elasticsearch mapping 的 embedding 必须是 dense_vector")
        try:
            return int(embedding["dims"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Elasticsearch mapping 的 embedding 缺少有效 dims") from exc

    def _read_aliases(self) -> dict[str, Mapping[str, Any] | None]:
        aliases: dict[str, Mapping[str, Any] | None] = {}
        for alias_name in (self.config.READ_ALIAS, self.config.WRITE_ALIAS):
            if self.client.indices.exists_alias(name=alias_name):
                aliases[alias_name] = self.client.indices.get_alias(name=alias_name)
            else:
                aliases[alias_name] = None
        return aliases

    def _reject_existing_aliases_before_create(self) -> None:
        aliases = self._read_aliases()
        for alias_name, targets in aliases.items():
            if targets is not None:
                conflict_targets = ", ".join(sorted(targets)) or "<none>"
                raise ValueError(
                    f"别名冲突: {alias_name} 绑定到 {conflict_targets}，"
                    "物理索引尚不存在，不能创建或重绑别名"
                )

    def _require_complete_aliases(
        self,
        aliases: Mapping[str, Mapping[str, Any] | None],
    ) -> None:
        missing_actions = self._alias_actions_for_missing_aliases(aliases)
        if missing_actions:
            missing_aliases = ", ".join(
                action["add"]["alias"] for action in missing_actions
            )
            raise RuntimeError(f"Elasticsearch 索引创建后别名缺失: {missing_aliases}")

    def _alias_actions_for_missing_aliases(
        self,
        aliases: Mapping[str, Mapping[str, Any] | None],
    ) -> list[dict[str, Any]]:
        index = self.config.PHYSICAL_INDEX
        actions: list[dict[str, Any]] = []
        for alias_name in (self.config.READ_ALIAS, self.config.WRITE_ALIAS):
            targets = aliases[alias_name]
            if targets is None:
                add: dict[str, Any] = {"index": index, "alias": alias_name}
                if alias_name == self.config.WRITE_ALIAS:
                    add["is_write_index"] = True
                actions.append({"add": add})
                continue

            target_indices = set(targets)
            if target_indices != {index}:
                conflict_targets = ", ".join(sorted(target_indices)) or "<none>"
                raise ValueError(
                    f"别名冲突: {alias_name} 绑定到 {conflict_targets}，"
                    f"期望 {index}"
                )
            if alias_name == self.config.WRITE_ALIAS:
                write_attributes = targets[index].get("aliases", {}).get(
                    alias_name,
                    {},
                )
                if write_attributes.get("is_write_index") is not True:
                    raise ValueError(
                        f"别名冲突: {alias_name} 在 {index} 必须标记为 write index"
                    )
        return actions

    def _initialization_result(
        self,
        status: InitializationStatus,
        vector_dims: int,
    ) -> IndexInitializationResult:
        return IndexInitializationResult(
            status=status,
            physical_index=self.config.PHYSICAL_INDEX,
            vector_dims=vector_dims,
            read_alias=self.config.READ_ALIAS,
            write_alias=self.config.WRITE_ALIAS,
        )

    def _validate_vector_dims(self, vector_dims: int) -> None:
        if vector_dims <= 0:
            raise ValueError("向量维度必须大于 0")
        configured = int(self.config.EMBEDDING_DIMS)
        if configured and configured != vector_dims:
            raise ValueError(
                f"向量维度与 ES_EMBEDDING_DIMS 不一致: "
                f"configured={configured}, actual={vector_dims}"
            )

    def bulk_stage_children(
        self,
        documents: Sequence[Document],
        vectors: Sequence[Sequence[float]],
        ingest_run_id: str,
    ) -> int:
        if len(documents) != len(vectors):
            raise ValueError("Child 数量与向量数量不一致")
        if not documents:
            return 0
        vector_dims = len(vectors[0])
        if any(len(vector) != vector_dims for vector in vectors):
            raise ValueError("同一批次的向量维度不一致")
        self.ensure_index(vector_dims)

        operations: list[dict[str, Any]] = []
        for document, vector in zip(documents, vectors):
            metadata = document.metadata or {}
            storage_id = build_storage_id(
                doc_id=metadata.get("doc_id"),
                doc_version=metadata.get("doc_version"),
                child_id=metadata.get("child_id"),
            )
            source = document_to_source(
                document,
                vector,
                storage_id=storage_id,
                status="staging",
                ingest_run_id=ingest_run_id,
            )
            operations.extend(
                [
                    {
                        "index": {
                            "_index": self.config.WRITE_ALIAS,
                            "_id": storage_id,
                        }
                    },
                    source,
                ]
            )

        response = self.client.bulk(
            operations=operations,
            refresh="wait_for",
        )
        if response.get("errors"):
            errors = []
            for item in response.get("items", []):
                operation = next(iter(item.values()), {})
                if operation.get("error"):
                    reason = operation["error"].get(
                        "reason",
                        str(operation["error"]),
                    )
                    errors.append(
                        f"{operation.get('_id', 'unknown')}: {reason}"
                    )
            raise RuntimeError(
                "Elasticsearch Bulk 存在失败项: " + "; ".join(errors)
            )
        return len(documents)

    def activate_version(
        self,
        doc_id: str,
        doc_version: str,
        ingest_run_id: str,
    ) -> int:
        response = self.client.update_by_query(
            index=self.config.PHYSICAL_INDEX,
            query={
                "bool": {
                    "filter": [
                        {"term": {"doc_id": doc_id}},
                        {"term": {"doc_version": doc_version}},
                        {"term": {"ingest_run_id": ingest_run_id}},
                        {"term": {"status": "staging"}},
                    ]
                }
            },
            script={"source": "ctx._source.status = 'active'"},
            refresh=True,
            conflicts="proceed",
        )
        return int(response.get("updated", 0))

    def deactivate_other_versions(
        self,
        doc_id: str,
        active_doc_version: str,
    ) -> int:
        response = self.client.update_by_query(
            index=self.config.PHYSICAL_INDEX,
            query={
                "bool": {
                    "filter": [
                        {"term": {"doc_id": doc_id}},
                        {"term": {"status": "active"}},
                    ],
                    "must_not": [
                        {"term": {"doc_version": active_doc_version}},
                    ],
                }
            },
            script={"source": "ctx._source.status = 'inactive'"},
            refresh=True,
            conflicts="proceed",
        )
        return int(response.get("updated", 0))

    def delete_ingest_run(self, ingest_run_id: str) -> int:
        response = self.client.delete_by_query(
            index=self.config.PHYSICAL_INDEX,
            query={
                "bool": {
                    "filter": [
                        {"term": {"ingest_run_id": ingest_run_id}},
                        {"term": {"status": "staging"}},
                    ]
                }
            },
            refresh=True,
            conflicts="proceed",
        )
        return int(response.get("deleted", 0))

    def delete_document(self, doc_id: str) -> int:
        response = self.client.delete_by_query(
            index=self.config.PHYSICAL_INDEX,
            query={"term": {"doc_id": doc_id}},
            refresh=True,
            conflicts="proceed",
        )
        return int(response.get("deleted", 0))

    def delete_document_version(self, doc_id: str, doc_version: str) -> int:
        response = self.client.delete_by_query(
            index=self.config.PHYSICAL_INDEX,
            query={
                "bool": {
                    "filter": [
                        {"term": {"doc_id": doc_id}},
                        {"term": {"doc_version": doc_version}},
                    ]
                }
            },
            refresh=True,
            conflicts="proceed",
        )
        return int(response.get("deleted", 0))

    def reactivate_other_versions(
        self,
        doc_id: str,
        excluded_doc_version: str,
    ) -> int:
        response = self.client.update_by_query(
            index=self.config.PHYSICAL_INDEX,
            query={
                "bool": {
                    "filter": [
                        {"term": {"doc_id": doc_id}},
                        {"term": {"status": "inactive"}},
                    ],
                    "must_not": [
                        {"term": {"doc_version": excluded_doc_version}},
                    ],
                }
            },
            script={"source": "ctx._source.status = 'active'"},
            refresh=True,
            conflicts="proceed",
        )
        return int(response.get("updated", 0))

    def count_children(self, *, status: str = "active") -> int:
        response = self.client.count(
            index=self.config.READ_ALIAS,
            query={"term": {"status": status}},
        )
        return int(response.get("count", 0))

    def aggregate_documents(self) -> list[dict[str, Any]]:
        response = self.client.search(
            index=self.config.READ_ALIAS,
            size=0,
            query={"term": {"status": "active"}},
            aggs={
                "documents": {
                    "terms": {"field": "doc_id", "size": 10_000},
                    "aggs": {
                        "sample": {
                            "top_hits": {
                                "size": 1,
                                "_source": [
                                    "source",
                                    "doc_version",
                                    "page_number",
                                ],
                            }
                        }
                    },
                }
            },
        )
        documents: list[dict[str, Any]] = []
        for bucket in (
            response.get("aggregations", {})
            .get("documents", {})
            .get("buckets", [])
        ):
            hits = bucket.get("sample", {}).get("hits", {}).get("hits", [])
            source = dict(hits[0].get("_source") or {}) if hits else {}
            documents.append(
                {
                    "doc_id": str(bucket.get("key", "")),
                    "source": source.get("source", "未知"),
                    "doc_version": source.get("doc_version", ""),
                    "child_count": int(bucket.get("doc_count", 0)),
                    "total_chunks": int(bucket.get("doc_count", 0)),
                    "total_pages": 0,
                    "parent_count": 0,
                    "pages": [],
                }
            )
        return documents


def build_storage_id(
    *,
    doc_id: Any,
    doc_version: Any,
    child_id: Any,
) -> str:
    values = [str(value or "").strip() for value in (doc_id, doc_version, child_id)]
    if not all(values):
        raise ValueError("storage_id 需要非空 doc_id、doc_version 和 child_id")
    return ":".join(values)


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
    if isinstance(source.get("embedding"), list):
        metadata["retrieval_embedding"] = list(source["embedding"])
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


def _is_resource_already_exists(exc: Exception) -> bool:
    error = getattr(exc, "error", "")
    body = getattr(exc, "body", None)
    body_type = (
        body.get("error", {}).get("type", "")
        if isinstance(body, Mapping) and isinstance(body.get("error"), Mapping)
        else ""
    )
    return "resource_already_exists_exception" in {
        str(error),
        str(body_type),
    } or "resource_already_exists_exception" in str(exc)
