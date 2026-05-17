"""
rag/vectorstore.py — child chunk 向量库管理
支持：文档增量添加、删除、查询、持久化，并与 parent docstore 协同工作。
"""
from __future__ import annotations

import json
from typing import Any, List, Optional

from langchain_core.documents import Document

try:
    from langchain_milvus import Milvus
except ImportError:
    Milvus = None  # type: ignore[assignment]

from config import milvus_config, rag_config
from rag.chunker import ChunkingResult
from rag.docstore import ParentDocStore, get_parent_docstore
from rag.embedder import get_embeddings


_vectorstore_instance: Optional[Any] = None


def _ensure_milvus_orm_connection(vectorstore: Any) -> None:
    """langchain-milvus 0.3.x 仍会用 ORM Collection，需要补注册 alias。"""
    try:
        from pymilvus.orm.connections import connections
    except ImportError:
        return

    alias = getattr(vectorstore, "alias", None)
    connection_args = getattr(vectorstore, "_connection_args", {}) or {}
    uri = connection_args.get("uri")
    if not alias or not uri or connections.has_connection(alias):
        return

    connect_kwargs = {
        "alias": alias,
        "uri": uri,
    }
    for key in ("user", "password", "db_name", "token", "timeout"):
        if connection_args.get(key):
            connect_kwargs[key] = connection_args[key]
    connections.connect(**connect_kwargs)


def _milvus_output_fields() -> list[str]:
    fields = [
        milvus_config.PRIMARY_FIELD,
        milvus_config.TEXT_FIELD,
        *milvus_config.METADATA_FIELDS,
    ]
    return list(dict.fromkeys(fields))


if Milvus is not None:
    class _MilvusWithOrmConnection(Milvus):  # type: ignore[misc, valid-type]
        def _init(self, *args, **kwargs) -> None:
            _ensure_milvus_orm_connection(self)
            super()._init(*args, **kwargs)

        def _collection_search(
            self,
            embedding_or_text: list[float] | dict[int, float] | str,
            k: int = 4,
            param: Optional[dict] = None,
            expr: Optional[str] = None,
            timeout: Optional[float] = None,
            **kwargs: Any,
        ) -> Optional[list[list[dict]]]:
            if self.col is None:
                return None

            if param is None:
                param = self._as_list(self.search_params)[0]

            return self.client.search(
                self.collection_name,
                data=[embedding_or_text],
                anns_field=self._vector_field,
                search_params=param,
                limit=k,
                filter=expr,
                output_fields=_milvus_output_fields(),
                timeout=self.timeout or timeout,
                **kwargs,
            )
else:
    _MilvusWithOrmConnection = None


def _get_milvus_class() -> Any:
    if Milvus is None:
        raise ImportError(
            "未安装 Milvus 依赖。请运行：pip install langchain-milvus 'pymilvus[milvus-lite]'"
        )
    if getattr(Milvus, "__module__", "").startswith("langchain_milvus"):
        return _MilvusWithOrmConnection
    return Milvus


def get_vectorstore(reset: bool = False) -> Any:
    """获取（或初始化）Milvus Lite 实例（单例）"""
    global _vectorstore_instance

    if _vectorstore_instance is not None and not reset:
        return _vectorstore_instance

    embeddings = get_embeddings()
    milvus_cls = _get_milvus_class()
    _vectorstore_instance = milvus_cls(
        embedding_function=embeddings,
        collection_name=milvus_config.COLLECTION_NAME,
        connection_args={"uri": milvus_config.URI},
        consistency_level=milvus_config.CONSISTENCY_LEVEL,
        index_params=milvus_config.INDEX_PARAMS,
        search_params=milvus_config.SEARCH_PARAMS,
        auto_id=False,
        primary_field=milvus_config.PRIMARY_FIELD,
        text_field=milvus_config.TEXT_FIELD,
        vector_field=milvus_config.VECTOR_FIELD,
        enable_dynamic_field=True,
    )
    return _vectorstore_instance


def add_documents(
    chunks: ChunkingResult | List[Document],
    doc_id: str,
    vectorstore: Optional[Any] = None,
    parent_docstore: Optional[ParentDocStore] = None,
) -> int:
    """
    增量添加 child chunks 到向量库，并将 parent chunks 写入 docstore。
    失败时执行文档级回滚，避免 child / parent 半成功。
    """
    vs = vectorstore or get_vectorstore()
    docstore = parent_docstore or get_parent_docstore()

    delete_document(doc_id, vs, docstore)

    if isinstance(chunks, ChunkingResult):
        parents = chunks.parents
        children = chunks.children
    else:
        parents = []
        children = chunks

    if not children:
        return 0

    try:
        if parents:
            docstore.upsert_parents(parents)

        ids = [
            child.metadata.get("child_id")
            or child.metadata.get("parent_id")
            or f"{doc_id}_{child.metadata.get('chunk_index', i)}"
            for i, child in enumerate(children)
        ]

        batch_size = 500
        added = 0
        for i in range(0, len(children), batch_size):
            batch_docs = children[i: i + batch_size]
            batch_ids = ids[i: i + batch_size]
            vs.add_documents(documents=batch_docs, ids=batch_ids)
            added += len(batch_docs)
        return added
    except Exception:
        delete_document(doc_id, vs, docstore)
        raise


def delete_document(
    doc_id: str,
    vectorstore: Optional[Any] = None,
    parent_docstore: Optional[ParentDocStore] = None,
) -> int:
    """删除指定 doc_id 的 child 向量和 parent 文档。"""
    vs = vectorstore or get_vectorstore()
    docstore = parent_docstore or get_parent_docstore()

    deleted_children = 0
    try:
        existing_ids = _get_ids_by_filter(vs, {"doc_id": doc_id})
        if existing_ids:
            vs.delete(ids=existing_ids)
            deleted_children = len(existing_ids)
    except Exception:
        pass

    try:
        docstore.delete_document(doc_id)
    except Exception:
        pass

    return deleted_children


def list_documents(
    vectorstore: Optional[Any] = None,
    parent_docstore: Optional[ParentDocStore] = None,
) -> List[dict]:
    """
    列出索引中的所有文档，按 doc_id 聚合 parent / child 数量。

    Returns:
        [{"doc_id": ..., "source": ..., "parent_count": ..., "child_count": ...}]
    """
    vs = vectorstore or get_vectorstore()
    docstore = parent_docstore or get_parent_docstore()

    docs: dict[str, dict] = {}

    try:
        result = _get_vectorstore_records(vs, include=["metadatas"])
        metadatas = result.get("metadatas", [])
    except Exception:
        metadatas = []

    for meta in metadatas:
        if not meta:
            continue
        doc_id = meta.get("doc_id", "unknown")
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "source": meta.get("source", "未知"),
                "total_pages": meta.get("total_pages", 0),
                "total_chunks": 0,
                "child_count": 0,
                "parent_count": 0,
                "doc_version": meta.get("doc_version", ""),
                "pages": set(),
            }
        docs[doc_id]["total_chunks"] += 1
        docs[doc_id]["child_count"] += 1
        page = meta.get("page")
        if page:
            docs[doc_id]["pages"].add(page)

    try:
        parent_docs = docstore.list_documents()
    except Exception:
        parent_docs = []

    for parent_doc in parent_docs:
        doc_id = parent_doc["doc_id"]
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "source": parent_doc.get("source", "未知"),
                "total_pages": 0,
                "total_chunks": 0,
                "child_count": 0,
                "parent_count": 0,
                "doc_version": parent_doc.get("doc_version", ""),
                "pages": set(),
            }
        docs[doc_id]["parent_count"] = parent_doc.get("parent_count", 0)
        docs[doc_id]["doc_version"] = parent_doc.get("doc_version", docs[doc_id]["doc_version"])
        if not docs[doc_id]["source"] or docs[doc_id]["source"] == "未知":
            docs[doc_id]["source"] = parent_doc.get("source", "未知")

    return [{**doc, "pages": sorted(doc["pages"])} for doc in docs.values()]


def get_collection_stats(
    vectorstore: Optional[Any] = None,
    parent_docstore: Optional[ParentDocStore] = None,
) -> dict:
    """返回 child collection 与 parent docstore 统计信息。"""
    vs = vectorstore or get_vectorstore()
    docstore = parent_docstore or get_parent_docstore()

    try:
        if hasattr(vs, "_collection"):
            child_count = vs._collection.count()
        elif hasattr(vs, "col") and hasattr(vs.col, "num_entities"):
            child_count = vs.col.num_entities
        elif hasattr(vs, "client") and hasattr(vs, "collection_name"):
            stats = vs.client.get_collection_stats(collection_name=vs.collection_name)
            child_count = int(stats.get("row_count", 0))
        else:
            child_count = 0
    except Exception:
        child_count = 0

    try:
        parent_count = docstore.count()
    except Exception:
        parent_count = 0

    return {
        "total_chunks": child_count,
        "total_children": child_count,
        "total_parents": parent_count,
        "collection_name": milvus_config.COLLECTION_NAME,
        "persist_dir": milvus_config.URI,
        "backend": "milvus-lite",
    }


_MILVUS_OPERATORS = {
    "$eq": "==",
    "$ne": "!=",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}


def _format_milvus_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return "null"
    return str(value)


def _format_milvus_list(values: list[Any]) -> str:
    return "[" + ", ".join(_format_milvus_value(v) for v in values) + "]"


def _to_milvus_filter(metadata_filter: Optional[dict]) -> str:
    """把项目现有 metadata filter 子集转换为 Milvus boolean expression。"""
    if not metadata_filter:
        return ""

    if "$and" in metadata_filter:
        clauses = [_to_milvus_filter(item) for item in metadata_filter["$and"]]
        return " and ".join(clause for clause in clauses if clause)

    if "$or" in metadata_filter:
        clauses = [_to_milvus_filter(item) for item in metadata_filter["$or"]]
        return " or ".join(f"({clause})" for clause in clauses if clause)

    clauses: list[str] = []
    for field, value in metadata_filter.items():
        if isinstance(value, dict):
            for operator, operand in value.items():
                if operator == "$in" and isinstance(operand, list):
                    clauses.append(f"{field} in {_format_milvus_list(operand)}")
                    continue
                milvus_operator = _MILVUS_OPERATORS.get(operator)
                if milvus_operator is None:
                    raise ValueError(f"不支持的 metadata filter 操作符: {operator}")
                clauses.append(f"{field} {milvus_operator} {_format_milvus_value(operand)}")
        elif isinstance(value, list):
            clauses.append(f"{field} in {_format_milvus_list(value)}")
        else:
            clauses.append(f"{field} == {_format_milvus_value(value)}")

    return " and ".join(clauses)


def _all_milvus_rows_expr() -> str:
    return f'{milvus_config.PRIMARY_FIELD} != ""'


def _get_ids_by_filter(vectorstore: Any, metadata_filter: dict) -> list[str]:
    if hasattr(vectorstore, "get"):
        result = vectorstore.get(where=metadata_filter)
        return list(result.get("ids") or [])

    expr = _to_milvus_filter(metadata_filter)
    if not expr:
        return []

    if hasattr(vectorstore, "get_pks"):
        return list(vectorstore.get_pks(expr=expr) or [])

    result = _get_vectorstore_records(vectorstore, metadata_filter=metadata_filter, include=["ids"])
    return list(result.get("ids") or [])


def _query_milvus_rows(vectorstore: Any, expr: str, limit: int = 10000) -> list[Any]:
    if hasattr(vectorstore, "client") and hasattr(vectorstore, "collection_name"):
        return list(vectorstore.client.query(
            collection_name=vectorstore.collection_name,
            filter=expr,
            output_fields=_milvus_output_fields(),
            limit=limit,
        ))

    if hasattr(vectorstore, "search_by_metadata"):
        return list(vectorstore.search_by_metadata(expr=expr, limit=limit))

    return []


def _rows_to_records(rows: list[Any], include: Optional[list[str]] = None) -> dict:
    include = include or ["ids", "documents", "metadatas"]
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for row in rows:
        if isinstance(row, Document):
            ids.append(str(row.metadata.get(milvus_config.PRIMARY_FIELD, "")))
            documents.append(row.page_content)
            metadatas.append(dict(row.metadata or {}))
            continue

        if not isinstance(row, dict):
            continue

        item = dict(row)
        row_id = item.pop(milvus_config.PRIMARY_FIELD, "")
        text = item.pop(milvus_config.TEXT_FIELD, "")
        item.pop(milvus_config.VECTOR_FIELD, None)

        nested_metadata = item.pop("metadata", None)
        if isinstance(nested_metadata, dict):
            item.update(nested_metadata)

        ids.append(str(row_id))
        documents.append(text)
        metadatas.append(item)

    result: dict[str, list] = {}
    if "ids" in include:
        result["ids"] = ids
    if "documents" in include:
        result["documents"] = documents
    if "metadatas" in include:
        result["metadatas"] = metadatas
    return result


def _get_vectorstore_records(
    vectorstore: Any,
    include: Optional[list[str]] = None,
    metadata_filter: Optional[dict] = None,
    limit: int = 10000,
) -> dict:
    if hasattr(vectorstore, "get"):
        kwargs: dict[str, Any] = {}
        if include is not None:
            kwargs["include"] = include
        if metadata_filter:
            kwargs["where"] = metadata_filter
        return vectorstore.get(**kwargs)

    expr = _to_milvus_filter(metadata_filter) if metadata_filter else _all_milvus_rows_expr()
    rows = _query_milvus_rows(vectorstore, expr=expr, limit=limit)
    return _rows_to_records(rows, include=include)


def get_all_child_documents(vectorstore: Optional[Any] = None) -> list[Document]:
    """读取全部 child chunks，用于构建 BM25 索引。"""
    vs = vectorstore or get_vectorstore()
    result = _get_vectorstore_records(vs, include=["documents", "metadatas"])
    return [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in zip(result.get("documents", []), result.get("metadatas", []))
    ]


def build_time_filter(time_intent: Optional[dict]) -> Optional[dict]:
    """
    把 query_rewriter 输出的 time_intent 转为项目通用 metadata filter。

    规则：
      - type in {year, before, after, range} → 硬过滤
      - type in {latest, none} → 返回 None（由 retriever 层做软排序 / 不处理）
      - field=upload_date → 单值字段直接 gte/lte
      - field=doc_date → 区间字段 [doc_date_min, doc_date_max]，并排除 has_doc_date=False
    """
    if not time_intent:
        return None
    t = time_intent.get("type")
    if t not in {"year", "before", "after", "range"}:
        return None

    rng = time_intent.get("range") or {}
    gte = rng.get("gte")
    lte = rng.get("lte")
    if gte is None or lte is None:
        return None

    field = time_intent.get("field", "doc_date")
    if field == "upload_date":
        return {"upload_date": {"$gte": gte, "$lte": lte}}

    # doc_date：存为 [min, max] 区间，判断两区间是否相交
    # 相交条件：min ≤ lte 且 max ≥ gte
    return {
        "$and": [
            {"has_doc_date": True},
            {"doc_date_min": {"$lte": lte}},
            {"doc_date_max": {"$gte": gte}},
        ]
    }


def similarity_search_with_threshold(
    query: str,
    k: int = 4,
    threshold: float | None = None,
    vectorstore: Optional[Any] = None,
    filter_doc_ids: Optional[List[str]] = None,
    metadata_filter: Optional[dict] = None,
) -> List[Document]:
    """
    带相似度阈值过滤的 child 级语义检索。

    metadata_filter: 项目通用 metadata filter，用于时间等硬过滤。
    """
    vs = vectorstore or get_vectorstore()
    threshold = threshold if threshold is not None else rag_config.SIMILARITY_THRESHOLD

    kwargs = {"query": query, "k": k}
    if metadata_filter:
        expr = _to_milvus_filter(metadata_filter)
        if expr:
            kwargs["expr"] = expr

    results_with_scores = vs.similarity_search_with_relevance_scores(**kwargs)

    filtered = []
    for doc, score in results_with_scores:
        if filter_doc_ids and doc.metadata.get("doc_id") not in filter_doc_ids:
            continue
        if score >= threshold:
            doc.metadata["similarity_score"] = round(score, 4)
            filtered.append(doc)

    return filtered
