from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from memory.long_term import extract_memory_candidates


@dataclass(frozen=True)
class StructuredMemory:
    user_id: str
    category: str
    entity_type: str
    key: str
    value: dict[str, Any]
    content: str
    source_session_id: str
    source_request_id: str
    confidence: float = 0.8
    memory_id: str = ""
    status: str = "active"

    def normalized_id(self) -> str:
        if self.memory_id:
            return self.memory_id
        digest = hashlib.sha256(
            json.dumps(
                {
                    "user_id": self.user_id,
                    "category": self.category,
                    "key": self.key,
                    "content": self.content,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:24]
        return f"mem_{digest}"


@dataclass(frozen=True)
class SemanticMemorySearchResult:
    memory: StructuredMemory
    score: float

    @property
    def content(self) -> str:
        return self.memory.content

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory.normalized_id(),
            "category": self.memory.category,
            "entity_type": self.memory.entity_type,
            "key": self.memory.key,
            "value": self.memory.value,
            "content": self.memory.content,
            "confidence": self.memory.confidence,
            "score": self.score,
            "source_session_id": self.memory.source_session_id,
            "source_request_id": self.memory.source_request_id,
        }


class SemanticMemoryStore(Protocol):
    backend: str

    def search(
        self,
        *,
        user_id: str,
        query: str,
        filters: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> list[SemanticMemorySearchResult]:
        ...

    def upsert(self, memory: StructuredMemory) -> int:
        ...

    def upsert_from_turn(
        self,
        *,
        user_id: str,
        user_message: str,
        assistant_answer: str,
        session_id: str,
        request_id: str,
    ) -> int:
        ...

    def delete_user(self, user_id: str) -> int:
        ...

    def list_user(self, user_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        ...


class NoopSemanticMemoryStore:
    backend = "disabled"

    def search(
        self,
        *,
        user_id: str,
        query: str,
        filters: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> list[SemanticMemorySearchResult]:
        del user_id, query, filters, limit
        return []

    def upsert(self, memory: StructuredMemory) -> int:
        del memory
        return 0

    def upsert_from_turn(
        self,
        *,
        user_id: str,
        user_message: str,
        assistant_answer: str,
        session_id: str,
        request_id: str,
    ) -> int:
        del user_id, user_message, assistant_answer, session_id, request_id
        return 0

    def delete_user(self, user_id: str) -> int:
        del user_id
        return 0

    def list_user(self, user_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        del user_id, limit
        return []


class SQLiteSemanticMemoryStore:
    backend = "sqlite_semantic"

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensure_schema()

    def search(
        self,
        *,
        user_id: str,
        query: str,
        filters: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> list[SemanticMemorySearchResult]:
        rows = self._load_rows(user_id=user_id, filters=filters)
        scored: list[SemanticMemorySearchResult] = []
        for row in rows:
            memory = _structured_memory_from_row(row)
            score = _score_memory(query, memory)
            if score > 0:
                scored.append(SemanticMemorySearchResult(memory=memory, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def upsert(self, memory: StructuredMemory) -> int:
        memory_id = memory.normalized_id()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO semantic_user_memories(
                    memory_id, user_id, category, entity_type, memory_key,
                    value_json, content, source_session_id, source_request_id,
                    confidence, status, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(memory_id) DO UPDATE SET
                    category=excluded.category,
                    entity_type=excluded.entity_type,
                    memory_key=excluded.memory_key,
                    value_json=excluded.value_json,
                    content=excluded.content,
                    source_session_id=excluded.source_session_id,
                    source_request_id=excluded.source_request_id,
                    confidence=excluded.confidence,
                    status=excluded.status,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    memory_id,
                    memory.user_id,
                    memory.category,
                    memory.entity_type,
                    memory.key,
                    json.dumps(memory.value, ensure_ascii=False, sort_keys=True),
                    memory.content,
                    memory.source_session_id,
                    memory.source_request_id,
                    float(memory.confidence),
                    memory.status,
                ),
            )
        return 1 if cursor.rowcount else 0

    def upsert_from_turn(
        self,
        *,
        user_id: str,
        user_message: str,
        assistant_answer: str,
        session_id: str,
        request_id: str,
    ) -> int:
        saved = 0
        for category, content in extract_memory_candidates(user_message, assistant_answer):
            saved += self.upsert(
                StructuredMemory(
                    user_id=user_id,
                    category=category,
                    entity_type=_entity_type_for(category),
                    key=_key_for(category, content),
                    value=_value_for(category, content),
                    content=content,
                    source_session_id=session_id,
                    source_request_id=request_id,
                )
            )
        return saved

    def delete_user(self, user_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM semantic_user_memories WHERE user_id=?", (user_id,))
            return cursor.rowcount

    def list_user(self, user_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT memory_id, user_id, category, entity_type, memory_key,
                       value_json, content, source_session_id, source_request_id,
                       confidence, status
                FROM semantic_user_memories
                WHERE user_id=?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [
            SemanticMemorySearchResult(memory=_structured_memory_from_row(row), score=0.0).to_public_dict()
            for row in rows
        ]

    def _load_rows(self, *, user_id: str, filters: dict[str, Any] | None) -> list[sqlite3.Row]:
        categories = list((filters or {}).get("category") or [])
        with self._connect() as conn:
            if categories:
                placeholders = ",".join("?" for _ in categories)
                return list(
                    conn.execute(
                        f"""
                        SELECT memory_id, user_id, category, entity_type, memory_key,
                               value_json, content, source_session_id, source_request_id,
                               confidence, status
                        FROM semantic_user_memories
                        WHERE user_id=? AND status='active' AND category IN ({placeholders})
                        ORDER BY updated_at DESC
                        """,
                        (user_id, *categories),
                    )
                )
            return list(
                conn.execute(
                    """
                    SELECT memory_id, user_id, category, entity_type, memory_key,
                           value_json, content, source_session_id, source_request_id,
                           confidence, status
                    FROM semantic_user_memories
                    WHERE user_id=? AND status='active'
                    ORDER BY updated_at DESC
                    """,
                    (user_id,),
                )
            )

    def _connect(self) -> sqlite3.Connection:
        if self.db_path != ":memory:":
            Path(self.db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_user_memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_session_id TEXT NOT NULL DEFAULT '',
                    source_request_id TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.8,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )


@dataclass
class MilvusSemanticMemoryStore:
    uri: str
    collection_name: str = "customer_long_term_memories"
    dimension: int = 64
    backend: str = field(default="milvus", init=False)

    def __post_init__(self) -> None:
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:  # pragma: no cover - optional production dependency
            raise RuntimeError("pymilvus is required when SEMANTIC_MEMORY_BACKEND=milvus") from exc
        self._client = MilvusClient(uri=self.uri)
        self._ensure_collection()

    def search(
        self,
        *,
        user_id: str,
        query: str,
        filters: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> list[SemanticMemorySearchResult]:
        expr = _milvus_filter(user_id, filters)
        rows = self._client.search(
            collection_name=self.collection_name,
            data=[_hash_embedding(query, self.dimension)],
            filter=expr,
            limit=limit,
            output_fields=[
                "memory_id",
                "user_id",
                "category",
                "entity_type",
                "memory_key",
                "value_json",
                "content",
                "source_session_id",
                "source_request_id",
                "confidence",
                "status",
            ],
        )
        results: list[SemanticMemorySearchResult] = []
        for row in rows[0] if rows else []:
            entity = row.get("entity", row)
            memory = _structured_memory_from_mapping(entity)
            results.append(SemanticMemorySearchResult(memory=memory, score=float(row.get("distance", 0.0))))
        return results

    def upsert(self, memory: StructuredMemory) -> int:
        payload = _memory_payload(memory)
        payload["vector"] = _hash_embedding(memory.content, self.dimension)
        self._client.upsert(collection_name=self.collection_name, data=[payload])
        return 1

    def upsert_from_turn(
        self,
        *,
        user_id: str,
        user_message: str,
        assistant_answer: str,
        session_id: str,
        request_id: str,
    ) -> int:
        saved = 0
        for category, content in extract_memory_candidates(user_message, assistant_answer):
            saved += self.upsert(
                StructuredMemory(
                    user_id=user_id,
                    category=category,
                    entity_type=_entity_type_for(category),
                    key=_key_for(category, content),
                    value=_value_for(category, content),
                    content=content,
                    source_session_id=session_id,
                    source_request_id=request_id,
                )
            )
        return saved

    def delete_user(self, user_id: str) -> int:
        self._client.delete(collection_name=self.collection_name, filter=f'user_id == "{user_id}"')
        return 0

    def list_user(self, user_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.search(user_id=user_id, query="", limit=limit)
        return [row.to_public_dict() for row in rows]

    def _ensure_collection(self) -> None:
        if self._client.has_collection(self.collection_name):
            return
        self._client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension,
            metric_type="COSINE",
            primary_field_name="memory_id",
            vector_field_name="vector",
            auto_id=False,
        )


def _structured_memory_from_row(row: sqlite3.Row) -> StructuredMemory:
    return _structured_memory_from_mapping(dict(row))


def _structured_memory_from_mapping(payload: dict[str, Any]) -> StructuredMemory:
    return StructuredMemory(
        memory_id=str(payload.get("memory_id") or ""),
        user_id=str(payload.get("user_id") or ""),
        category=str(payload.get("category") or ""),
        entity_type=str(payload.get("entity_type") or ""),
        key=str(payload.get("memory_key") or payload.get("key") or ""),
        value=json.loads(str(payload.get("value_json") or "{}")),
        content=str(payload.get("content") or ""),
        source_session_id=str(payload.get("source_session_id") or ""),
        source_request_id=str(payload.get("source_request_id") or ""),
        confidence=float(payload.get("confidence") or 0.0),
        status=str(payload.get("status") or "active"),
    )


def _memory_payload(memory: StructuredMemory) -> dict[str, Any]:
    return {
        "memory_id": memory.normalized_id(),
        "user_id": memory.user_id,
        "category": memory.category,
        "entity_type": memory.entity_type,
        "memory_key": memory.key,
        "value_json": json.dumps(memory.value, ensure_ascii=False, sort_keys=True),
        "content": memory.content,
        "source_session_id": memory.source_session_id,
        "source_request_id": memory.source_request_id,
        "confidence": float(memory.confidence),
        "status": memory.status,
    }


def _score_memory(query: str, memory: StructuredMemory) -> float:
    normalized_query = query.casefold()
    normalized_content = memory.content.casefold()
    score = 0.0
    if memory.category == "delivery_preference" and any(
        keyword in normalized_query for keyword in ("配送", "物流", "快递", "发货", "送货", "shipping", "delivery")
    ):
        score += 5.0
    for token in ("顺丰", "京东", "京东物流", "配送", "物流", "快递", "退款", "airbuds", "homehub"):
        if token.casefold() in normalized_query and token.casefold() in normalized_content:
            score += 2.0
    for token in re.findall(r"[\w\u4e00-\u9fff]+", normalized_query):
        if len(token) >= 2 and token in normalized_content:
            score += 1.0
    return score * max(0.1, min(1.0, memory.confidence))


def _entity_type_for(category: str) -> str:
    if category == "profile":
        return "profile"
    if category == "complaint":
        return "risk_signal"
    return "preference"


def _key_for(category: str, content: str) -> str:
    if category == "delivery_preference":
        return "shipping_preference"
    if category == "profile" and "姓名" in content:
        return "name"
    return category


def _value_for(category: str, content: str) -> dict[str, Any]:
    return {"category": category, "text": content}


def _hash_embedding(text: str, dimension: int) -> list[float]:
    if not text:
        return [0.0] * dimension
    vector = [0.0] * dimension
    for token in re.findall(r"[\w\u4e00-\u9fff]+", text.casefold()):
        digest = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        vector[digest % dimension] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _milvus_filter(user_id: str, filters: dict[str, Any] | None) -> str:
    clauses = [f'user_id == "{user_id}"', 'status == "active"']
    categories = list((filters or {}).get("category") or [])
    if categories:
        quoted = ", ".join(f'"{category}"' for category in categories)
        clauses.append(f"category in [{quoted}]")
    return " and ".join(clauses)
