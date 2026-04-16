"""
rag/docstore.py — parent chunk 文档存储
使用 SQLite 持久化 parent 内容，供 parent-child 检索回填。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from langchain_core.documents import Document

from config import docstore_config


_docstore_instance: Optional["ParentDocStore"] = None


class ParentDocStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._ensure_schema()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS parent_chunks (
                    parent_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    doc_version TEXT NOT NULL,
                    source TEXT,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_parent_chunks_doc_id ON parent_chunks(doc_id)"
            )

    def upsert_parents(self, parents: Iterable[Document]) -> int:
        rows = []
        for parent in parents:
            metadata = dict(parent.metadata)
            rows.append(
                (
                    metadata["parent_id"],
                    metadata["doc_id"],
                    metadata["doc_version"],
                    metadata.get("source", "未知"),
                    parent.page_content,
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                )
            )

        if not rows:
            return 0

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO parent_chunks
                (parent_id, doc_id, doc_version, source, content, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def delete_document(self, doc_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM parent_chunks WHERE doc_id = ?", (doc_id,))
            return cursor.rowcount

    def get_parents(self, parent_ids: Iterable[str]) -> Dict[str, Document]:
        parent_ids = list(dict.fromkeys(parent_ids))
        if not parent_ids:
            return {}

        placeholders = ", ".join("?" for _ in parent_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT parent_id, content, metadata
                FROM parent_chunks
                WHERE parent_id IN ({placeholders})
                """,
                parent_ids,
            ).fetchall()

        result = {}
        for parent_id, content, metadata_json in rows:
            metadata = json.loads(metadata_json)
            result[parent_id] = Document(page_content=content, metadata=metadata)
        return result

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM parent_chunks").fetchone()
        return int(row[0]) if row else 0

    def list_documents(self) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT doc_id, doc_version, COUNT(*) AS parent_count, MIN(source) AS source
                FROM parent_chunks
                GROUP BY doc_id, doc_version
                ORDER BY MIN(source), doc_id
                """
            ).fetchall()

        return [
            {
                "doc_id": doc_id,
                "doc_version": doc_version,
                "parent_count": parent_count,
                "source": source,
            }
            for doc_id, doc_version, parent_count, source in rows
        ]


def get_parent_docstore(reset: bool = False) -> ParentDocStore:
    global _docstore_instance

    if _docstore_instance is not None and not reset:
        return _docstore_instance

    _docstore_instance = ParentDocStore(docstore_config.DB_PATH)
    return _docstore_instance
