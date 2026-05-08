from __future__ import annotations

import re
import sqlite3
from pathlib import Path

DELIVERY_KEYWORDS = ("配送", "物流", "快递", "发货", "送货", "承运", "shipping", "delivery", "courier")


class UserMemoryManager:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensure_schema()

    def load_memories(self, user_id: str, current_query: str, *, limit: int = 5) -> list[str]:
        rows = self._load_rows(user_id)
        if not rows:
            return []
        query = current_query.casefold()
        if self._is_recall_query(query):
            return [row["content"] for row in rows[:limit]]

        scored = []
        for row in rows:
            content = row["content"]
            score = self._score(query, content.casefold(), row["category"])
            if score > 0:
                scored.append((score, content))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [content for _, content in scored[:limit]]

    def save_from_turn(self, user_id: str, user_message: str, assistant_answer: str) -> int:
        memories = self._extract_memories(user_message, assistant_answer)
        saved = 0
        with self._connect() as conn:
            for category, content in memories:
                if category == "delivery_preference":
                    self._delete_delivery_preferences(conn, user_id)
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO user_memories(user_id, category, content, created_at)
                    VALUES(?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (user_id, category, content),
                )
                saved += cursor.rowcount
        return saved

    def delete_memories(self, user_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM user_memories WHERE user_id = ?", (user_id,))
            return cursor.rowcount

    def list_memories(self, user_id: str, *, limit: int = 100) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT category, content, created_at
                FROM user_memories
                WHERE user_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [
            {
                "category": row["category"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    async def aload_memories(self, user_id: str, current_query: str) -> list[str]:
        return self.load_memories(user_id, current_query)

    async def adelete_memories(self, user_id: str) -> int:
        return self.delete_memories(user_id)

    def _extract_memories(self, user_message: str, assistant_answer: str) -> list[tuple[str, str]]:
        del assistant_answer
        text = " ".join(user_message.split())
        normalized = text.casefold()
        memories: list[tuple[str, str]] = []
        if self._is_recall_query(normalized) or "?" in text or "？" in text:
            return memories
        if any(keyword in normalized for keyword in ("喜欢", "偏好", "优先", "prefer", "preference")):
            category = "delivery_preference" if self._is_delivery_related(normalized) else "preference"
            memories.append((category, f"用户偏好：{text}"))
        if any(keyword in normalized for keyword in ("投诉", "不满", "差评", "complaint")):
            memories.append(("complaint", f"用户投诉记录：{text}"))
        name_match = re.search(r"我叫([\w\u4e00-\u9fff]{2,12})", text)
        if name_match:
            memories.append(("profile", f"用户姓名：{name_match.group(1)}"))
        return memories

    def _load_rows(self, user_id: str) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return list(
                conn.execute(
                    """
                    SELECT category, content, created_at
                    FROM user_memories
                    WHERE user_id = ?
                    ORDER BY created_at DESC, id DESC
                    """,
                    (user_id,),
                )
            )

    @staticmethod
    def _is_recall_query(query: str) -> bool:
        return any(keyword in query for keyword in ("记得", "偏好", "历史", "之前", "remember", "preference"))

    @staticmethod
    def _score(query: str, content: str, category: str) -> int:
        score = 0
        if category == "delivery_preference" and UserMemoryManager._is_delivery_related(query):
            score += 5
        for token in ("顺丰", "京东", "京东物流", "配送", "物流", "快递", "发货", "送货", "投诉", "退款", "airbuds", "homehub"):
            if token.casefold() in query and token.casefold() in content:
                score += 2
        for token in re.findall(r"[a-zA-Z0-9_]+", query):
            if token.casefold() in content:
                score += 1
        return score

    @staticmethod
    def _is_delivery_related(text: str) -> bool:
        return any(keyword.casefold() in text for keyword in DELIVERY_KEYWORDS)

    @staticmethod
    def _delete_delivery_preferences(conn: sqlite3.Connection, user_id: str) -> int:
        cursor = conn.execute(
            """
            DELETE FROM user_memories
            WHERE user_id = ?
              AND (
                category = 'delivery_preference'
                OR (
                  category = 'preference'
                  AND (
                    content LIKE '%配送%'
                    OR content LIKE '%物流%'
                    OR content LIKE '%快递%'
                    OR content LIKE '%发货%'
                    OR content LIKE '%送货%'
                    OR content LIKE '%承运%'
                    OR lower(content) LIKE '%shipping%'
                    OR lower(content) LIKE '%delivery%'
                    OR lower(content) LIKE '%courier%'
                  )
                )
              )
            """,
            (user_id,),
        )
        return cursor.rowcount

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
                CREATE TABLE IF NOT EXISTS user_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, content)
                )
                """
            )
