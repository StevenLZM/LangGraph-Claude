from __future__ import annotations


class UserMemoryManager:
    async def load_memories(self, user_id: str, current_query: str) -> list[str]:
        return []

    async def delete_memories(self, user_id: str) -> None:
        return None
