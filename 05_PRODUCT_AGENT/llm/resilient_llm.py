from __future__ import annotations


class ResilientLLM:
    async def ainvoke(self, messages: list[object]) -> str:
        return "offline_stub"
