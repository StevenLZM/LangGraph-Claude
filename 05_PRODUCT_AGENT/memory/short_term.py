from __future__ import annotations

from langchain_core.messages import BaseMessage


class ContextWindowManager:
    def trim(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        return messages
