from __future__ import annotations

from langchain_core.messages import BaseMessage, SystemMessage


class ContextWindowManager:
    def __init__(self, *, max_messages: int = 16, max_tokens: int = 3200) -> None:
        self.max_messages = max_messages
        self.max_tokens = max_tokens

    def count_tokens(self, messages: list[BaseMessage]) -> int:
        return sum(self._estimate_tokens(str(message.content)) for message in messages)

    def trim(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        if len(messages) <= self.max_messages and self.count_tokens(messages) <= self.max_tokens:
            return messages

        recent_messages = list(messages[-self.max_messages :])
        old_messages = list(messages[: -self.max_messages])
        if not old_messages:
            return self._fit_to_budget(recent_messages)

        summary = self._summarize(old_messages)
        trimmed = [SystemMessage(content=summary, additional_kwargs={"type": "summary"})]
        trimmed.extend(recent_messages)
        return self._fit_to_budget(trimmed)

    def _fit_to_budget(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        fitted = list(messages)
        while len(fitted) > 1 and self.count_tokens(fitted) > self.max_tokens:
            # 保留摘要和最新消息，优先丢弃最早的普通消息。
            remove_index = 1 if isinstance(fitted[0], SystemMessage) else 0
            del fitted[remove_index]
        return fitted

    def _summarize(self, messages: list[BaseMessage]) -> str:
        first = self._shorten(str(messages[0].content)) if messages else ""
        last = self._shorten(str(messages[-1].content)) if messages else ""
        return f"[早期对话摘要] 已压缩 {len(messages)} 条早期消息。起始：{first}；最近：{last}"

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 2)

    @staticmethod
    def _shorten(text: str, limit: int = 48) -> str:
        text = " ".join(text.split())
        return text if len(text) <= limit else f"{text[:limit]}..."
