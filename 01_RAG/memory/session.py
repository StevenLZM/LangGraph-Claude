"""
memory/session.py — 会话记忆管理
负责：
- 多会话隔离存储
- 历史消息窗口裁剪（防止 token 超限）
- 会话状态持久化（可选：JSON 文件）
"""
from __future__ import annotations
import json
import uuid
from pathlib import Path
from typing import Optional

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from config import rag_config


class SessionManager:
    """
    会话记忆管理器（单例）
    - 按 session_id 隔离存储对话历史
    - 超过最大轮数时自动裁剪旧消息
    """

    def __init__(self, max_history: int | None = None):
        self._store: dict[str, ChatMessageHistory] = {}
        self._max_history = max_history or rag_config.MAX_HISTORY_MESSAGES

    def get_or_create(self, session_id: str) -> ChatMessageHistory:
        """获取或创建会话历史"""
        if session_id not in self._store:
            self._store[session_id] = ChatMessageHistory()
        return self._store[session_id]

    def get_history(self, session_id: str) -> ChatMessageHistory:
        """向 RunnableWithMessageHistory 提供的回调函数"""
        return self.get_or_create(session_id)

    def add_exchange(
        self,
        session_id: str,
        human_msg: str,
        ai_msg: str,
    ) -> None:
        """
        添加一轮对话（Human + AI），并在超出上限时裁剪
        裁剪策略：保留最新的 N 轮（FIFO）
        """
        history = self.get_or_create(session_id)
        history.add_user_message(human_msg)
        history.add_ai_message(ai_msg)

        # 裁剪：每轮2条消息，超出后删最旧一轮
        messages = history.messages
        max_msgs = self._max_history * 2  # 每轮 = 1 human + 1 ai
        if len(messages) > max_msgs:
            # 保留最新的 max_msgs 条
            trimmed = messages[-max_msgs:]
            history.clear()
            for msg in trimmed:
                history.add_message(msg)

    def get_messages(self, session_id: str) -> list[BaseMessage]:
        """获取会话的所有消息"""
        return self.get_or_create(session_id).messages

    def clear_session(self, session_id: str) -> None:
        """清空指定会话历史"""
        if session_id in self._store:
            self._store[session_id].clear()

    def delete_session(self, session_id: str) -> None:
        """删除整个会话"""
        self._store.pop(session_id, None)

    def list_sessions(self) -> list[dict]:
        """列出所有会话及其消息数"""
        return [
            {
                "session_id": sid,
                "message_count": len(hist.messages),
                "turns": len(hist.messages) // 2,
            }
            for sid, hist in self._store.items()
        ]

    def get_formatted_history(self, session_id: str) -> str:
        """格式化历史消息为字符串（调试用）"""
        messages = self.get_messages(session_id)
        lines = []
        for msg in messages:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def new_session(self) -> str:
        """创建新会话，返回 session_id"""
        session_id = str(uuid.uuid4())[:8]
        self.get_or_create(session_id)
        return session_id

    def export_session(self, session_id: str) -> list[dict]:
        """导出会话历史为可序列化格式"""
        messages = self.get_messages(session_id)
        return [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in messages
        ]


# ── 全局单例 ──────────────────────────────────────────────────────
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
