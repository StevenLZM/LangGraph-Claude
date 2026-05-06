from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class CustomerServiceState(TypedDict, total=False):
    session_id: str
    user_id: str

    messages: Annotated[list[BaseMessage], add_messages]
    window_size: int
    total_turns: int

    user_profile: dict[str, Any]
    user_memories: list[str]
    memory_summary: str
    order_context: dict[str, Any] | None

    needs_human_transfer: bool
    transfer_reason: str

    token_used: int
    response_time_ms: int
    quality_score: int | None
    tool_name: str

    _started_at: float
