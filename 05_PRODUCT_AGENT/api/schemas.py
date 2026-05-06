from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class ChatResponse(BaseModel):
    session_id: str
    user_id: str
    answer: str
    needs_human_transfer: bool
    transfer_reason: str
    order_context: dict[str, Any] | None
    token_used: int
    response_time_ms: int
    quality_score: int | None
    user_memories: list[str] = []
    memory_summary: str = ""


class DeleteMemoriesResponse(BaseModel):
    user_id: str
    deleted: int
