"""Pydantic 结构化输出模型 —— 完全对齐 ENGINEERING.md §6.2。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    id: str
    question: str
    recommended_sources: list[Literal["web", "academic", "code", "kb"]]
    status: Literal["pending", "researching", "done"] = "pending"


class ResearchPlan(BaseModel):
    sub_questions: list[SubQuestion]
    estimated_depth: Literal["quick", "standard", "deep"] = "standard"


class Evidence(BaseModel):
    sub_question_id: str
    source_type: Literal["web", "academic", "code", "kb"]
    source_url: str
    snippet: str
    relevance_score: float = 0.0
    fetched_at: str = ""

    # 定义数据结构的序列化输出，继承BaseModel里本身也有，不写也行，写了就覆盖
    def __repr__(self):
        return f"Evidence(sub_question_id={self.sub_question_id}, source_type={self.source_type}, url={self.source_url}, score={self.relevance_score}, content={self.fetched_at})"


class ReflectionResult(BaseModel):
    coverage_by_subq: dict[str, int] = Field(default_factory=dict)
    missing_aspects: list[str] = Field(default_factory=list)
    next_action: Literal["sufficient", "need_more_research", "force_complete"] = "sufficient"
    additional_queries: list[str] | None = None


class Citation(BaseModel):
    idx: int
    source_url: str
    title: str | None = None
