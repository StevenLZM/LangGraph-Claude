"""FastAPI 请求 / 响应 schema —— ENGINEERING.md §8.2。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agents.schemas import ResearchPlan


class StartReq(BaseModel):
    research_query: str
    audience: str = "intermediate"
    depth: Literal["quick", "standard", "deep"] = "standard"


class StartResp(BaseModel):
    thread_id: str
    interrupt: dict | None = Field(default=None, description="若触发了 HITL，该字段非空；否则忽略")
    final_report: str | None = None
    report_path: str | None = None


class ResumeReq(BaseModel):
    plan: ResearchPlan


class TurnReq(BaseModel):
    research_query: str


class ThreadInfo(BaseModel):
    thread_id: str
    last_query: str = ""
    has_report: bool = False
