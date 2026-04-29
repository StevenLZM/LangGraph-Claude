from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EventType = Literal["thought", "tool_call", "tool_result", "plan", "step", "final", "error"]


@dataclass(slots=True)
class AgentEvent:
    type: EventType
    title: str
    content: str = ""
    tool: str | None = None
    tool_input: Any = None
    tool_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentRunResult:
    final_answer: str
    events: list[AgentEvent]
    raw_state: dict[str, Any] = field(default_factory=dict)
