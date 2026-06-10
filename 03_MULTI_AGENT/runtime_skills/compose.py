"""Small helpers for composing reusable runtime skill prompts."""
from __future__ import annotations


def compose_system_prompt(base: str, *skills: str) -> str:
    """Append skill contracts to an agent system prompt."""
    blocks = [base.strip(), *(skill.strip() for skill in skills if skill.strip())]
    return "\n\n".join(blocks)
