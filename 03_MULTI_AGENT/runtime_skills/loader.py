"""Load runtime skill prompts from Markdown skill files."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SKILLS_ROOT = _PROJECT_ROOT / "skills"


@lru_cache(maxsize=None)
def load_skill_prompt(skill_name: str) -> str:
    """Load a runtime skill's Markdown body without YAML frontmatter."""
    path = _SKILLS_ROOT / skill_name / "SKILL.md"
    text = path.read_text(encoding="utf-8")
    return _strip_frontmatter(text).strip()


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end == -1:
        return text
    return text[end + len("\n---\n") :]
