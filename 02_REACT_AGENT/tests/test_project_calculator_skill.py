from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = ROOT / ".codex" / "skills" / "calculator"
SCRIPT = SKILL_DIR / "scripts" / "calculate.py"


def _run_calculator(expression: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), expression],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_project_calculator_skill_has_required_files():
    assert (SKILL_DIR / "SKILL.md").is_file()
    assert (SKILL_DIR / "agents" / "openai.yaml").is_file()
    assert SCRIPT.is_file()


def test_project_calculator_skill_frontmatter_is_valid():
    content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    _, frontmatter, _ = content.split("---", 2)

    assert "name: calculator" in frontmatter
    assert "description: Use when" in frontmatter
    assert "arithmetic" in frontmatter.lower()


def test_project_calculator_script_evaluates_numeric_expression():
    result = _run_calculator("1234 * 5678")

    assert result.returncode == 0
    assert result.stdout.strip() == "计算结果: 1234 * 5678 = 7006652"


def test_project_calculator_script_supports_safe_math_functions():
    result = _run_calculator("sqrt(144)")

    assert result.returncode == 0
    assert result.stdout.strip() == "计算结果: sqrt(144) = 12.0"


def test_project_calculator_script_rejects_unsafe_expression():
    result = _run_calculator("__import__('os').system('ls')")

    assert result.returncode == 1
    assert "计算错误" in result.stdout
