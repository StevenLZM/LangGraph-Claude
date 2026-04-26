from __future__ import annotations

from scripts.run_local import _prompt_plan_decision


def test_prompt_plan_decision_accepts_plan_when_stdin_is_missing():
    proposed = {
        "plan": {
            "sub_questions": [
                {
                    "id": "sq1",
                    "question": "分析框架",
                    "recommended_sources": ["web"],
                    "status": "pending",
                }
            ],
            "estimated_depth": "quick",
        }
    }

    def missing_stdin(_prompt: str) -> str:
        raise EOFError

    decision = _prompt_plan_decision(proposed, input_fn=missing_stdin)

    assert decision == {"plan": proposed["plan"], "action": "accept"}
