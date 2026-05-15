from __future__ import annotations

import json
from typing import Any


JUDGE_PROMPT = """你是 RAG 答案评测员。请只根据题目、期望要点、检索上下文和答案评分。

输出严格 JSON：
{"score": 0-100, "reason": "一句话说明"}

题目：{question}
期望要点：{expected}
检索上下文：{context}
答案：{answer}
"""


def judge_generation(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    from rag.chain import _get_llm

    llm = _get_llm()
    prompt = JUDGE_PROMPT.format(
        question=case.get("question", ""),
        expected=", ".join(case.get("answer_keywords") or case.get("expected_keywords") or []),
        context=_context_preview(response),
        answer=response.get("answer", ""),
    )
    raw = llm.invoke(prompt)
    text = getattr(raw, "content", raw)
    parsed = _parse_json(str(text))
    if parsed is None:
        return {"judge_score": 0, "judge_reason": "LLM Judge returned invalid JSON"}
    return {
        "judge_score": int(parsed.get("score", 0)),
        "judge_reason": str(parsed.get("reason", "")),
    }


def _context_preview(response: dict[str, Any]) -> str:
    docs = response.get("sources") or []
    parts = []
    for doc in docs[:5]:
        content = getattr(doc, "page_content", None)
        if content is None and isinstance(doc, dict):
            content = doc.get("content") or doc.get("page_content")
        if content:
            parts.append(str(content)[:500])
    return "\n---\n".join(parts)


def _parse_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

