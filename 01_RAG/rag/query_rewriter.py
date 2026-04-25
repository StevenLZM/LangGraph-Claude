"""
rag/query_rewriter.py — 带时间意图识别的查询改写器

输入：用户原始 query + chat_history
输出：
{
    "rewritten_query": str,          # 代词消解后的独立问题
    "time_intent": {
        "type": "latest"|"year"|"before"|"after"|"range"|"none",
        "field": "doc_date"|"upload_date",
        "range": {"gte": int, "lte": int} | None,  # YYYYMMDD
        "sort": "desc" | None,
    },
}
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import DEEPSEEK_BASE_URL, DASHSCOPE_BASE_URL, llm_config


# ────────────────────────────────────────────────────────────────
# 时间锚点：所有"今年/上个月/近 N 天"相对时间表达都以此为准
# 允许外部注入（测试时固定日期），默认取系统当天
# ────────────────────────────────────────────────────────────────
def _today() -> date:
    return date.today()


# ────────────────────────────────────────────────────────────────
# LLM 工厂：沿用 chain.py 的 provider 优先级
# ────────────────────────────────────────────────────────────────
def _get_rewrite_llm(temperature: float = 0.0):
    provider = llm_config.provider()
    model = llm_config.REWRITE_MODEL

    if provider == "deepseek":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=llm_config.DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            max_retries=2,
        )
    if provider == "dashscope":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=llm_config.DASHSCOPE_API_KEY,
            openai_api_base=DASHSCOPE_BASE_URL,
            max_retries=2,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=llm_config.ANTHROPIC_API_KEY,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=llm_config.OPENAI_API_KEY,
        )
    raise EnvironmentError("未配置可用的 LLM API Key")


# ────────────────────────────────────────────────────────────────
# Prompt：让 LLM 直接返回结构化 JSON
# ────────────────────────────────────────────────────────────────
REWRITE_PROMPT_TEMPLATE = """你是查询改写与时间意图识别助手。对下面的"最新问题"完成两件事：

1) 将代词/省略还原为独立完整的问题（不增不减原意图）。
2) 识别时间意图，按规则输出结构化 JSON。

【今日日期】{today}（锚点，用于解析"今年/上个月/近 N 天"等相对时间）

【字段选择 field】
- 用户说"最新提交 / 最近上传 / 最新归档" → field = "upload_date"
- 其他情况（包括默认）→ field = "doc_date"

【意图类型 type 判定规则】
- "最新/最近" + 无具体时间 → type = "latest"，range = null，sort = "desc"
- "YYYY 年" / "YYYY" 单独年份 → type = "year"，range = {{"gte": YYYY0101, "lte": YYYY1231}}
- "YYYY 年之前 / 以前 / 早于 YYYY" → type = "before"，range = {{"gte": 0, "lte": (YYYY-1)1231}}
- "YYYY 年之后 / 以后 / 以来 / 晚于 YYYY" → type = "after"，range = {{"gte": (YYYY+1)0101, "lte": 99991231}}
- "Q1/Q2/Q3/Q4" / "上个月" / "本月" / "近 N 天" / "上季度" / "YYYY 年 M 月" → type = "range"，按今日锚点算具体 [gte, lte]
- 没有任何时间表达 → type = "none"，range = null，sort = null

【排序 sort】
- type = "latest" → sort = "desc"
- 其他所有情况 → sort = null

【输出格式】严格的单行 JSON，不加任何注释、不加 markdown 围栏：
{{"rewritten_query": "...", "time_intent": {{"type": "...", "field": "...", "range": null | {{"gte": int, "lte": int}}, "sort": null | "desc"}}}}

【对话历史】
{chat_history}

【最新问题】{question}

JSON 输出："""


# ────────────────────────────────────────────────────────────────
# 正则兜底：LLM 失败或 JSON 非法时，用规则退化处理
# 只需覆盖最常见的表达，无需完整
# ────────────────────────────────────────────────────────────────
_LATEST_RE = re.compile(r"(最新|最近|newest|latest)")
_UPLOAD_FIELD_RE = re.compile(r"(上传|提交|归档|入库)")
_YEAR_RE = re.compile(r"(\d{4})\s*年")
_BEFORE_RE = re.compile(r"(\d{4})\s*年\s*(之前|以前|前)")
_AFTER_RE = re.compile(r"(\d{4})\s*年\s*(之后|以后|以来|后)")
_QUARTER_RE = re.compile(r"(\d{4})?\s*(?:年)?\s*Q([1-4])", re.IGNORECASE)
_LAST_N_DAYS_RE = re.compile(r"近\s*(\d+)\s*天")
_LAST_MONTH_RE = re.compile(r"上个?月|上月")
_THIS_MONTH_RE = re.compile(r"本月|这个月")


def _fallback_time_intent(query: str, today: date) -> dict[str, Any]:
    """纯规则的时间意图识别。覆盖不全时返回 type=none。"""
    field = "upload_date" if _UPLOAD_FIELD_RE.search(query) else "doc_date"

    # before 优先于 year（"2023 年之前"里同时命中两者）
    if m := _BEFORE_RE.search(query):
        y = int(m.group(1))
        return {
            "type": "before", "field": field,
            "range": {"gte": 0, "lte": (y - 1) * 10000 + 1231},
            "sort": None,
        }
    if m := _AFTER_RE.search(query):
        y = int(m.group(1))
        return {
            "type": "after", "field": field,
            "range": {"gte": (y + 1) * 10000 + 101, "lte": 99991231},
            "sort": None,
        }
    if m := _QUARTER_RE.search(query):
        y = int(m.group(1)) if m.group(1) else today.year
        q = int(m.group(2))
        q_start = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}[q]
        q_end = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}[q]
        return {
            "type": "range", "field": field,
            "range": {
                "gte": y * 10000 + q_start[0] * 100 + q_start[1],
                "lte": y * 10000 + q_end[0] * 100 + q_end[1],
            },
            "sort": None,
        }
    if m := _LAST_N_DAYS_RE.search(query):
        n = int(m.group(1))
        start = today - timedelta(days=n)
        return {
            "type": "range", "field": field,
            "range": {"gte": _d2int(start), "lte": _d2int(today)},
            "sort": None,
        }
    if _LAST_MONTH_RE.search(query):
        first_this = today.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        first_prev = last_prev.replace(day=1)
        return {
            "type": "range", "field": field,
            "range": {"gte": _d2int(first_prev), "lte": _d2int(last_prev)},
            "sort": None,
        }
    if _THIS_MONTH_RE.search(query):
        first = today.replace(day=1)
        return {
            "type": "range", "field": field,
            "range": {"gte": _d2int(first), "lte": _d2int(today)},
            "sort": None,
        }
    if m := _YEAR_RE.search(query):
        y = int(m.group(1))
        return {
            "type": "year", "field": field,
            "range": {"gte": y * 10000 + 101, "lte": y * 10000 + 1231},
            "sort": None,
        }
    if _LATEST_RE.search(query):
        return {"type": "latest", "field": field, "range": None, "sort": "desc"}
    return {"type": "none", "field": field, "range": None, "sort": None}


def _d2int(d: date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


# ────────────────────────────────────────────────────────────────
# 主 API
# ────────────────────────────────────────────────────────────────
def rewrite_query(
    question: str,
    chat_history: str = "",
    today: date | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    """
    将 question 改写为独立问题，并识别时间意图。

    use_llm=False 时只走正则兜底，便于单元测试与零成本调试。
    """
    anchor = today or _today()
    question = question.strip()

    # 规则兜底结果（无论是否调 LLM 都先算一次，作为回退）
    fallback_intent = _fallback_time_intent(question, anchor)
    fallback_result = {"rewritten_query": question, "time_intent": fallback_intent}

    if not use_llm:
        return fallback_result

    try:
        llm = _get_rewrite_llm()
        prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT_TEMPLATE)
        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke({
            "question": question,
            "chat_history": chat_history or "（无历史）",
            "today": anchor.strftime("%Y-%m-%d"),
        })
        parsed = _parse_json(raw)
        if parsed is None:
            return fallback_result
        return _normalize(parsed, fallback_result)
    except Exception as e:
        print(f"[query_rewriter] LLM 调用失败，回退到规则: {e}")
        return fallback_result


def _parse_json(raw: str) -> dict | None:
    """从 LLM 输出中解析 JSON，容忍 markdown 围栏。"""
    s = raw.strip()
    # 去掉可能的 ```json ... ``` 围栏
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    # 从第一个 { 到最后一个 }
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return None


def _normalize(parsed: dict, fallback: dict) -> dict:
    """校验 LLM 输出，缺字段时用 fallback 补齐。"""
    rewritten = parsed.get("rewritten_query") or fallback["rewritten_query"]
    ti = parsed.get("time_intent") or {}
    fti = fallback["time_intent"]

    valid_types = {"latest", "year", "before", "after", "range", "none"}
    valid_fields = {"doc_date", "upload_date"}

    t = ti.get("type") if ti.get("type") in valid_types else fti["type"]
    f = ti.get("field") if ti.get("field") in valid_fields else fti["field"]
    r = ti.get("range")
    if r is not None and not (isinstance(r, dict) and "gte" in r and "lte" in r):
        r = fti["range"]
    s = ti.get("sort") if ti.get("sort") in (None, "desc") else fti["sort"]

    if t == "none":
        r, s = None, None
    elif t == "latest":
        r, s = None, "desc"

    return {
        "rewritten_query": rewritten,
        "time_intent": {"type": t, "field": f, "range": r, "sort": s},
    }
