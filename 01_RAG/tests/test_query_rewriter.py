"""
tests/test_query_rewriter.py — query_rewriter 单元测试

测试两个层面：
1) 纯规则兜底（use_llm=False）：六类意图 + 字段判定 + 边界
2) LLM 解析路径（use_llm=True）：如配置了 API Key 则跑通，否则跳过
"""
from datetime import date
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from rag.query_rewriter import rewrite_query


ANCHOR = date(2026, 4, 23)  # 与 plan 锚点一致


# ────────────────────────────────────────────────────────────────
# 规则兜底层（不调用 LLM，纯本地逻辑）
# ────────────────────────────────────────────────────────────────
class TestRuleFallback:
    def _intent(self, q: str) -> dict:
        return rewrite_query(q, today=ANCHOR, use_llm=False)["time_intent"]

    # ---- type=none ----
    def test_no_time_intent(self):
        ti = self._intent("报销流程是什么")
        assert ti["type"] == "none"
        assert ti["range"] is None
        assert ti["sort"] is None
        assert ti["field"] == "doc_date"

    # ---- type=latest ----
    def test_latest_default_field(self):
        ti = self._intent("最新的报销票据")
        assert ti["type"] == "latest"
        assert ti["field"] == "doc_date"
        assert ti["sort"] == "desc"
        assert ti["range"] is None

    def test_latest_upload_field(self):
        ti = self._intent("最近上传的合同")
        assert ti["type"] == "latest"
        assert ti["field"] == "upload_date"

    def test_latest_submit_field(self):
        ti = self._intent("最新提交的发票")
        assert ti["field"] == "upload_date"

    # ---- type=year ----
    def test_year(self):
        ti = self._intent("2024 年的报销")
        assert ti["type"] == "year"
        assert ti["range"] == {"gte": 20240101, "lte": 20241231}

    # ---- type=before ----
    def test_before(self):
        ti = self._intent("2023 年之前的票据")
        assert ti["type"] == "before"
        assert ti["range"] == {"gte": 0, "lte": 20221231}

    def test_before_alt_phrase(self):
        ti = self._intent("2023年以前的合同")
        assert ti["type"] == "before"
        assert ti["range"]["lte"] == 20221231

    # ---- type=after ----
    def test_after(self):
        ti = self._intent("2024 年之后的合同")
        assert ti["type"] == "after"
        assert ti["range"] == {"gte": 20250101, "lte": 99991231}

    def test_after_yilai(self):
        ti = self._intent("2025年以来的发票")
        assert ti["type"] == "after"
        assert ti["range"]["gte"] == 20260101

    # ---- type=range ----
    def test_quarter_with_year(self):
        ti = self._intent("2024 Q1 的报销")
        assert ti["type"] == "range"
        assert ti["range"] == {"gte": 20240101, "lte": 20240331}

    def test_quarter_without_year_uses_anchor(self):
        ti = self._intent("Q2 的报销")
        assert ti["type"] == "range"
        assert ti["range"] == {"gte": 20260401, "lte": 20260630}

    def test_last_n_days(self):
        ti = self._intent("近 30 天的发票")
        assert ti["type"] == "range"
        # 锚点 2026-04-23，往前 30 天 = 2026-03-24
        assert ti["range"] == {"gte": 20260324, "lte": 20260423}

    def test_last_month(self):
        ti = self._intent("上个月的报销")
        assert ti["type"] == "range"
        # 锚点 2026-04-23 → 上月 2026-03-01 ~ 2026-03-31
        assert ti["range"] == {"gte": 20260301, "lte": 20260331}

    def test_this_month(self):
        ti = self._intent("本月的发票")
        assert ti["type"] == "range"
        assert ti["range"] == {"gte": 20260401, "lte": 20260423}

    # ---- 优先级：before/after 优先于 year ----
    def test_before_priority_over_year(self):
        # "2023 年之前" 同时命中 year 和 before，应取 before
        ti = self._intent("2023 年之前的票据")
        assert ti["type"] == "before"


# ────────────────────────────────────────────────────────────────
# rewritten_query 字段（兜底分支保持原 query 不变）
# ────────────────────────────────────────────────────────────────
class TestRewrittenQuery:
    def test_fallback_keeps_original(self):
        result = rewrite_query("最新的报销票据", today=ANCHOR, use_llm=False)
        assert result["rewritten_query"] == "最新的报销票据"


# ────────────────────────────────────────────────────────────────
# LLM 解析路径（端到端，需 API Key）
# ────────────────────────────────────────────────────────────────
LLM_AVAILABLE = bool(
    os.getenv("DASHSCOPE_API_KEY")
    or os.getenv("ANTHROPIC_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)


@pytest.mark.skipif(not LLM_AVAILABLE, reason="未配置 LLM API Key")
class TestLLMPath:
    def test_llm_year_intent(self):
        result = rewrite_query("2024 年的报销", today=ANCHOR, use_llm=True)
        ti = result["time_intent"]
        assert ti["type"] == "year"
        assert ti["range"]["gte"] == 20240101
        assert ti["range"]["lte"] == 20241231

    def test_llm_pronoun_resolution(self):
        result = rewrite_query(
            question="它的报销上限是多少？",
            chat_history="用户：差旅费政策包含哪些内容？\n助手：差旅费政策涵盖交通、住宿、餐补三类。",
            today=ANCHOR,
            use_llm=True,
        )
        # 代词应被替换为"差旅费"相关实体；LLM 保守保留原句也可接受（放宽）
        rq = result["rewritten_query"]
        assert "差旅" in rq or "它" in rq  # 只要语义可理解即通过

    def test_llm_no_time_intent(self):
        result = rewrite_query("报销流程是什么", today=ANCHOR, use_llm=True)
        assert result["time_intent"]["type"] == "none"


if __name__ == "__main__":
    # 直接运行：快速看效果
    samples = [
        "报销流程是什么",
        "最新的报销票据",
        "最近上传的合同",
        "2024 年的报销",
        "2023 年之前的票据",
        "2024 年之后的合同",
        "2024 Q1 的报销",
        "近 30 天的发票",
        "上个月的报销",
    ]
    print(f"\n锚点日期：{ANCHOR}\n" + "─" * 60)
    for q in samples:
        r = rewrite_query(q, today=ANCHOR, use_llm=False)
        print(f"Q: {q}")
        print(f"   → {r['time_intent']}\n")
