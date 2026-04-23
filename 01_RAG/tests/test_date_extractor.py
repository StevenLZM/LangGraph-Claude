"""
tests/test_date_extractor.py — date_extractor 单元测试

只测正则路径（零成本）。LLM 路径在端到端验证中覆盖。
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.date_extractor import extract_dates


# 关闭缓存与 LLM，纯规则
def _extract(text: str):
    return extract_dates(text, doc_id="test", use_llm_fallback=False, use_cache=False)


class TestRegex:
    def test_no_date(self):
        r = _extract("这是一段没有日期的文本")
        assert r.found is False
        assert r.min == 0 and r.max == 0

    def test_chinese_full(self):
        r = _extract("开票日期：2024年3月15日")
        assert r.found
        assert r.min == 20240315 and r.max == 20240315

    def test_numeric_dash(self):
        r = _extract("签订时间 2025-07-01")
        assert r.min == 20250701

    def test_numeric_slash(self):
        r = _extract("日期 2024/12/31")
        assert r.min == 20241231

    def test_numeric_dot(self):
        r = _extract("Date: 2023.06.15")
        assert r.min == 20230615

    def test_chinese_year_month_only(self):
        r = _extract("报销周期为2024年8月")
        assert r.found
        assert r.min == 20240801

    def test_multiple_dates_min_max(self):
        text = "开票日期2024年1月10日，消费日期2024年3月25日，提交日期2024年4月1日"
        r = _extract(text)
        assert r.found
        assert r.min == 20240110
        assert r.max == 20240401

    def test_invalid_date_skipped(self):
        # 月份 13 应被丢弃
        r = _extract("无效日期 2024-13-01，有效 2024-05-15")
        assert r.found
        assert r.min == 20240515 and r.max == 20240515

    def test_dedup(self):
        # 重复日期只算一次
        r = _extract("2024年3月15日 又见 2024-3-15")
        assert r.found
        assert r.min == r.max == 20240315

    def test_year_out_of_range(self):
        # 1899 与 2100 都应丢弃
        r = _extract("1899-01-01 和 2100-12-31 都不要")
        assert not r.found

    def test_mixed_formats(self):
        text = "首次2023年6月1日，复审2024/2/14，结案2024.10.20"
        r = _extract(text)
        assert r.found
        assert r.min == 20230601
        assert r.max == 20241020


if __name__ == "__main__":
    samples = [
        "无日期文本",
        "开票日期：2024年3月15日，金额 500 元",
        "首次2023年6月1日，复审2024/2/14，结案2024.10.20",
        "报销周期为2024年8月",
    ]
    for s in samples:
        r = _extract(s)
        print(f"{s[:40]:40s} → found={r.found} min={r.min} max={r.max}")
