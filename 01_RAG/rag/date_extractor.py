"""
rag/date_extractor.py — 文档内容日期抽取器

输入：parent chunk 文本（或任意文本）
输出：DateExtractionResult{min: int, max: int, found: bool}（YYYYMMDD 整数）

策略：
1. 正则兜底：覆盖常见格式，零成本
2. LLM 回退：正则空命中且 LLM_FALLBACK=True 时调用，约束 JSON 输出
3. SQLite 缓存：按 doc_id + chunk_hash 缓存结果，避免重复 LLM 调用
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import BASE_DIR, DASHSCOPE_BASE_URL, llm_config, rag_config


# ────────────────────────────────────────────────────────────────
# 数据结构
# ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DateExtractionResult:
    min: int  # YYYYMMDD，无日期=0
    max: int  # YYYYMMDD，无日期=0
    found: bool

    @classmethod
    def empty(cls) -> "DateExtractionResult":
        return cls(min=0, max=0, found=False)

    @classmethod
    def from_dates(cls, dates: list[int]) -> "DateExtractionResult":
        if not dates:
            return cls.empty()
        return cls(min=min(dates), max=max(dates), found=True)


# ────────────────────────────────────────────────────────────────
# 正则模式
# ────────────────────────────────────────────────────────────────
# YYYY年MM月DD日 / YYYY年MM月（缺日补 01）
_RE_CN_FULL = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_RE_CN_YM = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月(?!\s*\d)")
# YYYY-MM-DD / YYYY/MM/DD / YYYY.MM.DD
_RE_NUM = re.compile(r"(\d{4})[\-/.](\d{1,2})[\-/.](\d{1,2})")


def _to_int(y: int, m: int, d: int) -> Optional[int]:
    """简单合法性校验：年 1900~2099，月 1~12，日 1~31。"""
    if not (1900 <= y <= 2099 and 1 <= m <= 12 and 1 <= d <= 31):
        return None
    return y * 10000 + m * 100 + d


def _regex_extract(text: str) -> list[int]:
    """返回所有抽到的 YYYYMMDD 整数（去重，排序前）。"""
    found: set[int] = set()

    for m in _RE_CN_FULL.finditer(text):
        d = _to_int(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            found.add(d)

    for m in _RE_NUM.finditer(text):
        d = _to_int(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            found.add(d)

    # YM 在 full 之后跑，避免重复匹配
    for m in _RE_CN_YM.finditer(text):
        d = _to_int(int(m.group(1)), int(m.group(2)), 1)
        if d:
            found.add(d)

    return sorted(found)


# ────────────────────────────────────────────────────────────────
# LLM 回退
# ────────────────────────────────────────────────────────────────
_LLM_PROMPT = """从下列文本中抽取所有出现的具体日期（开票日期、消费日期、签订日期、报销日期等）。

【规则】
- 只抽取明确的年月日，年份必须是 1900-2099
- 缺日的写为该月 01 日；缺月的不要抽取
- 同一日期出现多次只算一次
- 没有日期就返回空数组
- 只输出 JSON，不加任何注释、不加 markdown 围栏

【输出格式】
{{"dates": ["YYYY-MM-DD", "YYYY-MM-DD"]}}

【文本】
{text}

JSON 输出："""


def _llm_extract(text: str) -> list[int]:
    """调 LLM 抽日期，失败/格式错时返回空列表。"""
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        provider = llm_config.provider()
        model = llm_config.REWRITE_MODEL  # 复用轻量模型
        if provider == "dashscope":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model, temperature=0.0,
                openai_api_key=llm_config.DASHSCOPE_API_KEY,
                openai_api_base=DASHSCOPE_BASE_URL, max_retries=2,
            )
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model=model, temperature=0.0, api_key=llm_config.ANTHROPIC_API_KEY)
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=model, temperature=0.0, api_key=llm_config.OPENAI_API_KEY)
        else:
            return []

        chain = ChatPromptTemplate.from_template(_LLM_PROMPT) | llm | StrOutputParser()
        # 截断输入避免超长（parent chunk 通常 ~900 token，足够）
        raw = chain.invoke({"text": text[:3000]})
        return _parse_llm_dates(raw)
    except Exception as e:
        print(f"[date_extractor] LLM 抽取失败: {e}")
        return []


def _parse_llm_dates(raw: str) -> list[int]:
    """从 LLM 输出解析日期数组，容忍 markdown 围栏。"""
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return []

    out: list[int] = []
    for s in data.get("dates", []):
        m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", str(s).strip())
        if m:
            d = _to_int(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if d:
                out.append(d)
    return out


# ────────────────────────────────────────────────────────────────
# SQLite 缓存
# ────────────────────────────────────────────────────────────────
_cache_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn

    cache_path = Path(rag_config.DATE_CACHE_PATH)
    if not cache_path.is_absolute():
        cache_path = BASE_DIR / cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    _conn = sqlite3.connect(str(cache_path), check_same_thread=False)
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS date_cache (
            cache_key TEXT PRIMARY KEY,
            min_date INTEGER NOT NULL,
            max_date INTEGER NOT NULL,
            found INTEGER NOT NULL
        )
    """)
    _conn.commit()
    return _conn


def _cache_key(doc_id: str, text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{doc_id}:{h}"


def _cache_get(key: str) -> Optional[DateExtractionResult]:
    with _cache_lock:
        cur = _get_conn().execute(
            "SELECT min_date, max_date, found FROM date_cache WHERE cache_key=?", (key,)
        )
        row = cur.fetchone()
    if row is None:
        return None
    return DateExtractionResult(min=row[0], max=row[1], found=bool(row[2]))


def _cache_set(key: str, result: DateExtractionResult) -> None:
    with _cache_lock:
        conn = _get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO date_cache (cache_key, min_date, max_date, found) VALUES (?, ?, ?, ?)",
            (key, result.min, result.max, int(result.found)),
        )
        conn.commit()


# ────────────────────────────────────────────────────────────────
# 主 API
# ────────────────────────────────────────────────────────────────
def extract_dates(
    text: str,
    doc_id: str = "anonymous",
    use_llm_fallback: Optional[bool] = None,
    use_cache: bool = True,
) -> DateExtractionResult:
    """
    抽取文本中的业务日期，返回区间 [min, max]。

    Args:
        text: parent chunk 或任意文本
        doc_id: 用于缓存 key
        use_llm_fallback: None=取 config 默认；True=正则空命中时调 LLM
        use_cache: 是否查询/写入 SQLite 缓存
    """
    if not text or not text.strip():
        return DateExtractionResult.empty()

    if not rag_config.DATE_EXTRACTION_ENABLED:
        return DateExtractionResult.empty()

    key = _cache_key(doc_id, text) if use_cache else None
    if use_cache and key:
        cached = _cache_get(key)
        if cached is not None:
            return cached

    # 1. 正则
    dates = _regex_extract(text)

    # 2. LLM 兜底
    use_llm = (
        use_llm_fallback
        if use_llm_fallback is not None
        else rag_config.DATE_EXTRACTION_LLM_FALLBACK
    )
    if not dates and use_llm:
        dates = _llm_extract(text)

    result = DateExtractionResult.from_dates(dates)
    if use_cache and key:
        _cache_set(key, result)
    return result
