"""报告归档 —— ENGINEERING.md §9.2。"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from config.settings import settings


def _slug(text: str, n: int = 40) -> str:
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"[^\w\u4e00-\u9fff\-]", "", text)
    return text[:n] or "untitled"


def report_path(query: str, thread_id: str) -> Path:
    base = Path(settings.reports_dir)
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return base / f"{ts}_{_slug(query)}_{thread_id}.md"


def save(query: str, thread_id: str, content: str) -> str:
    p = report_path(query, thread_id)
    p.write_text(content, encoding="utf-8")
    return str(p)


def list_reports(limit: int = 20) -> list[dict]:
    base = Path(settings.reports_dir)
    if not base.exists():
        return []
    files = sorted(base.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in files[:limit]:
        tid = _parse_tid(p.name)
        out.append(
            {
                "path": str(p),
                "name": p.name,
                "thread_id": tid,
                "mtime": p.stat().st_mtime,
                "size": p.stat().st_size,
            }
        )
    return out


def read_report(path_or_tid: str) -> str:
    # 允许按 path 或 thread_id 读
    p = Path(path_or_tid)
    if p.exists():
        return p.read_text(encoding="utf-8")
    # 尝试 thread_id 匹配
    base = Path(settings.reports_dir)
    if base.exists():
        for f in base.glob(f"*_{path_or_tid}.md"):
            return f.read_text(encoding="utf-8")
    raise FileNotFoundError(path_or_tid)


def find_by_thread(thread_id: str) -> str | None:
    base = Path(settings.reports_dir)
    if not base.exists():
        return None
    for f in sorted(base.glob(f"*_{thread_id}.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        return str(f)
    return None


def _parse_tid(name: str) -> str:
    # 文件名格式：{ts}_{slug}_{tid}.md → 取最后一段（去 .md）
    stem = name[:-3] if name.endswith(".md") else name
    parts = stem.rsplit("_", 1)
    return parts[1] if len(parts) == 2 else ""
