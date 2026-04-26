# -*- coding: utf-8 -*-
from __future__ import annotations
import re
import hashlib
from pathlib import Path
from typing import List, Tuple
from langchain_core.documents import Document


def _compute_doc_id(file_path: str) -> str:
    return hashlib.md5(Path(file_path).name.encode()).hexdigest()[:12]


def _clean_text(text: str) -> str:
    text = re.sub(r"^\s*[-\u2013\u2014]?\s*\d+\s*[-\u2013\u2014]?\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.replace("\u3000", " ").replace("\t", " ")
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _merge_broken_sentences(
    pages: List[Tuple[int, str, List[dict]]]
) -> List[Tuple[int, str, List[dict]]]:
    END_PUNCT = {"。", "！", "？", ".", "!", "?", "；", ";", "…", "\u201d", "\u2019", "）", ")"}
    merged: List[Tuple[int, str, List[dict]]] = []
    for i, (page_num, text, blocks) in enumerate(pages):
        if i > 0:
            prev_page, last_text, last_blocks = merged[-1]
            last_char = last_text.rstrip()[-1] if last_text.strip() else ""
            first_line = text.lstrip().split("\n")[0] if text.strip() else ""
            if (last_char and last_char not in END_PUNCT
                    and first_line
                    and not re.match(r"^第[一二三四五六七八九十\d]+[章节]", first_line)):
                merged[-1] = (
                    prev_page,
                    last_text.rstrip() + text.lstrip(),
                    last_blocks + blocks,
                )
                continue
        merged.append((page_num, text, blocks))
    return merged


def _rows_to_markdown(rows: list) -> str:
    cleaned: list[list[str]] = []
    for row in rows:
        if not row:
            continue
        norm_row = [
            (cell if cell is not None else "").strip().replace("|", "\\|").replace("\n", " ")
            for cell in row
        ]
        if any(c for c in norm_row):
            cleaned.append(norm_row)
    if not cleaned:
        return ""
    n_cols = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (n_cols - len(r)) for r in cleaned]
    header = cleaned[0]
    sep = ["---"] * n_cols
    body = cleaned[1:] if len(cleaned) > 1 else []
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _extract_structured_blocks(page, page_num: int) -> List[dict]:
    """
    用 pymupdf dict mode 抽取结构化 block：
      - heading: 字号 ≥ 正文中位字号 * 1.18，或 bold 且短行
      - table  : page.find_tables() → markdown
      - paragraph: 其余正文
    """
    text_dict = page.get_text("dict")
    sizes: list[float] = []
    for blk in text_dict.get("blocks", []):
        if blk.get("type") != 0:
            continue
        for line in blk.get("lines", []):
            for span in line.get("spans", []):
                if span.get("text", "").strip():
                    sizes.append(float(span.get("size", 0.0)))
    if not sizes:
        return []
    sizes_sorted = sorted(sizes)
    body_size = sizes_sorted[len(sizes_sorted) // 2] or 1.0

    items: list[tuple[float, dict]] = []
    table_rects: list[tuple[float, float, float, float]] = []

    try:
        tables = page.find_tables()
        tlist = getattr(tables, "tables", None) or list(tables)
    except Exception:
        tlist = []
    for tbl in tlist:
        try:
            rows = tbl.extract()
        except Exception:
            continue
        md = _rows_to_markdown(rows)
        if not md:
            continue
        bbox = tuple(tbl.bbox)
        table_rects.append(bbox)
        items.append((bbox[1], {"type": "table", "text": md, "page": page_num}))

    def _in_table(bbox):
        x0, y0, x1, y1 = bbox
        for tx0, ty0, tx1, ty1 in table_rects:
            if x0 >= tx0 - 2 and x1 <= tx1 + 2 and y0 >= ty0 - 2 and y1 <= ty1 + 2:
                return True
        return False

    for blk in text_dict.get("blocks", []):
        if blk.get("type") != 0:
            continue
        bbox = tuple(blk.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        if _in_table(bbox):
            continue
        max_size = 0.0
        bold = False
        line_texts: list[str] = []
        for line in blk.get("lines", []):
            parts: list[str] = []
            for span in line.get("spans", []):
                t = span.get("text", "")
                if not t.strip():
                    continue
                parts.append(t)
                max_size = max(max_size, float(span.get("size", 0.0)))
                if "Bold" in span.get("font", "") or (span.get("flags", 0) & 16):
                    bold = True
            if parts:
                line_texts.append("".join(parts).strip())
        text = "\n".join(line_texts).strip()
        if not text:
            continue

        is_heading = (
            max_size >= body_size * 1.18
            or (bold and len(text) <= 40 and "\n" not in text)
        )
        if is_heading:
            ratio = max_size / body_size if body_size > 0 else 1.0
            level = 1 if ratio >= 1.6 else (2 if ratio >= 1.3 else 3)
            items.append((bbox[1], {
                "type": "heading", "level": level, "text": text, "page": page_num,
            }))
        else:
            items.append((bbox[1], {
                "type": "paragraph", "text": text, "page": page_num,
            }))

    items.sort(key=lambda x: x[0])
    return [it[1] for it in items]


def load_pdf_pymupdf(file_path: str) -> List[Document]:
    try:
        import fitz
    except ImportError:
        raise ImportError("pip install pymupdf")
    doc = fitz.open(file_path)
    file_name = Path(file_path).name
    doc_id = _compute_doc_id(file_path)
    total_pages = len(doc)
    raw_pages: List[Tuple[int, str, List[dict]]] = []
    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text("text")
        cleaned = _clean_text(text)
        try:
            structured = _extract_structured_blocks(page, page_num + 1)
        except Exception:
            structured = []
        if cleaned.strip():
            raw_pages.append((page_num + 1, cleaned, structured))
    doc.close()
    merged = _merge_broken_sentences(raw_pages)
    return [
        Document(
            page_content=text,
            metadata={"source": file_name, "file_path": file_path,
                      "page": pg, "total_pages": total_pages, "doc_id": doc_id,
                      "structured_blocks": blocks}
        )
        for pg, text, blocks in merged if text.strip()
    ]


def load_pdf_pypdf(file_path: str) -> List[Document]:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pip install pypdf")
    reader = PdfReader(file_path)
    file_name = Path(file_path).name
    doc_id = _compute_doc_id(file_path)
    total_pages = len(reader.pages)
    docs = []
    for i, page in enumerate(reader.pages):
        text = _clean_text(page.extract_text() or "")
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": file_name, "file_path": file_path,
                          "page": i + 1, "total_pages": total_pages, "doc_id": doc_id}
            ))
    return docs


def load_pdf(file_path: str) -> List[Document]:
    try:
        docs = load_pdf_pymupdf(file_path)
        if docs:
            return docs
    except Exception:
        pass
    return load_pdf_pypdf(file_path)


def load_documents_from_dir(directory: str) -> List[Document]:
    all_docs = []
    for pdf_file in sorted(Path(directory).glob("*.pdf")):
        try:
            docs = load_pdf(str(pdf_file))
            all_docs.extend(docs)
            print(f"  ok {pdf_file.name}: {len(docs)} pages")
        except Exception as e:
            print(f"  failed {pdf_file.name}: {e}")
    return all_docs


def get_doc_metadata(file_path: str) -> dict:
    try:
        import fitz
        doc = fitz.open(file_path)
        info = {"file_name": Path(file_path).name, "total_pages": len(doc),
                "doc_id": _compute_doc_id(file_path),
                "file_size_kb": round(Path(file_path).stat().st_size / 1024, 1)}
        doc.close()
        return info
    except Exception:
        return {"file_name": Path(file_path).name, "total_pages": 0,
                "doc_id": _compute_doc_id(file_path),
                "file_size_kb": round(Path(file_path).stat().st_size / 1024, 1)}
