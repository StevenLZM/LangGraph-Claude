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


def _merge_broken_sentences(pages: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    END_PUNCT = {"。", "！", "？", ".", "!", "?", "；", ";", "…", "\u201d", "\u2019", "）", ")"}
    merged = []
    for i, (page_num, text) in enumerate(pages):
        if i > 0:
            _, last_text = merged[-1]
            last_char = last_text.rstrip()[-1] if last_text.strip() else ""
            first_line = text.lstrip().split("\n")[0] if text.strip() else ""
            if (last_char and last_char not in END_PUNCT
                    and first_line
                    and not re.match(r"^第[一二三四五六七八九十\d]+[章节]", first_line)):
                merged[-1] = (merged[-1][0], last_text.rstrip() + text.lstrip())
                continue
        merged.append((page_num, text))
    return merged


def load_pdf_pymupdf(file_path: str) -> List[Document]:
    try:
        import fitz
    except ImportError:
        raise ImportError("pip install pymupdf")
    doc = fitz.open(file_path)
    file_name = Path(file_path).name
    doc_id = _compute_doc_id(file_path)
    total_pages = len(doc)
    raw_pages: List[Tuple[int, str]] = []
    for page_num in range(total_pages):
        text = doc[page_num].get_text("text")
        cleaned = _clean_text(text)
        if cleaned.strip():
            raw_pages.append((page_num + 1, cleaned))
    doc.close()
    merged = _merge_broken_sentences(raw_pages)
    return [
        Document(
            page_content=text,
            metadata={"source": file_name, "file_path": file_path,
                      "page": pg, "total_pages": total_pages, "doc_id": doc_id}
        )
        for pg, text in merged if text.strip()
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
