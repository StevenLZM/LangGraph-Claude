"""
tests/e2e_date_aware_retrieval.py — 端到端验证脚本

流程：
1. 清空向量库并重建
2. Ingest data/documents/ 下的 PDF（分块时自动抽取日期）
3. 针对六类意图 query，观察 time_intent、metadata filter、命中顺序
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chunker import chunk_documents
from rag.loader import load_pdf
from rag.query_rewriter import rewrite_query
from rag.retriever import reset_retriever_cache, retrieve_with_hybrid
from rag.vectorstore import add_documents, get_vectorstore

from config import path_config


def ingest_all():
    pdfs = sorted(path_config.DOCUMENTS_DIR.glob("*.pdf"))
    print(f"\n发现 {len(pdfs)} 份 PDF")

    vs = get_vectorstore()

    # 清空现有集合以保证 metadata schema 干净
    try:
        existing = vs.get()
        if existing.get("ids"):
            vs.delete(ids=existing["ids"])
            print(f"已清空旧索引 {len(existing['ids'])} 条")
    except Exception as e:
        print(f"清空失败（忽略）: {e}")

    for pdf in pdfs:
        print(f"\n→ Ingest {pdf.name}")
        pages = load_pdf(str(pdf))
        chunks = chunk_documents(pages)
        doc_id = pages[0].metadata["doc_id"]
        n = add_documents(chunks, doc_id, vs)
        # 打印样本 metadata
        sample = chunks.children[0].metadata if chunks.children else {}
        date_info = {k: sample.get(k) for k in ("upload_date", "doc_date_min", "doc_date_max", "has_doc_date")}
        print(f"   parents={len(chunks.parents)} children={n} 日期={date_info}")

    reset_retriever_cache()


def run_query(q: str, tag: str):
    print("\n" + "═" * 70)
    print(f"[{tag}] {q}")
    print("═" * 70)

    rw = rewrite_query(q, use_llm=False)  # 用规则版，避免 LLM 依赖
    print(f"time_intent = {rw['time_intent']}")

    docs = retrieve_with_hybrid(
        query=rw["rewritten_query"],
        time_intent=rw["time_intent"],
        top_k=3,
    )
    print(f"TopK={len(docs)}")
    for i, d in enumerate(docs, 1):
        md = d.metadata
        print(
            f"  {i}. source={md.get('source')!r:40s} "
            f"doc_date=[{md.get('doc_date_min')}, {md.get('doc_date_max')}] "
            f"upload={md.get('upload_date')} score={md.get('best_child_score')}"
        )


def main():
    ingest_all()

    # 六类意图各一条
    cases = [
        ("none", "AI Agent 的核心能力"),
        ("latest", "最新的发票"),
        ("year", "2024 年的发票"),
        ("before", "2023 年之前的发票"),
        ("after", "2024 年之后的合同"),
        ("range", "近 90 天的发票"),
    ]
    for tag, q in cases:
        run_query(q, tag)


if __name__ == "__main__":
    main()
