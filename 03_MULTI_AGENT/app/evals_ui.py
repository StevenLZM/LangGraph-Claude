"""InsightLoop Eval Dashboard —— 看 evals/results 下历次评测产出。

启动：
    streamlit run app/evals_ui.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "evals" / "results"
DIMS = ["coverage", "accuracy", "citation", "overall"]
DIM_CN = {"coverage": "覆盖度", "accuracy": "准确性", "citation": "引用质量", "overall": "综合"}


def _list_runs() -> list[str]:
    if not RESULTS_ROOT.exists():
        return []
    return sorted(
        [p.name for p in RESULTS_ROOT.iterdir() if p.is_dir() and (p / "results.jsonl").exists()],
        reverse=True,
    )


def _load_run(run_id: str) -> pd.DataFrame:
    path = RESULTS_ROOT / run_id / "results.jsonl"
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            case = rec.get("case", {})
            score = rec.get("score") or {}
            rows.append({
                "case_id": case.get("id"),
                "category": case.get("category"),
                "query": case.get("query"),
                "audience": case.get("audience"),
                "elapsed_sec": rec.get("elapsed_sec"),
                "evidence_count": rec.get("evidence_count"),
                "report_path": rec.get("report_path"),
                "report_md": rec.get("report_md", ""),
                "error": rec.get("error"),
                "coverage": score.get("coverage"),
                "accuracy": score.get("accuracy"),
                "citation": score.get("citation"),
                "overall": score.get("overall"),
                "rationale": score.get("rationale", ""),
            })
    return pd.DataFrame(rows)


def _means(df: pd.DataFrame) -> dict[str, float]:
    ok = df[df["error"].isna()]
    return {d: round(float(ok[d].mean()), 1) if not ok[d].dropna().empty else 0.0 for d in DIMS}


def main():
    st.set_page_config(page_title="InsightLoop Eval", layout="wide")
    st.title("📊 InsightLoop 评测看板")

    runs = _list_runs()
    if not runs:
        st.warning("还没有评测结果。先跑：`PYTHONPATH=. python -m evals.run --limit 1`")
        return

    mode = st.sidebar.radio("模式", ["单 run 详情", "两 run 对比"])

    if mode == "单 run 详情":
        run_id = st.sidebar.selectbox("选择 run", runs, index=0)
        df = _load_run(run_id)

        ok_cnt = int(df["error"].isna().sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("用例数", len(df))
        c2.metric("成功", ok_cnt)
        c3.metric("失败", len(df) - ok_cnt)
        c4.metric("平均用时(s)", round(float(df["elapsed_sec"].dropna().mean() or 0), 1))

        st.markdown("### 维度均值")
        m = _means(df)
        st.bar_chart(pd.DataFrame(
            {"分数": [m[d] for d in DIMS]}, index=[DIM_CN[d] for d in DIMS],
        ))

        st.markdown("### 用例明细")
        st.dataframe(
            df[["case_id", "category", "coverage", "accuracy", "citation", "overall",
                "elapsed_sec", "evidence_count", "error"]],
            use_container_width=True, hide_index=True,
        )

        st.markdown("### 单条详情")
        case_ids = df["case_id"].dropna().tolist()
        if case_ids:
            sel = st.selectbox("选择用例", case_ids)
            row = df[df["case_id"] == sel].iloc[0]
            left, right = st.columns([1, 1])
            with left:
                st.markdown(f"**Query**: {row['query']}")
                st.markdown(f"**Category / Audience**: {row['category']} / {row['audience']}")
                if row["error"]:
                    st.error(f"错误: {row['error']}")
                else:
                    st.markdown(
                        f"- 覆盖: {row['coverage']} / 准确: {row['accuracy']} / 引用: {row['citation']} "
                        f"/ **综合: {row['overall']}**"
                    )
                    st.markdown(f"**Rationale**: {row['rationale']}")
                    if row.get("report_path"):
                        st.caption(f"报告归档: `{row['report_path']}`")
            with right:
                if row.get("report_md"):
                    with st.expander("📄 报告全文", expanded=False):
                        st.markdown(row["report_md"])

    else:  # 两 run 对比
        if len(runs) < 2:
            st.warning("至少需要两次 run 才能对比")
            return
        a = st.sidebar.selectbox("Run A", runs, index=0)
        b = st.sidebar.selectbox("Run B", runs, index=1)
        df_a, df_b = _load_run(a), _load_run(b)
        ma, mb = _means(df_a), _means(df_b)
        chart_df = pd.DataFrame(
            {a: [ma[d] for d in DIMS], b: [mb[d] for d in DIMS]},
            index=[DIM_CN[d] for d in DIMS],
        )
        st.markdown("### 维度均值对比")
        st.bar_chart(chart_df)
        st.markdown("### 逐用例对比")
        merged = df_a.merge(
            df_b, on="case_id", how="outer", suffixes=(f"_{a}", f"_{b}")
        )[["case_id", f"overall_{a}", f"overall_{b}"]]
        merged["delta"] = merged[f"overall_{b}"] - merged[f"overall_{a}"]
        st.dataframe(merged, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
