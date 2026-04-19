"""本地真实调用闭环脚本 —— 不通过 FastAPI，直接驱动图；用于快速验证 LLM/工具接通情况。

用法：
    cd 03_MULTI_AGENT
    PYTHONPATH=.:../01_RAG python -m scripts.run_local "分析 2025 年开源 Agent 框架格局"
"""
from __future__ import annotations

import asyncio
import logging
import sys
import uuid

from langgraph.types import Command

from app import bootstrap


async def run(query: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    await bootstrap.startup()
    try:
        g = bootstrap.app_state.graph
        tid = uuid.uuid4().hex[:8]
        cfg = {"configurable": {"thread_id": tid}}

        print(f"\n=== Turn 1: Planner + interrupt ===\nthread_id={tid}\n")
        r1 = await g.ainvoke(
            {"research_query": query, "audience": "intermediate", "messages": [], "evidence": []},
            config=cfg,
        )
        intr = r1.get("__interrupt__")
        if not intr:
            print("❌ 未触发 interrupt，直接返回：", r1)
            return

        proposed = intr[0].value if isinstance(intr, list) else intr.value
        print("📋 Planner 生成的计划：")
        for sq in proposed["plan"]["sub_questions"]:
            print(f"  - {sq['id']}: {sq['question']} (sources={sq['recommended_sources']})")

        print("\n=== 自动接受计划（CLI 场景，真实 UI 处由用户编辑）===")
        resumed = await g.ainvoke(Command(resume={"plan": proposed["plan"]}), config=cfg)

        report = resumed.get("final_report", "")
        print(f"\n=== 报告 ===\n（{len(report)} 字, {len(resumed.get('citations', []))} 引用, 路径={resumed.get('report_path')}）\n")
        print(report[:2000])
        if len(report) > 2000:
            print(f"\n... [{len(report) - 2000} more chars] ...")
    finally:
        await bootstrap.shutdown()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "分析 2025 年开源 Agent 框架格局，重点对比 LangGraph / AutoGen / CrewAI"
    asyncio.run(run(q))
