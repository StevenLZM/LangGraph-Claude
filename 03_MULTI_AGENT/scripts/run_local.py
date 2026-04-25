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
        # await意味着main方法可以继续执行，不是r1可以跳过继续执行
        r1 = await g.ainvoke(
            {"research_query": query, "audience": "intermediate", "messages": [], "evidence": []},
            config=cfg,
        )
        intr = r1.get("__interrupt__")
        if not intr:
            print("❌ 未触发 interrupt，直接返回：", r1)
            return
        print(f"中断信息：{intr}")

        proposed = intr[0].value if isinstance(intr, list) else intr.value
        print("📋 Planner 生成的计划：")
        for sq in proposed["plan"]["sub_questions"]:
            print(f"  - {sq['id']}: {sq['question']} (sources={sq['recommended_sources']})")

        # ========== 用户交互部分 ==========
        print("\n" + "=" * 60)
        print("请选择操作：")
        print("  1. 接受计划，开始执行")
        print("  2. 修改计划（将重新生成）")
        print("  3. 取消任务")
        print("=" * 60)
        
        choice = input("请输入选项 (1/2/3): ").strip()
        
        if choice == "1":
            # 接受计划
            decision = {"plan": proposed["plan"], "action": "accept"}
            print("\n✅ 计划已接受，开始执行...")
            
        elif choice == "2":
            # 修改计划：获取用户想要修改的内容
            print("\n✏️ 修改计划：")
            modified_plan = proposed["plan"].copy()
            
            for i, sq in enumerate(modified_plan["sub_questions"], 1):
                print(f"\n问题 {i}: {sq['question']}")
                new_question = input(f"  新问题 (回车保持不变): ").strip()
                if new_question:
                    sq["question"] = new_question
                
                new_sources = input(f"  新来源 (逗号分隔, 回车保持不变): ").strip()
                if new_sources:
                    sq["recommended_sources"] = [s.strip() for s in new_sources.split(",")]
            
            decision = {"plan": modified_plan, "action": "accept"}
            print("\n✅ 计划已更新，开始执行...")
            
        elif choice == "3":
            # 取消任务
            print("\n❌ 任务已取消")
            return
            
        else:
            print("\n⚠️ 无效选项，默认接受计划")
            decision = {"plan": proposed["plan"], "action": "accept"}
        # ==================================

        # print("\n=== 自动接受计划（CLI 场景，真实 UI 处由用户编辑）===")
        resumed = await g.ainvoke(Command(resume={"plan": proposed["plan"]}), config=cfg)

        report = resumed.get("final_report", "")
        print(f"\n=== 报告 ===\n（{len(report)} 字, {len(resumed.get('citations', []))} 引用, 路径={resumed.get('report_path')}）\n")
        # print(report[:2000])
        # if len(report) > 2000:
        #     print(f"\n... [{len(report) - 2000} more chars] ...")
    finally:
        await bootstrap.shutdown()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "分析 2025 年开源 Agent 框架格局，重点对比 LangGraph / AutoGen / CrewAI / LlamaIndex"
    asyncio.run(run(q))
