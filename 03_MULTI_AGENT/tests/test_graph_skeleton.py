"""仅验证图能编译 + 节点齐全（不调真实 LLM）。完整闭环测试见 test_end_to_end_offline.py。"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from graph.workflow import build_graph


def test_graph_compiles_and_has_all_nodes():
    graph = build_graph(checkpointer=MemorySaver())
    nodes = set(graph.get_graph().nodes.keys())
    expected = {
        "planner",
        "supervisor",
        "web_researcher",
        "academic_researcher",
        "code_researcher",
        "kb_researcher",
        "reflector",
        "writer",
    }
    assert expected.issubset(nodes), f"缺少节点: {expected - nodes}"


def test_state_typed_dict_imports():
    from graph.state import ResearchState, merge_evidence

    assert ResearchState is not None
    assert callable(merge_evidence)
