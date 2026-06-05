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
        "research_subgraph",
        "reflector",
        "writer",
    }
    assert expected.issubset(nodes), f"缺少节点: {expected - nodes}"
    assert {
        "web_researcher",
        "academic_researcher",
        "code_researcher",
        "kb_researcher",
    }.isdisjoint(nodes)

    subgraphs = dict(graph.get_subgraphs())
    assert "research_subgraph" in subgraphs
    research_nodes = set(subgraphs["research_subgraph"].get_graph().nodes.keys())
    expected_research_nodes = {
        "research_dispatcher",
        "web_researcher",
        "academic_researcher",
        "code_researcher",
        "kb_researcher",
    }
    assert expected_research_nodes.issubset(research_nodes)


def test_state_typed_dict_imports():
    from graph.state import ResearchState, merge_evidence

    assert ResearchState is not None
    assert callable(merge_evidence)
