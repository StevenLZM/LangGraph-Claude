"""Research subgraph assembly.

The parent graph treats research as one stage. This subgraph owns the
sub_question x source fan-out and fans all researcher outputs back into the
shared ResearchState reducers.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.researcher_academic import academic_researcher_node
from agents.researcher_code import code_researcher_node
from agents.researcher_kb import kb_researcher_node
from agents.researcher_web import web_researcher_node
from config.tracing import tagged_node
from graph.state import ResearchState

_SOURCE_TO_NODE = {
    "web": "web_researcher",
    "academic": "academic_researcher",
    "code": "code_researcher",
    "kb": "kb_researcher",
}


def build_research_sends(state: ResearchState) -> list[Send]:
    """Build Send tasks for all pending sub_questions and requested sources."""
    sends: list[Send] = []
    for sq in state.get("plan") or []:
        if getattr(sq, "status", "pending") == "done":
            continue
        sources = getattr(sq, "recommended_sources", []) or ["web"]
        for src in sources:
            node = _SOURCE_TO_NODE.get(src)
            if node is None:
                continue
            sends.append(
                Send(
                    node,
                    {
                        "sub_question": sq,
                        "research_query": state.get("research_query", ""),
                    },
                )
            )
    return sends


async def research_dispatcher_node(state: ResearchState) -> dict[str, Any]:
    return {}


def research_dispatch_route(state: ResearchState) -> Any:
    return build_research_sends(state) or END


def build_research_subgraph():
    wf = StateGraph(ResearchState)

    wf.add_node("research_dispatcher", research_dispatcher_node)
    wf.add_node("web_researcher", tagged_node("web_researcher", web_researcher_node))
    wf.add_node("academic_researcher", tagged_node("academic_researcher", academic_researcher_node))
    wf.add_node("code_researcher", tagged_node("code_researcher", code_researcher_node))
    wf.add_node("kb_researcher", tagged_node("kb_researcher", kb_researcher_node))

    wf.add_edge(START, "research_dispatcher")
    wf.add_conditional_edges(
        "research_dispatcher",
        research_dispatch_route,
        {
            "web_researcher": "web_researcher",
            "academic_researcher": "academic_researcher",
            "code_researcher": "code_researcher",
            "kb_researcher": "kb_researcher",
            END: END,
        },
    )
    for node in ("web_researcher", "academic_researcher", "code_researcher", "kb_researcher"):
        wf.add_edge(node, END)

    return wf.compile(name="research_subgraph")
