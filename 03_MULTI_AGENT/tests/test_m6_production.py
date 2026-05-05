from __future__ import annotations

import importlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_eval_dataset_has_twenty_balanced_cases():
    rows = [
        json.loads(line)
        for line in (ROOT / "evals" / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 20
    assert len({row["id"] for row in rows}) == 20
    assert Counter(row["category"] for row in rows) == {
        "技术": 5,
        "产业": 5,
        "对比": 5,
        "追问": 5,
    }
    assert {row["audience"] for row in rows} <= {"beginner", "intermediate", "expert"}
    assert all(row["query"].strip() for row in rows)


def test_docker_and_makefile_assets_define_m6_entrypoints():
    dockerfile = ROOT / "Dockerfile"
    compose = ROOT / "docker-compose.yml"
    dockerignore = ROOT / ".dockerignore"
    makefile = ROOT / "Makefile"

    assert dockerfile.exists()
    assert compose.exists()
    assert dockerignore.exists()
    assert makefile.exists()

    dockerfile_text = dockerfile.read_text(encoding="utf-8")
    compose_text = compose.read_text(encoding="utf-8")
    makefile_text = makefile.read_text(encoding="utf-8")

    assert "pip install" in dockerfile_text
    assert "requirements.txt" in dockerfile_text
    assert "api:" in compose_text
    assert "ui:" in compose_text
    assert "8080:8080" in compose_text
    assert "8501:8501" in compose_text
    assert "INSIGHTLOOP_API=http://api:8080" in compose_text
    assert "eval:" in makefile_text
    assert "eval-smoke:" in makefile_text


def test_streamlit_ui_prefers_insightloop_api_and_accepts_legacy_api_base_url(monkeypatch):
    sys.modules.pop("app.streamlit_ui", None)
    monkeypatch.delenv("INSIGHTLOOP_API", raising=False)
    monkeypatch.setenv("API_BASE_URL", "http://api:8080")

    module = importlib.import_module("app.streamlit_ui")

    assert module.API_BASE == "http://api:8080"

    sys.modules.pop("app.streamlit_ui", None)
    monkeypatch.setenv("INSIGHTLOOP_API", "http://explicit:8080")

    module = importlib.import_module("app.streamlit_ui")

    assert module.API_BASE == "http://explicit:8080"


@pytest.mark.asyncio
async def test_tagged_node_adds_agent_tag_and_metadata():
    from config.tracing import tagged_node

    async def fake_node(state):
        return {"seen": state["value"]}

    node = tagged_node("planner", fake_node)

    assert "agent:planner" in node.config["tags"]
    assert node.config["metadata"]["agent"] == "planner"
    assert await node.ainvoke({"value": 7}) == {"seen": 7}


def test_workflow_wraps_all_business_nodes_with_langsmith_tags(monkeypatch):
    from graph import workflow

    seen: list[str] = []

    def fake_tagged_node(name, fn):
        seen.append(name)
        return fn

    monkeypatch.setattr(workflow, "tagged_node", fake_tagged_node, raising=False)

    workflow.build_graph()

    assert seen == [
        "planner",
        "supervisor",
        "web_researcher",
        "academic_researcher",
        "code_researcher",
        "kb_researcher",
        "reflector",
        "writer",
    ]
