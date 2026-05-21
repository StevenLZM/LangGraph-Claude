# Repository Guidelines

## Project Structure & Module Organization

This is a Python monorepo for staged AI Agent projects. Each numbered directory is mostly self-contained:

- `01_RAG/`: RAG knowledge-base app, `rag/`, `memory/`, `mcp_local/`, `evals/`, and `tests/`.
- `02_REACT_AGENT/`: Streamlit ReAct and plan/execute demo, with `agent/`, `tools/`, `config/`, and `tests/`.
- `03_MULTI_AGENT/`: InsightLoop LangGraph system, with `agents/`, `graph/`, `tools/`, `app/`, `evals/`, and `tests/`.
- `04_HUMAN_IN_THE_LOOP/`: design and engineering notes for HITL patterns.
- `05_PRODUCT_AGENT/`: production customer-service agent, with `api/`, `agent/`, `memory/`, `llm/`, `monitoring/`, `infra/`, and `tests/`.
- `docs/superpowers/`: design specs and implementation plans.

## Build, Test, and Development Commands

Install dependencies in the module you are changing:

```bash
cd 01_RAG && pip install -r requirements.txt
cd 02_REACT_AGENT && pip install -r requirements.txt
cd 03_MULTI_AGENT && pip install -r requirements.txt
cd 05_PRODUCT_AGENT && pip install -r requirements.txt
```

Common local commands:

```bash
cd 01_RAG && streamlit run app.py
cd 02_REACT_AGENT && streamlit run app.py
cd 03_MULTI_AGENT && make test
cd 03_MULTI_AGENT && make api
cd 03_MULTI_AGENT && make ui
cd 05_PRODUCT_AGENT && uvicorn api.main:app --host 0.0.0.0 --port 8000
cd 05_PRODUCT_AGENT && docker compose up --build
```

Use `PYTHONPATH=.` for package modules, for example `cd 03_MULTI_AGENT && PYTHONPATH=. python -m scripts.run_local "query"`.

## Coding Style & Naming Conventions

Use Python 3.11+ for LangGraph and MCP work. Follow existing style: 4-space indentation, practical type hints, `snake_case` functions, `PascalCase` classes, and uppercase environment constants. Keep cross-project imports behind adapters such as `03_MULTI_AGENT/tools/kb_retriever.py`.

## Testing Guidelines

Tests use `pytest` and live under each module’s `tests/` directory. Name files `test_*.py` and keep offline unit tests independent of API keys. Run from the module root:

```bash
pytest tests -q
pytest tests/test_reranker.py -q
```

For quality checks, use module eval commands such as `python -m evals.run --dry-run`, `make eval-smoke`, or `make eval`.

## Commit & Pull Request Guidelines

Git history uses short imperative messages, often with a module prefix, for example `01_RAG 评测系统`, `chore: ...`, or `Add RocketMQ business messaging to product agent`. Keep commits scoped to one module or feature.

PRs should include a summary, changed module paths, test/eval commands run, configuration changes, and screenshots or API examples for UI/API behavior. Link related issues or design docs.

## Security & Configuration Tips

Many modules use `.env.example`; copy to `.env` and fill real keys locally. This private repo intentionally tracks some `.env`, SQLite, and vector data for reproducible demos, so review secrets and generated artifacts carefully before sharing externally.
