PYTHON ?= python
INSIGHTLOOP_API ?= http://localhost:8080

.PHONY: test eval eval-smoke api ui docker-up docker-down docker-config

test:
	PYTHONPATH=. pytest tests -q

eval:
	PYTHONPATH=. $(PYTHON) -m evals.run

eval-smoke:
	PYTHONPATH=. $(PYTHON) -m evals.run --limit 1

api:
	PYTHONPATH=. uvicorn app.api:app --host 0.0.0.0 --port 8080

ui:
	INSIGHTLOOP_API=$(INSIGHTLOOP_API) PYTHONPATH=. streamlit run app/streamlit_ui.py --server.address=0.0.0.0 --server.port=8501

docker-up:
	docker compose up --build

docker-down:
	docker compose down

docker-config:
	docker compose config
