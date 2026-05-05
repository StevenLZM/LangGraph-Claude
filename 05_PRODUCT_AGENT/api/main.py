from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import build_customer_service_graph
from api.schemas import ChatRequest, ChatResponse
from api.settings import settings
from api.ui import CUSTOMER_SERVICE_UI

customer_service_graph = build_customer_service_graph()

app = FastAPI(title=settings.app_name, version=settings.app_version)


@app.get("/", response_class=HTMLResponse)
async def customer_service_workspace() -> str:
    return CUSTOMER_SERVICE_UI


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "graph_ready": customer_service_graph is not None,
        "dependencies": {
            "api": "ok",
            "redis": "configured" if settings.redis_url else "not_configured",
            "database": "configured" if settings.database_url else "not_configured",
            "llm": settings.llm_mode,
        },
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    result = customer_service_graph.invoke(
        {
            "session_id": req.session_id,
            "user_id": req.user_id,
            "messages": [HumanMessage(content=req.message)],
        },
        config={"configurable": {"thread_id": req.session_id}},
    )
    last_message = result["messages"][-1]
    answer = last_message.content if isinstance(last_message, AIMessage) else str(last_message)
    return ChatResponse(
        session_id=result.get("session_id", req.session_id),
        user_id=result.get("user_id", req.user_id),
        answer=answer,
        needs_human_transfer=result.get("needs_human_transfer", False),
        transfer_reason=result.get("transfer_reason", ""),
        order_context=result.get("order_context"),
        token_used=result.get("token_used", 0),
        response_time_ms=result.get("response_time_ms", 0),
        quality_score=result.get("quality_score"),
    )
