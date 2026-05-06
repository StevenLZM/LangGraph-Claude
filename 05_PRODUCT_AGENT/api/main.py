from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages import SystemMessage

from agent.graph import build_customer_service_graph
from api.schemas import ChatRequest, ChatResponse, DeleteMemoriesResponse
from api.settings import settings
from api.ui import CUSTOMER_SERVICE_UI
from memory.long_term import UserMemoryManager
from memory.session_store import SessionStore
from memory.short_term import ContextWindowManager

customer_service_graph = build_customer_service_graph()
context_window_manager = ContextWindowManager()
session_store = SessionStore(settings.memory_db)
user_memory_manager = UserMemoryManager(settings.memory_db)

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
            "memory": "sqlite",
            "llm": settings.llm_mode,
        },
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    loaded_session = session_store.load_session(req.session_id)
    prior_messages = loaded_session["messages"] if loaded_session else []
    user_memories = user_memory_manager.load_memories(req.user_id, req.message)
    messages = context_window_manager.trim(
        [*prior_messages, HumanMessage(content=req.message)]
    )
    result = customer_service_graph.invoke(
        {
            "session_id": req.session_id,
            "user_id": req.user_id,
            "messages": messages,
            "user_memories": user_memories,
        },
        config={"configurable": {"thread_id": req.session_id}},
    )
    last_message = result["messages"][-1]
    answer = last_message.content if isinstance(last_message, AIMessage) else str(last_message)
    memory_summary = _extract_memory_summary(result.get("messages", []))
    user_memory_manager.save_from_turn(req.user_id, req.message, answer)
    session_store.save_session(
        session_id=req.session_id,
        user_id=req.user_id,
        messages=result.get("messages", []),
        metadata={
            "summary": memory_summary,
            "needs_human_transfer": result.get("needs_human_transfer", False),
            "transfer_reason": result.get("transfer_reason", ""),
            "token_used": result.get("token_used", 0),
            "quality_score": result.get("quality_score"),
        },
    )
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
        user_memories=user_memories,
        memory_summary=memory_summary,
    )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    session = session_store.get_public_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return session


@app.delete("/users/{user_id}/memories", response_model=DeleteMemoriesResponse)
async def delete_user_memories(user_id: str) -> DeleteMemoriesResponse:
    deleted = user_memory_manager.delete_memories(user_id)
    return DeleteMemoriesResponse(user_id=user_id, deleted=deleted)


def _extract_memory_summary(messages: list[object]) -> str:
    for message in messages:
        if isinstance(message, SystemMessage) and message.additional_kwargs.get("type") == "summary":
            return str(message.content)
    return ""
