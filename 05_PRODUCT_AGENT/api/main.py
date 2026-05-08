from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages import SystemMessage

from agent.graph import build_customer_service_graph
from api.middleware.rate_limiter import RateLimiter, TokenBudgetExceeded
from api.schemas import ChatRequest, ChatResponse, DeleteMemoriesResponse
from api.settings import settings
from api.ui import CUSTOMER_SERVICE_UI
from memory.long_term import UserMemoryManager
from memory.session_store import SessionStore
from memory.short_term import ContextWindowManager
from llm.factory import build_customer_service_llm
from monitoring.evaluator import AutoQualityEvaluator
from monitoring.metrics import record_chat_error, record_chat_request, render_prometheus_metrics
from monitoring.tracing import build_trace_config, configure_langsmith

customer_service_graph = build_customer_service_graph()
context_window_manager = ContextWindowManager()
session_store = SessionStore(settings.memory_db)
user_memory_manager = UserMemoryManager(settings.memory_db)
quality_evaluator = AutoQualityEvaluator(alert_threshold=settings.quality_alert_threshold)
customer_service_llm_setup = build_customer_service_llm(settings)
customer_service_llm = customer_service_llm_setup.llm
rate_limiter = RateLimiter(
    redis_url=settings.redis_url,
    user_rate_limit_per_minute=settings.user_rate_limit_per_minute,
    global_qps_limit=settings.global_qps_limit,
    single_request_token_budget=settings.single_request_token_budget,
    global_hourly_token_budget=settings.global_hourly_token_budget,
)

app = FastAPI(title=settings.app_name, version=settings.app_version)
configure_langsmith(
    enabled=settings.langchain_tracing_v2,
    endpoint=settings.langchain_endpoint,
    api_key=settings.langchain_api_key,
    project=settings.langchain_project,
)


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
            "llm_startup_error": customer_service_llm_setup.startup_error,
            "rate_limiter": rate_limiter.backend,
        },
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    return render_prometheus_metrics()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    started_at = time.perf_counter()
    try:
        await rate_limiter.check_user_rate(req.user_id)
        await rate_limiter.check_global_qps()
    except HTTPException as exc:
        status = "rate_limited" if exc.status_code == 429 else "error"
        record_chat_error(status=status, session_id=req.session_id)
        raise

    loaded_session = session_store.load_session(req.session_id)
    prior_messages = loaded_session["messages"] if loaded_session else []
    user_memories = user_memory_manager.load_memories(req.user_id, req.message)
    messages = context_window_manager.trim(
        [*prior_messages, HumanMessage(content=req.message)]
    )
    estimated_tokens = context_window_manager.count_tokens(messages)
    try:
        await rate_limiter.reserve_token_budget(estimated_tokens)
    except TokenBudgetExceeded as exc:
        return _build_degraded_chat_response(
            req=req,
            messages=messages,
            user_memories=user_memories,
            memory_summary=_extract_memory_summary(messages),
            reason=exc.reason,
            token_used=exc.token_count,
            started_at=started_at,
        )

    trace_config = build_trace_config(
        session_id=req.session_id,
        user_id=req.user_id,
        environment=settings.observability_env,
        app_version=settings.app_version,
    )
    result = customer_service_graph.invoke(
        {
            "session_id": req.session_id,
            "user_id": req.user_id,
            "messages": messages,
            "user_memories": user_memories,
        },
        config={
            "configurable": {"thread_id": req.session_id},
            "tags": trace_config["tags"],
            "metadata": trace_config["metadata"],
        },
    )
    last_message = result["messages"][-1]
    answer = last_message.content if isinstance(last_message, AIMessage) else str(last_message)
    llm_trace = _offline_llm_trace(settings.llm_mode, result.get("tool_name", ""), user_memories)
    if settings.llm_mode != "offline_stub":
        llm_answer, llm_trace = await _maybe_generate_llm_answer(
            question=req.message,
            draft_answer=answer,
            user_memories=user_memories,
            result=result,
        )
        if llm_answer:
            answer = llm_answer
            result_messages = list(result.get("messages", []))
            if result_messages and isinstance(result_messages[-1], AIMessage):
                result_messages[-1] = AIMessage(
                    content=answer,
                    additional_kwargs={
                        **result_messages[-1].additional_kwargs,
                        "llm_trace": llm_trace,
                    },
                )
                result = {**result, "messages": result_messages}
    memory_summary = _extract_memory_summary(result.get("messages", []))
    response_time_ms = result.get("response_time_ms", 0)
    token_used = result.get("token_used", 0)
    evaluation = quality_evaluator.evaluate(
        question=req.message,
        answer=answer,
        context={
            "order_context": result.get("order_context"),
            "needs_human_transfer": result.get("needs_human_transfer", False),
            "transfer_reason": result.get("transfer_reason", ""),
            "user_memories": user_memories,
        },
    )
    quality_score = evaluation.score
    saved_memories = user_memory_manager.save_from_turn(req.user_id, req.message, answer)
    if saved_memories:
        user_memories = user_memory_manager.load_memories(req.user_id, req.message)
    session_store.save_session(
        session_id=req.session_id,
        user_id=req.user_id,
        messages=result.get("messages", []),
        metadata={
            "summary": memory_summary,
            "needs_human_transfer": result.get("needs_human_transfer", False),
            "transfer_reason": result.get("transfer_reason", ""),
            "token_used": token_used,
            "quality_score": quality_score,
            "quality_evaluation": evaluation.to_dict(),
            "quality_alert": not evaluation.passed,
            "trace_metadata": trace_config["metadata"],
            "llm_trace": llm_trace,
        },
    )
    record_chat_request(
        status="transferred" if result.get("needs_human_transfer", False) else "success",
        session_id=req.session_id,
        response_time_ms=response_time_ms,
        token_used=token_used,
        quality_score=quality_score,
        needs_human_transfer=result.get("needs_human_transfer", False),
    )
    return ChatResponse(
        session_id=result.get("session_id", req.session_id),
        user_id=result.get("user_id", req.user_id),
        answer=answer,
        needs_human_transfer=result.get("needs_human_transfer", False),
        transfer_reason=result.get("transfer_reason", ""),
        order_context=result.get("order_context"),
        token_used=token_used,
        response_time_ms=response_time_ms,
        quality_score=quality_score,
        user_memories=user_memories,
        memory_summary=memory_summary,
        degraded=False,
        degrade_reason="",
        llm_trace=llm_trace,
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


def _build_degraded_chat_response(
    *,
    req: ChatRequest,
    messages: list[object],
    user_memories: list[str],
    memory_summary: str,
    reason: str,
    token_used: int,
    started_at: float,
) -> ChatResponse:
    answer = _degraded_answer(reason)
    response_messages = [
        *messages,
        AIMessage(
            content=answer,
            additional_kwargs={"degraded": True, "degrade_reason": reason},
        ),
    ]
    response_time_ms = max(0, int((time.perf_counter() - started_at) * 1000))
    evaluation = quality_evaluator.evaluate(
        question=req.message,
        answer=answer,
        context={"degraded": True, "degrade_reason": reason},
    )
    session_store.save_session(
        session_id=req.session_id,
        user_id=req.user_id,
        messages=response_messages,
        metadata={
            "summary": memory_summary,
            "needs_human_transfer": False,
            "transfer_reason": "",
            "token_used": token_used,
            "quality_score": evaluation.score,
            "quality_evaluation": evaluation.to_dict(),
            "quality_alert": not evaluation.passed,
            "degraded": True,
            "degrade_reason": reason,
            "llm_trace": _offline_llm_trace(settings.llm_mode, "degraded", user_memories),
        },
    )
    record_chat_request(
        status="degraded",
        session_id=req.session_id,
        response_time_ms=response_time_ms,
        token_used=token_used,
        quality_score=evaluation.score,
    )
    return ChatResponse(
        session_id=req.session_id,
        user_id=req.user_id,
        answer=answer,
        needs_human_transfer=False,
        transfer_reason="",
        order_context=None,
        token_used=token_used,
        response_time_ms=response_time_ms,
        quality_score=evaluation.score,
        user_memories=user_memories,
        memory_summary=memory_summary,
        degraded=True,
        degrade_reason=reason,
        llm_trace=_offline_llm_trace(settings.llm_mode, "degraded", user_memories),
    )


def _degraded_answer(reason: str) -> str:
    if reason == "single_request_token_budget_exceeded":
        return (
            "当前问题内容超过单次 Token 预算，我先给出简化回复："
            "请缩短问题或拆成多轮咨询，我可以继续帮你查询订单、物流、商品库存或退款。"
        )
    return (
        "当前系统已达到全局 Token 预算，为控制成本先给出简化回复："
        "我可以继续帮你查询订单、物流、商品库存或退款；请稍后重试获取完整处理结果。"
    )


async def _maybe_generate_llm_answer(
    *,
    question: str,
    draft_answer: str,
    user_memories: list[str],
    result: dict,
) -> tuple[str, dict]:
    trace = _offline_llm_trace(settings.llm_mode, result.get("tool_name", ""), user_memories)
    if customer_service_llm_setup.startup_error:
        return draft_answer, {
            **trace,
            "reasoning_summary": f"LLM 未启用，已使用规则回答：{customer_service_llm_setup.startup_error}",
        }
    prompt_messages = _build_llm_messages(
        question=question,
        draft_answer=draft_answer,
        user_memories=user_memories,
        result=result,
    )
    try:
        metadata_result = await customer_service_llm.ainvoke_with_metadata(prompt_messages)
    except AttributeError:
        content = await customer_service_llm.ainvoke(prompt_messages)
        return str(content), {
            **trace,
            "used_llm": True,
            "model_used": "primary",
            "fallback_used": False,
            "reasoning_summary": _reasoning_summary(result.get("tool_name", ""), user_memories),
        }
    except Exception as exc:
        return draft_answer, {
            **trace,
            "used_llm": False,
            "model_used": "offline",
            "fallback_used": False,
            "reasoning_summary": f"LLM 调用失败，已使用规则回答：{_llm_error_message(exc)}",
        }

    return metadata_result.content, {
        **trace,
        "used_llm": True,
        "model_used": metadata_result.model_used,
        "fallback_used": metadata_result.fallback_used,
        "reasoning_summary": _reasoning_summary(result.get("tool_name", ""), user_memories),
    }


def _build_llm_messages(
    *,
    question: str,
    draft_answer: str,
    user_memories: list[str],
    result: dict,
) -> list[object]:
    context = {
        "tool_name": result.get("tool_name", ""),
        "order_context": result.get("order_context"),
        "needs_human_transfer": result.get("needs_human_transfer", False),
        "transfer_reason": result.get("transfer_reason", ""),
        "user_memories": user_memories,
        "draft_answer": draft_answer,
    }
    return [
        SystemMessage(
            content=(
                "你是电商智能客服。请基于后端工具结果回答用户，不要编造订单、物流或退款信息。"
                "如果没有订单号，只能说明会参考用户偏好，并提示提供订单号查询实际承运商。"
                "回答要简洁、礼貌、中文。"
            )
        ),
        HumanMessage(content=f"用户问题：{question}\n后端上下文：{context}"),
    ]


def _offline_llm_trace(mode: str, tool_name: str, user_memories: list[str] | None = None) -> dict:
    return {
        "mode": mode,
        "used_llm": False,
        "model_used": "offline",
        "fallback_used": False,
        "tool_name": tool_name,
        "reasoning_summary": _reasoning_summary(tool_name, list(user_memories or [])),
    }


def _reasoning_summary(tool_name: str, user_memories: list[str]) -> str:
    if tool_name == "delivery_preference":
        if user_memories:
            return "识别为无订单号快递咨询，已读取配送偏好，并提示提供订单号确认实际承运商。"
        return "识别为无订单号快递咨询，未找到配送偏好，已提示提供订单号。"
    if tool_name == "get_logistics":
        return "识别为物流查询，已调用物流工具获取承运商和配送状态。"
    if tool_name == "get_order":
        return "识别为订单查询，已调用订单工具获取订单状态。"
    if tool_name == "apply_refund":
        return "识别为退款请求，已按退款确认流程处理。"
    if tool_name == "save_user_memory":
        return "识别为偏好表达，已交由长期记忆保存。"
    if tool_name == "load_user_memory":
        return "识别为记忆召回，已读取用户长期记忆。"
    if tool_name == "human_transfer":
        return "识别为转人工或高风险诉求，已标记人工接手。"
    if tool_name == "degraded":
        return "请求进入降级路径，未调用 LLM。"
    return "使用规则客服路径生成回答。"


def _llm_error_message(exc: Exception) -> str:
    if customer_service_llm_setup.startup_error:
        return customer_service_llm_setup.startup_error
    return str(exc)
