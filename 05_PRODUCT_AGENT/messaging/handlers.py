from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from memory.long_term import UserMemoryManager
from memory.session_store import SessionStore
from messaging.events import RocketMQEvent
from monitoring.evaluator import AutoQualityEvaluator


class PostprocessEventHandler:
    def __init__(
        self,
        *,
        session_store: SessionStore,
        user_memory_manager: UserMemoryManager,
        quality_evaluator: AutoQualityEvaluator | None = None,
    ) -> None:
        self.session_store = session_store
        self.user_memory_manager = user_memory_manager
        self.quality_evaluator = quality_evaluator or AutoQualityEvaluator()

    def handle(self, event: RocketMQEvent) -> dict[str, str]:
        if event.event_type != "customer_service.postprocess_requested":
            return {"status": "ignored"}

        session_id = str(event.payload.get("session_id") or event.aggregate_id)
        user_id = str(event.payload.get("user_id") or "")
        question = str(event.payload.get("question") or "")
        answer = str(event.payload.get("answer") or "")
        loaded = self.session_store.load_session(session_id)
        metadata = dict((loaded or {}).get("metadata") or {})
        processed_ids = list(metadata.get("postprocess_event_ids") or [])
        if event.event_id in processed_ids:
            return {"status": "already_processed"}

        evaluation = self.quality_evaluator.evaluate(
            question=question,
            answer=answer,
            context={"postprocess_event_id": event.event_id},
        )
        memory_saved_count = self.user_memory_manager.save_from_turn(user_id, question, answer)
        processed_ids.append(event.event_id)
        metadata.update(
            {
                "quality_score": evaluation.score,
                "quality_evaluation": evaluation.to_dict(),
                "quality_alert": not evaluation.passed,
                "memory_saved_count": memory_saved_count,
                "postprocess_status": "processed",
                "postprocess_event_ids": processed_ids,
            }
        )
        messages = list((loaded or {}).get("messages") or [])
        if not messages:
            messages = [HumanMessage(content=question), AIMessage(content=answer)]
        self.session_store.save_session(
            session_id=session_id,
            user_id=user_id,
            messages=messages,
            metadata=metadata,
        )
        return {"status": "processed"}
