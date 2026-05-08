from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QualityEvaluation:
    accuracy: int
    politeness: int
    completeness: int
    score: int
    passed: bool
    issues: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "politeness": self.politeness,
            "completeness": self.completeness,
            "score": self.score,
            "passed": self.passed,
            "issues": list(self.issues),
        }


class AutoQualityEvaluator:
    def __init__(self, *, alert_threshold: int = 70) -> None:
        self.alert_threshold = alert_threshold
        self.alert_events: list[dict[str, Any]] = []

    def evaluate(self, *, question: str, answer: str, context: dict[str, Any] | None = None) -> QualityEvaluation:
        context = dict(context or {})
        issues: list[str] = []
        accuracy = self._score_accuracy(question, answer, context, issues)
        politeness = self._score_politeness(answer, issues)
        completeness = self._score_completeness(question, answer, context, issues)
        score = round(accuracy * 0.4 + politeness * 0.3 + completeness * 0.3)
        evaluation = QualityEvaluation(
            accuracy=accuracy,
            politeness=politeness,
            completeness=completeness,
            score=score,
            passed=score >= self.alert_threshold,
            issues=issues,
        )
        if not evaluation.passed:
            self.trigger_alert(question=question, answer=answer, evaluation=evaluation)
        return evaluation

    def trigger_alert(self, *, question: str, answer: str, evaluation: QualityEvaluation) -> None:
        event = {
            "level": "warning",
            "message": f"客服回答质量低（{evaluation.score}分）",
            "question": question,
            "answer": answer,
            "score": evaluation.score,
            "issues": list(evaluation.issues),
        }
        self.alert_events.append(event)
        logger.warning(
            "quality_alert score=%s issues=%s question=%s",
            evaluation.score,
            evaluation.issues,
            question,
        )

    def _score_accuracy(
        self,
        question: str,
        answer: str,
        context: dict[str, Any],
        issues: list[str],
    ) -> int:
        lowered_answer = answer.lower()
        lowered_question = question.lower()
        order_context = context.get("order_context") or {}
        if _contains_any(answer, ["不知道", "不清楚", "无法回答"]):
            issues.append("回答未解决用户问题")
            return 35
        order_id = str(order_context.get("order_id") or "")
        if order_id and order_id not in answer:
            issues.append("回答缺少订单号")
            return 65
        if "refund_status" in order_context and "退款" not in answer:
            issues.append("回答缺少退款处理信息")
            return 65
        if "tracking_no" in order_context and not _contains_any(answer, ["物流", "运单", "承运", "状态"]):
            issues.append("回答缺少物流状态信息")
            return 68
        if "order" in lowered_question and not order_context and not _contains_any(lowered_answer, ["order", "订单"]):
            issues.append("回答未覆盖订单咨询意图")
            return 68
        return 90

    def _score_politeness(self, answer: str, issues: list[str]) -> int:
        if _contains_any(answer, ["闭嘴", "烦", "自己看", "不知道。"]):
            issues.append("客服语气不够友好")
            return 55
        if _contains_any(answer, ["请", "帮", "可以", "已为你", "我可以"]):
            return 90
        return 78

    def _score_completeness(
        self,
        question: str,
        answer: str,
        context: dict[str, Any],
        issues: list[str],
    ) -> int:
        if len(answer.strip()) < 8:
            issues.append("回答过短")
            return 45
        order_context = context.get("order_context") or {}
        if order_context and len(answer.strip()) < 20:
            issues.append("回答缺少关键上下文说明")
            return 65
        if _contains_any(question, ["退款", "refund"]) and "确认" not in answer and "已提交" not in answer:
            issues.append("退款场景缺少确认或提交状态")
            return 68
        return 88


def _contains_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)
