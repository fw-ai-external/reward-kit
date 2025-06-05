from typing import Any, Dict, List, Optional

from reward_kit.models import EvaluateResult, MetricResult
from reward_kit.typed_interface import reward_function

__all__ = ["adapt_metric"]

try:
    from deepeval.metrics.base_metric import BaseMetric, BaseConversationalMetric
    from deepeval.test_case import LLMTestCase, ConversationalTestCase
except Exception:  # pragma: no cover - deepeval is optional
    BaseMetric = None
    BaseConversationalMetric = None
    LLMTestCase = None
    ConversationalTestCase = None


def _metric_name(metric: Any) -> str:
    name = getattr(metric, "__name__", None)
    if name and name not in {"Base Metric", "Base Conversational Metric", "Base Multimodal Metric"}:
        return str(name)
    name = getattr(metric, "name", None)
    if name:
        return str(name)
    return metric.__class__.__name__


def adapt_metric(metric: Any):
    """Adapt a deepeval metric object into a reward-kit reward function."""

    @reward_function
    def wrapped(
        messages: List[Dict[str, Any]],
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluateResult:
        if BaseMetric is None or LLMTestCase is None:
            raise ImportError(
                "deepeval must be installed to use this integration"
            )
        if not messages:
            return EvaluateResult(score=0.0, reason="No messages", metrics={})

        output = messages[-1].get("content", "")
        input_msg = ""
        if len(messages) >= 2:
            input_msg = messages[-2].get("content", "")

        if isinstance(metric, BaseConversationalMetric):
            turns = [
                LLMTestCase(
                    input=messages[i - 1].get("content", "") if i > 0 else "",
                    actual_output=msg.get("content", ""),
                )
                for i, msg in enumerate(messages)
            ]
            test_case = ConversationalTestCase(turns=turns)
        else:
            test_case = LLMTestCase(
                input=input_msg,
                actual_output=output,
                expected_output=ground_truth,
            )

        metric.measure(test_case, **kwargs)
        score = float(metric.score or 0.0)
        reason = getattr(metric, "reason", None)
        name = _metric_name(metric)
        metrics = {name: MetricResult(score=score, reason=reason or "", is_score_valid=True)}
        return EvaluateResult(score=score, reason=reason, metrics=metrics)

    return wrapped
