from typing import Any, Dict, List

from reward_kit import EvaluateResult, Message, MetricResult, reward_function


@reward_function
def evaluate(  # Renamed from informativeness_reward
    messages: List[Message],
    **kwargs: Any,  # Added **kwargs to match typical signature for deployed functions
) -> EvaluateResult:
    """
    Evaluates the informativeness of an assistant response based on
    specificity markers and content density.
    """
    if not messages or messages[-1].role != "assistant":
        return EvaluateResult(
            score=0.0,
            reason="No assistant response found",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    success=False,
                    reason="No assistant response found",
                )
            },
        )

    response = messages[-1].content if messages[-1].content is not None else ""
    metrics = {}

    # 1. Length check - reward concise but informative responses
    length = len(response)
    length_score = min(length / 1000.0, 1.0)  # Cap at 1000 chars
    metrics["length"] = MetricResult(
        score=length_score * 0.2,  # 20% weight
        success=length_score > 0,
        reason=f"Response length: {length} chars",
    )

    # 2. Specificity markers
    specificity_markers = [
        "specifically",
        "in particular",
        "for example",
        "such as",
        "notably",
        "precisely",
        "exactly",
    ]
    marker_count = sum(
        1 for marker in specificity_markers if marker.lower() in response.lower()
    )
    marker_score = min(marker_count / 2.0, 1.0)  # Cap at 2 markers
    metrics["specificity"] = MetricResult(
        score=marker_score * 0.3,  # 30% weight
        success=marker_count > 0,
        reason=f"Found {marker_count} specificity markers",
    )

    # 3. Content density (simple heuristic based on ratio of content words to total)
    content_words = [
        "information",
        "data",
        "analysis",
        "recommend",
        "solution",
        "approach",
        "technique",
        "method",
    ]
    word_count = len(response.split())
    content_word_count = sum(
        1 for word in content_words if word.lower() in response.lower()
    )

    if word_count > 0:
        density_score = min(
            content_word_count / (word_count / 20), 1.0
        )  # Normalize by expecting ~5% density
    else:
        density_score = 0.0

    metrics["content_density"] = MetricResult(
        score=density_score * 0.5,  # 50% weight
        success=density_score > 0.1,
        reason=f"Content density: {content_word_count} content words in {word_count} total words",
    )

    # Calculate final score as weighted sum of metrics
    final_score = sum(metric.score for metric in metrics.values())
    # Determine overall reason based on score
    overall_reason = "Evaluation based on length, specificity, and content density."
    if final_score > 0.7:
        overall_reason = "Response is informative."
    elif final_score < 0.3:
        overall_reason = "Response lacks informativeness."

    return EvaluateResult(score=final_score, reason=overall_reason, metrics=metrics)
