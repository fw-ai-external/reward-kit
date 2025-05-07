from reward_kit import EvaluateResult, MetricResult, reward_function, Message
from typing import List


@reward_function
def evaluate(
    messages: List[Message], original_messages: List[Message] = list(), **kwargs
) -> EvaluateResult:
    """
    Evaluate a sample entry by counting words in the response.

    Args:
        messages: List of conversation messages
        original_messages: Original messages (usually without the response being evaluated)
        **kwargs: Additional parameters

    Returns:
        Dict with score and metrics information
    """
    # If this is the first message, there's nothing to evaluate
    if not messages:
        return EvaluateResult(score=0.0, reason="No messages found")

    # Get the last message (assistant's response)
    last_message = messages[-1]
    if last_message is not None and last_message.content is not None:
        content = last_message.content
    else:
        content = ""

    # Simple evaluation: count the number of words
    word_count = len(content.split())
    score = min(word_count / 100, 1.0)  # Cap at 1.0

    return EvaluateResult(
        score=score,
        reason=f"Word count: {word_count}",
        metrics={
            "word_count": MetricResult(
                score=score, reason=f"Word count: {word_count}"
            )
        },
    )
