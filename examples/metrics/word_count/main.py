from typing import List

from reward_kit import EvaluateResult, Message, MetricResult, reward_function


@reward_function
def evaluate(messages: List[Message], **kwargs) -> EvaluateResult:
    """
    Evaluate a sample entry by counting words in the response.

    Args:
        messages: List of conversation messages, where the last message is the
                  assistant's response to be evaluated.
        **kwargs: Additional parameters

    Returns:
        EvaluateResult with score and metrics information
    """
    # If this is the first message, there's nothing to evaluate
    if not messages:
        return EvaluateResult(
            score=0.0, reason="No messages found", is_score_valid=False
        )

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
        is_score_valid=True,
        metrics={
            "word_count": MetricResult(
                score=score,
                is_score_valid=word_count
                > 0,  # Basic is_score_valid if there are any words
                reason=f"Word count: {word_count}",
            )
        },
    )
