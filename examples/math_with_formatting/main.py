"""
Math with Formatting Evaluation Example

This example shows how to create a custom reward function for math problems
that evaluates both numerical accuracy and response formatting. Here we combine
the built-in math reward function with a custom format checker.
"""

import re
from typing import Any, Dict, List, Optional, Union

from reward_kit import EvaluateResult, MetricResult, reward_function
from reward_kit.models import Message

# Import the existing reward function from reward-kit
from reward_kit.rewards.math import math_reward


def check_think_answer_format(text: str) -> bool:
    """Check if text follows <think>...</think><answer>...</answer> format."""
    if not text:
        return False
    pattern = r"^<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>$"
    return bool(re.match(pattern, text.strip()))


@reward_function
def evaluate(
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[str] = None,
    **kwargs,
) -> EvaluateResult:
    """
    Evaluate math problem solving considering both accuracy and formatting.

    This function demonstrates how to combine multiple evaluation criteria:
    - Numerical accuracy using built-in math evaluation
    - Response format compliance using custom logic

    Args:
        messages: The conversation messages including the math solution
        ground_truth: Expected answer for comparison
        **kwargs: Additional parameters (like tolerance)

    Returns:
        EvaluateResult with combined score and detailed metrics
    """
    # Get the assistant's response
    assistant_message = messages[-1]
    if isinstance(assistant_message, dict):
        assistant_response = assistant_message.get("content", "")
    else:
        assistant_response = assistant_message.content or ""

    # Evaluate numerical accuracy using built-in function
    accuracy_result = math_reward(
        messages=messages, ground_truth=ground_truth, **kwargs
    )

    # Evaluate format compliance
    format_correct = check_think_answer_format(assistant_response)
    format_score = 1.0 if format_correct else 0.0

    # Combine scores (average of accuracy and format)
    combined_score = (accuracy_result.score + format_score) / 2.0

    # Create detailed metrics
    metrics = {
        "accuracy": MetricResult(
            score=accuracy_result.score,
            reason=f"Numerical accuracy: {accuracy_result.reason}",
            is_score_valid=True,
        ),
        "format": MetricResult(
            score=format_score,
            reason=f"Format compliance: {'correct' if format_correct else 'incorrect'} <think>...</think><answer>...</answer> structure",
            is_score_valid=True,
        ),
    }

    return EvaluateResult(
        score=combined_score,
        reason=f"Combined score: {combined_score:.2f} (accuracy: {accuracy_result.score:.2f}, format: {format_score:.2f})",
        metrics=metrics,
    )
