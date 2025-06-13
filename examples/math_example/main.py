"""
Math Evaluation Example

This example shows how to create a custom reward function for math problems
that reuses existing reward-kit functionality. Here we import and use the
built-in math reward function to evaluate mathematical reasoning.
"""

from typing import Any, Dict, List, Optional, Union

from reward_kit import EvaluateResult, reward_function
from reward_kit.models import Message

# Import the existing reward function from reward-kit
from reward_kit.rewards.math import math_reward


@reward_function
def evaluate_math_problem(
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[str] = None,
    **kwargs,
) -> EvaluateResult:
    """
    Evaluate math problem solving accuracy using numerical answer extraction.

    This function demonstrates how to reuse existing reward-kit functions
    while allowing for future customization if needed.

    Args:
        messages: The conversation messages including the math solution
        ground_truth: Expected answer for comparison
        **kwargs: Additional parameters (like tolerance)

    Returns:
        EvaluateResult with score and metrics
    """
    # For now, we directly use the built-in math evaluation function
    # In a real scenario, you might add preprocessing, custom logic, or
    # combine multiple reward functions here

    return math_reward(messages=messages, ground_truth=ground_truth, **kwargs)
