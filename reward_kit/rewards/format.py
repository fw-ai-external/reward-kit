"""
Reward functions for validating text format.

This module provides reward functions that validate if text responses
adhere to specific formatting requirements, such as containing specific tags
in the correct order.
"""

import re
from typing import Dict, List, Any, Union, Optional # Added Optional

from ..typed_interface import reward_function
from ..models import Message, EvaluateResult, MetricResult


@reward_function
def format_reward(
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[Union[List[Message], List[Dict[str, Any]]]] = None, # Not used by this function but part of standard signature
    format_regex: str = r"^<think>\n.*?</think>\n<answer>\n.*?</answer>$",
    require_exact_match: bool = True,
    **kwargs: Any
) -> EvaluateResult:
    """
    Reward function that validates if text follows a specific format pattern.
    The model's response is assumed to be the last message in the `messages` list.

    By default, this checks for <think> and <answer> tags in the correct order,
    ensuring proper separation of reasoning and final answer.

    Args:
        messages: List of conversation messages, where `messages[-1]` is the model's response.
        ground_truth: Optional. Expected assistant response trajectory. Not directly used by this format reward.
        format_regex: Regular expression pattern to match. Default checks for
                      <think>...</think> followed by <answer>...</answer>.
        require_exact_match: If True, the entire text must match the pattern.
                             If False, pattern just needs to be found in text.
        **kwargs: Additional arguments.

    Returns:
        EvaluateResult with score 1.0 if format is correct, 0.0 otherwise
    """
    # Get last message (the model's response)
    if not messages or len(messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            metrics={
                "format_check": MetricResult(
                    score=0.0, success=False, reason="No messages provided"
                )
            },
        )

    response = messages[-1]

    # Check if it's an assistant message with content
    if isinstance(response, Message):
        if response.role != "assistant" or not response.content:
            return EvaluateResult(
                score=0.0,
                reason="No assistant response found",
                metrics={
                    "format_check": MetricResult(
                        score=0.0,
                        success=False,
                        reason="Message not from assistant or has no content",
                    )
                },
            )
        text = response.content
    elif isinstance(response, dict):
        if response.get("role") != "assistant" or not response.get("content"):
            return EvaluateResult(
                score=0.0,
                reason="No assistant response found",
                metrics={
                    "format_check": MetricResult(
                        score=0.0,
                        success=False,
                        reason="Message not from assistant or has no content",
                    )
                },
            )
        text = response.get("content", "")
    else: # Should not happen if messages contains dict or Message, but to be safe / satisfy linters
        return EvaluateResult(
            score=0.0,
            reason="Last message is of unexpected type.",
            metrics={
                "format_check": MetricResult(
                    score=0.0, success=False, reason="Invalid message type in messages."
                )
            },
        )

    # Compile the regex with DOTALL flag to match across newlines
    pattern = re.compile(format_regex, re.DOTALL)

    # Check if the text matches the pattern
    if require_exact_match:
        match = pattern.match(text)
    else:
        match = pattern.search(text)

    if match:
        return EvaluateResult(
            score=1.0,
            reason="Format is correct",
            metrics={
                "format_check": MetricResult(
                    score=1.0,
                    success=True,
                    reason="Text follows the required format pattern",
                )
            },
        )
    else:
        return EvaluateResult(
            score=0.0,
            reason="Format is incorrect",
            metrics={
                "format_check": MetricResult(
                    score=0.0,
                    success=False,
                    reason="Text does not follow the required format pattern",
                )
            },
        )
