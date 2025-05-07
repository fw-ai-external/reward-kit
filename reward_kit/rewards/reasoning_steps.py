"""
Reward functions for evaluating reasoning steps.

This module provides reward functions that evaluate whether a model's response
contains adequate step-by-step reasoning, rewarding structured thinking.
"""

import re
from typing import Dict, List, Any, Union, Optional, Set, Pattern

from ..typed_interface import reward_function
from ..models import Message, EvaluateResult, MetricResult


@reward_function
def reasoning_steps_reward(
    messages: Union[List[Dict[str, Any]], List[Message]],
    pattern: Optional[str] = None,
    min_steps: int = 3,
    max_steps: Optional[int] = None,
    exclusive_patterns: bool = False,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Reward function that evaluates step-by-step reasoning in model responses.

    This function checks if the model's response contains indicators of structured
    reasoning, such as numbered steps, bullet points, or transitional phrases.

    Args:
        messages: List of conversation messages
        pattern: Optional custom regex pattern to use for detecting reasoning steps
        min_steps: Minimum number of steps required for full score
        max_steps: Optional maximum number of steps (default: None)
        exclusive_patterns: Whether to use only the custom pattern (True) or
                           combine it with default patterns (False)
        **kwargs: Additional arguments

    Returns:
        EvaluateResult with score based on the number of reasoning steps detected
    """
    # Get last message (the model's response)
    if not messages or len(messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            metrics={
                "reasoning_steps": MetricResult(
                    score=0.0, success=False, reason="No messages provided"
                )
            },
        )

    response = messages[-1]

    # Extract response text
    if isinstance(response, Message):
        if response.role != "assistant" or not response.content:
            return EvaluateResult(
                score=0.0,
                reason="No assistant response found",
                metrics={
                    "reasoning_steps": MetricResult(
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
                    "reasoning_steps": MetricResult(
                        score=0.0,
                        success=False,
                        reason="Message not from assistant or has no content",
                    )
                },
            )
        text = response.get("content", "")

    # Default patterns for detecting reasoning steps
    default_patterns = [
        r"Step\s+\d+[:.]\s+",  # "Step 1: " or "Step 1. "
        r"^\s*\d+\.\s+",  # Numbered list items at start of line like "1. "
        r"\n\s*\d+\.\s+",  # Numbered list items preceded by newline
        r"\n\s*-\s+",  # Bullet points with hyphens preceded by newline
        r"\n\s*\*\s+",  # Bullet points with asterisks preceded by newline
        r"\b(?:First|Second|Third|Fourth|Fifth|Next|Then|Finally)[,:]",  # Transition words
        r"\b(?:Let's|I will|To solve this|To begin)[,:]",  # Process indicators
    ]

    # Determine which patterns to use
    patterns_to_use = []
    if pattern and exclusive_patterns:
        # Use only the custom pattern
        patterns_to_use = [pattern]
    elif pattern:
        # Use both custom and default patterns
        patterns_to_use = [pattern] + default_patterns
    else:
        # Use only default patterns
        patterns_to_use = default_patterns

    # Combine patterns into a single regex
    combined_pattern = "|".join(f"(?:{p})" for p in patterns_to_use)

    # Find all matches
    matches = re.findall(combined_pattern, text, re.MULTILINE)
    num_steps = len(matches)

    # Calculate score based on number of steps found
    if num_steps == 0:
        score = 0.0
    elif max_steps is not None:
        # Scale between min_steps and max_steps
        score = min(
            1.0,
            max(0.0, (num_steps - min_steps + 1) / (max_steps - min_steps + 1)),
        )
    else:
        # Scale based on min_steps
        score = min(1.0, num_steps / min_steps)

    # Determine if the response has enough steps to be successful
    success = num_steps >= min_steps

    # Generate metrics for types of steps found
    step_metrics = {}

    # Check for explicit numbered steps (e.g., "Step 1:")
    explicit_steps = len(re.findall(r"Step\s+\d+[:.]\s+", text, re.MULTILINE))
    if explicit_steps > 0:
        step_metrics["explicit_steps"] = MetricResult(
            score=min(1.0, explicit_steps / min_steps),
            success=explicit_steps >= min_steps,
            reason=f"Found {explicit_steps} explicit steps",
        )

    # Check for numbered lists (e.g., "1. ")
    numbered_lists = len(re.findall(r"(?:^|\n)\s*\d+\.\s+", text, re.MULTILINE))
    if numbered_lists > 0:
        step_metrics["numbered_lists"] = MetricResult(
            score=min(1.0, numbered_lists / min_steps),
            success=numbered_lists >= min_steps,
            reason=f"Found {numbered_lists} numbered list items",
        )

    # Check for bullet points (e.g., "- " or "* ")
    bullets = len(re.findall(r"(?:^|\n)\s*[-*]\s+", text, re.MULTILINE))
    if bullets > 0:
        step_metrics["bullet_points"] = MetricResult(
            score=min(1.0, bullets / min_steps),
            success=bullets >= min_steps,
            reason=f"Found {bullets} bullet points",
        )

    # Check for transition phrases (e.g., "First", "Next", "Finally")
    transitions = len(
        re.findall(
            r"\b(?:First|Second|Third|Next|Then|Finally)[,:]",
            text,
            re.MULTILINE,
        )
    )
    if transitions > 0:
        step_metrics["transition_phrases"] = MetricResult(
            score=min(1.0, transitions / min_steps),
            success=transitions >= min_steps,
            reason=f"Found {transitions} transition phrases",
        )

    # Prepare overall metrics
    metrics = {
        "reasoning_steps": MetricResult(
            score=score,
            success=success,
            reason=f"Found {num_steps} reasoning steps (minimum required: {min_steps})",
        ),
        **step_metrics,
    }

    # Prepare detailed reason
    reason = f"Detected {num_steps} reasoning steps (required: {min_steps})"
    if max_steps:
        reason += f", max: {max_steps}"

    return EvaluateResult(score=score, reason=reason, metrics=metrics)


@reward_function
def sequence_reward(
    messages: Union[List[Dict[str, Any]], List[Message]],
    sequence_terms: Optional[List[str]] = None,
    min_matches: int = 3,
    case_sensitive: bool = False,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Reward function that evaluates sequential reasoning in model responses.

    This function checks if the model's response follows a specific sequence
    of reasoning steps or includes a minimum number of required terms in order.

    Args:
        messages: List of conversation messages
        sequence_terms: List of terms that should appear in sequence
        min_matches: Minimum number of sequence terms required for full score
        case_sensitive: Whether matching should be case-sensitive
        **kwargs: Additional arguments

    Returns:
        EvaluateResult with score based on sequence matching
    """
    # Get last message (the model's response)
    if not messages or len(messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            metrics={
                "sequence_reasoning": MetricResult(
                    score=0.0, success=False, reason="No messages provided"
                )
            },
        )

    response = messages[-1]

    # Extract response text
    if isinstance(response, Message):
        if response.role != "assistant" or not response.content:
            return EvaluateResult(
                score=0.0,
                reason="No assistant response found",
                metrics={
                    "sequence_reasoning": MetricResult(
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
                    "sequence_reasoning": MetricResult(
                        score=0.0,
                        success=False,
                        reason="Message not from assistant or has no content",
                    )
                },
            )
        text = response.get("content", "")

    # Default sequence terms if none provided
    if not sequence_terms:
        sequence_terms = [
            "First",
            "Second",
            "Third",
            "Fourth",
            "Fifth",
            "Next",
            "Then",
            "Finally",
            "Conclusion",
        ]

    # Prepare for matching
    found_terms = []
    last_position = -1

    # Case sensitivity handling
    if not case_sensitive:
        text = text.lower()
        sequence_terms = [term.lower() for term in sequence_terms]

    # Find terms in sequence
    for term in sequence_terms:
        position = text.find(term, last_position + 1)
        if position > last_position:
            found_terms.append(term)
            last_position = position

    # Calculate score based on number of sequential terms found
    num_matches = len(found_terms)
    score = min(1.0, num_matches / min_matches)
    success = num_matches >= min_matches

    # Prepare metrics
    metrics = {
        "sequence_reasoning": MetricResult(
            score=score,
            success=success,
            reason=f"Found {num_matches} sequential terms (minimum required: {min_matches})",
        ),
        "sequential_terms_found": MetricResult(
            score=score,
            success=success,
            reason=f"Sequential terms found: {', '.join(found_terms)}",
        ),
    }

    # Prepare reason
    reason = f"Detected {num_matches} sequential reasoning terms in order (required: {min_matches})"

    return EvaluateResult(score=score, reason=reason, metrics=metrics)
