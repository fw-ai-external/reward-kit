"""
Reward functions for evaluating repetition in model responses.

This module provides reward functions that penalize repetitive text in model responses,
encouraging more diverse and information-rich outputs.
"""

import re
from typing import Dict, List, Any, Union, Optional, Set, Callable, Tuple

from ..typed_interface import reward_function
from ..models import Message, EvaluateResult, MetricResult


def get_ngrams(
    text: str, n: int, language: str = "en"
) -> Tuple[List[Tuple[str, ...]], int]:
    """
    Extract n-grams from text based on language.

    Args:
        text: The text to extract n-grams from
        n: Size of the n-grams
        language: Language of the text (affects tokenization)

    Returns:
        Tuple of (list of n-grams, total n-gram count)
    """
    if language == "en":
        # For English, split on whitespace
        words = text.lower().split()
    elif language == "zh":
        # For Chinese, try to use jieba for better word segmentation
        try:
            import jieba

            words = list(jieba.cut(text))
        except ImportError:
            # Fall back to character-level segmentation if jieba is not available
            words = list(text)
    else:
        # For other languages, default to whitespace tokenization
        words = text.lower().split()

    # Generate n-grams
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(tuple(words[i : i + n]))

    return ngrams, len(ngrams)


@reward_function
def repetition_penalty_reward(
    messages: Union[List[Dict[str, Any]], List[Message]],
    ngram_size: int = 3,
    max_penalty: float = 0.5,
    language: str = "en",
    **kwargs: Any,
) -> EvaluateResult:  # Change back to EvaluateResult for correct typing
    """
    Reward function that penalizes repetitive text in model responses.

    This function computes repetition by examining unique n-grams in the response
    and penalizes texts with a high proportion of repeated phrases.

    Args:
        messages: List of conversation messages
        ngram_size: Size of n-grams to check for repetition
        max_penalty: Maximum penalty to apply for repetitive text
        language: Language of the text (affects tokenization)
        **kwargs: Additional arguments

    Returns:
        EvaluateResult with score penalizing repetition
    """
    # Get last message (the model's response)
    if not messages or len(messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            metrics={
                "repetition": MetricResult(
                    score=0.0, success=False, reason="No messages provided"
                )
            },
        )

    response = messages[-1]

    # Extract response text
    if isinstance(response, Message):
        if response.role != "assistant":
            return {
                "score": 0.0,
                "reason": "No assistant response found",
                "metrics": {
                    "repetition": {
                        "score": 0.0,
                        "success": False,
                        "reason": "Message not from assistant",
                    }
                },
            }
        text = response.content or ""  # Handle None content as empty string
    elif isinstance(response, dict):
        if response.get("role") != "assistant":
            return {
                "score": 0.0,
                "reason": "No assistant response found",
                "metrics": {
                    "repetition": {
                        "score": 0.0,
                        "success": False,
                        "reason": "Message not from assistant",
                    }
                },
            }
        text = response.get("content", "")

    # Empty response - no repetition to penalize
    if not text.strip():
        # For empty response, we return a perfect score since there's no repetition
        result = {
            "score": 1.0,
            "reason": "Empty response, no repetition to penalize",
            "metrics": {
                "repetition": {
                    "score": 1.0,
                    "success": True,
                    "reason": "Empty response",
                },
                "unique_ngram_ratio": {
                    "score": 1.0,
                    "success": True,
                    "reason": "Empty response",
                },
                "repetition_penalty": {
                    "score": 1.0,
                    "success": True,
                    "reason": "No penalty applied to empty response",
                },
            },
        }
        return result

    # Get n-grams from the response
    ngrams, total = get_ngrams(text, ngram_size, language)

    # Not enough tokens for the specified n-gram size
    if total < 1:
        return EvaluateResult(
            score=1.0,
            reason=f"Text too short for {ngram_size}-gram analysis",
            metrics={
                "repetition": MetricResult(
                    score=1.0,
                    success=True,
                    reason=f"Text too short for {ngram_size}-gram analysis",
                )
            },
        )

    # Count unique n-grams
    unique_ngrams = len(set(ngrams))

    # Calculate repetition ratio
    repetition_ratio = 1.0 - (unique_ngrams / total)

    # Calculate final score (normalize penalty to [0, 1] range)
    # Higher repetition ratio -> lower score
    penalty = repetition_ratio * max_penalty
    score = max(0.0, 1.0 - penalty)  # Ensure score is non-negative

    # Determine success based on repetition ratio threshold
    # Low repetition (high unique ratio) is success
    success = repetition_ratio < 0.2  # Less than 20% repetition is good

    # Prepare reason and metrics
    reason = f"Repetition ratio: {repetition_ratio:.2f}, Unique {ngram_size}-grams: {unique_ngrams}/{total}"

    metrics = {
        "repetition": MetricResult(score=score, success=success, reason=reason),
        "unique_ngram_ratio": MetricResult(
            score=1.0 - repetition_ratio,  # Higher is better
            success=success,
            reason=f"Unique {ngram_size}-gram ratio: {1.0 - repetition_ratio:.2f}",
        ),
        "repetition_penalty": MetricResult(
            score=1.0 - penalty,  # Inverse of penalty for consistency
            success=success,
            reason=f"Applied repetition penalty: {penalty:.2f}",
        ),
    }

    # Return a dict that has the same structure as EvaluateResult for testing compatibility
    result = {
        "score": score,
        "reason": reason,
        "metrics": {
            key: {
                "score": metric.score,
                "success": metric.success,
                "reason": metric.reason,
            }
            for key, metric in metrics.items()
        },
    }
    return result


@reward_function
def diversity_reward(
    messages: Union[List[Dict[str, Any]], List[Message]],
    ngram_sizes: List[int] = [1, 2, 3],
    weights: Optional[List[float]] = None,
    language: str = "en",
    **kwargs: Any,
) -> EvaluateResult:  # Change back to EvaluateResult for correct typing
    """
    Reward function that measures lexical diversity in model responses.

    This function computes diversity across multiple n-gram sizes and combines them
    into a weighted score to encourage varied vocabulary and phrasing.

    Args:
        messages: List of conversation messages
        ngram_sizes: List of n-gram sizes to evaluate
        weights: Optional list of weights for each n-gram size (normalized if provided)
        language: Language of the text (affects tokenization)
        **kwargs: Additional arguments

    Returns:
        EvaluateResult with score based on lexical diversity
    """
    # Get last message (the model's response)
    if not messages or len(messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            metrics={
                "diversity": MetricResult(
                    score=0.0, success=False, reason="No messages provided"
                )
            },
        )

    response = messages[-1]

    # Extract response text
    if isinstance(response, Message):
        if response.role != "assistant":
            return {
                "score": 0.0,
                "reason": "No assistant response found",
                "metrics": {
                    "diversity": {
                        "score": 0.0,
                        "success": False,
                        "reason": "Message not from assistant",
                    }
                },
            }
        text = response.content or ""  # Handle None content as empty string
    elif isinstance(response, dict):
        if response.get("role") != "assistant":
            return {
                "score": 0.0,
                "reason": "No assistant response found",
                "metrics": {
                    "diversity": {
                        "score": 0.0,
                        "success": False,
                        "reason": "Message not from assistant",
                    }
                },
            }
        text = response.get("content", "")

    # Empty response
    if not text.strip():
        return {
            "score": 0.0,
            "reason": "Empty response",
            "metrics": {
                "diversity": {
                    "score": 0.0,
                    "success": False,
                    "reason": "Empty response",
                }
            },
        }

    # Set default weights if not provided
    if weights is None:
        # Higher weights for larger n-grams (more important for diversity)
        weights = [0.2, 0.3, 0.5][: len(ngram_sizes)]

    # Ensure weights match the number of n-gram sizes
    if len(weights) != len(ngram_sizes):
        # Truncate or expand weights list as needed
        if len(weights) > len(ngram_sizes):
            weights = weights[: len(ngram_sizes)]
        else:
            # Fill with equal weights for missing values
            missing_weight = (1.0 - sum(weights)) / (
                len(ngram_sizes) - len(weights)
            )
            weights.extend([missing_weight] * (len(ngram_sizes) - len(weights)))

    # Normalize weights to sum to 1
    total_weight = sum(weights)
    if total_weight != 1.0:
        weights = [w / total_weight for w in weights]

    # Calculate diversity for each n-gram size
    diversity_scores = {}
    ratios = {}

    for size, weight in zip(ngram_sizes, weights):
        ngrams, total = get_ngrams(text, size, language)

        if total < 1:
            # Text too short for this n-gram size
            diversity_scores[f"ngram_{size}"] = 1.0
            ratios[f"ngram_{size}"] = 1.0
            continue

        unique_count = len(set(ngrams))
        ratio = unique_count / total

        diversity_scores[f"ngram_{size}"] = ratio * weight
        ratios[f"ngram_{size}"] = ratio

    # Calculate final weighted score
    final_score = sum(diversity_scores.values())

    # Determine success based on overall diversity
    success = final_score > 0.6  # Threshold can be adjusted

    # Prepare metrics for each n-gram size
    size_metrics = {}
    for size, ratio in ratios.items():
        size_metrics[size] = MetricResult(
            score=ratio,
            success=ratio > 0.7,  # Higher threshold for individual n-gram sizes
            reason=f"Diversity ratio for {size}: {ratio:.2f}",
        )

    # Prepare overall metrics
    metrics = {
        "diversity": MetricResult(
            score=final_score,
            success=success,
            reason=f"Overall weighted diversity score: {final_score:.2f}",
        ),
        **size_metrics,
    }

    # Return a dict that has the same structure as EvaluateResult for testing compatibility
    result = {
        "score": final_score,
        "reason": f"Lexical diversity score: {final_score:.2f}",
        "metrics": {
            key: {
                "score": metric.score,
                "success": metric.success,
                "reason": metric.reason,
            }
            for key, metric in metrics.items()
        },
    }
    return result
