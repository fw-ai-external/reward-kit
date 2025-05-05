"""
Utility functions for integrating reward-kit reward functions with TRL.

This module provides helper functions for:
1. Converting reward-kit reward functions to TRL-compatible format
2. Combining multiple reward functions with weights
3. Creating GRPO-specific format rewards
"""

import os
import sys
import re
from typing import List, Dict, Any, Optional, Union, Callable

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import reward-kit components
from reward_kit.reward_function import RewardFunction, reward_function
from reward_kit.models import RewardOutput, MetricRewardOutput, EvaluateResult, MetricResult


def create_combined_reward(
    reward_functions: List[RewardFunction],
    weights: Optional[List[float]] = None,
    normalize: bool = True
) -> Callable:
    """
    Combine multiple reward functions with optional weights.
    
    Args:
        reward_functions: List of RewardFunction instances
        weights: Optional weights for each reward function
        normalize: Whether to normalize weights to sum to 1.0
        
    Returns:
        A callable function compatible with TRL
    """
    # Validate inputs
    if len(reward_functions) == 0:
        raise ValueError("Must provide at least one reward function")
    
    # Normalize weights if provided
    if weights:
        if len(weights) != len(reward_functions):
            raise ValueError("Number of weights must match number of reward functions")
        if normalize:
            weight_sum = sum(weights)
            if weight_sum != 1.0:
                weights = [w / weight_sum for w in weights]
    else:
        # Equal weights for all reward functions
        weights = [1.0 / len(reward_functions) for _ in range(len(reward_functions))]
    
    # Create adapters for each reward function
    adapters = [rf.get_trl_adapter() for rf in reward_functions]
    
    def combined_adapter(batch_input, batch_orig_input=None, **adapter_kwargs):
        """Combined adapter function that works with TRL."""
        # Collect scores from all reward functions
        all_scores = []
        for i, adapter in enumerate(adapters):
            scores = adapter(batch_input, batch_orig_input, **adapter_kwargs)
            all_scores.append(scores)
        
        # Combine weighted scores for each sample
        combined_scores = []
        for i in range(len(all_scores[0])):
            weighted_sum = sum(scores[i] * weight for scores, weight in zip(all_scores, weights))
            combined_scores.append(weighted_sum)
        
        return combined_scores
    
    return combined_adapter


@reward_function
def grpo_format_reward(
    messages: List[Dict[str, Any]], 
    original_messages: Optional[List[Dict[str, Any]]] = None,
    think_tag: str = "<think>",
    answer_tag: str = "<answer>",
    **kwargs
) -> EvaluateResult:
    """
    Reward function that checks if the completion has the GRPO specific format.
    
    Args:
        messages: List of conversation messages
        original_messages: Original messages for context
        think_tag: Tag to use for reasoning (default: "<think>")
        answer_tag: Tag to use for answers (default: "<answer>")
        
    Returns:
        EvaluateResult with score based on format compliance
    """
    # Get the assistant's message
    if not messages or len(messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            metrics={"format": MetricResult(score=0.0, success=False, reason="No messages provided")}
        )
    
    # Extract response text from last message (assistant's response)
    response = messages[-1]
    if response.get("role") != "assistant" or not response.get("content"):
        return EvaluateResult(
            score=0.0,
            reason="No assistant response found",
            metrics={"format": MetricResult(score=0.0, success=False, reason="No assistant response")}
        )
    
    text = response.get("content", "")
    
    # Check for think/answer tags
    think_pattern = f"{re.escape(think_tag)}(.*?){re.escape(think_tag.replace('<', '</'))}"
    answer_pattern = f"{re.escape(answer_tag)}(.*?){re.escape(answer_tag.replace('<', '</'))}"
    
    think_match = re.search(think_pattern, text, re.DOTALL)
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    
    has_think = bool(think_match)
    has_answer = bool(answer_match)
    
    # Check for correct order (think should come before answer)
    correct_order = True
    if has_think and has_answer:
        think_pos = text.find(think_tag)
        answer_pos = text.find(answer_tag)
        correct_order = think_pos < answer_pos
    
    # Calculate score based on format compliance
    if has_think and has_answer and correct_order:
        score = 1.0
        reason = "Format is compliant with think/answer tags in correct order"
    elif has_think and has_answer:
        score = 0.5
        reason = "Has both think and answer tags but in incorrect order"
    elif has_think:
        score = 0.3
        reason = "Has think tag but missing answer tag"
    elif has_answer:
        score = 0.2
        reason = "Has answer tag but missing think tag"
    else:
        score = 0.0
        reason = "Missing both think and answer tags"
    
    # Create metrics
    metrics = {
        "has_think": MetricResult(
            score=1.0 if has_think else 0.0,
            success=has_think,
            reason=f"{'Has' if has_think else 'Missing'} think tag"
        ),
        "has_answer": MetricResult(
            score=1.0 if has_answer else 0.0,
            success=has_answer,
            reason=f"{'Has' if has_answer else 'Missing'} answer tag"
        ),
        "correct_order": MetricResult(
            score=1.0 if correct_order else 0.0,
            success=correct_order,
            reason=f"Tags are in {'correct' if correct_order else 'incorrect'} order"
        )
    }
    
    return EvaluateResult(
        score=score,
        reason=reason,
        metrics=metrics
    )


def create_grpo_reward(
    content_reward: RewardFunction,
    format_weight: float = 0.3,
    content_weight: float = 0.7,
    think_tag: str = "<think>",
    answer_tag: str = "<answer>"
) -> Callable:
    """
    Create a combined reward function for GRPO-style training.
    
    Args:
        content_reward: RewardFunction for content quality (accuracy, helpfulness, etc.)
        format_weight: Weight for format compliance (default: 0.3)
        content_weight: Weight for content quality (default: 0.7)
        think_tag: Tag to use for reasoning (default: "<think>")
        answer_tag: Tag to use for answers (default: "<answer>")
        
    Returns:
        A callable function compatible with GRPO
    """
    # Create format reward function
    format_rf = RewardFunction(
        func=grpo_format_reward,
        think_tag=think_tag,
        answer_tag=answer_tag
    )
    
    # Combine rewards
    return create_combined_reward(
        reward_functions=[format_rf, content_reward],
        weights=[format_weight, content_weight],
        normalize=True
    )


def prepare_grpo_message_format(
    text: str,
    system_prompt: str = None
) -> List[Dict[str, str]]:
    """
    Convert a text response to a message format for GRPO evaluation.
    
    Args:
        text: The model's text response
        system_prompt: Optional system prompt for context
        
    Returns:
        List of messages in the format expected by reward functions
    """
    messages = []
    
    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Add the response as assistant message
    messages.append({"role": "assistant", "content": text})
    
    return messages


def apply_reward_to_responses(
    reward_function: Union[RewardFunction, Callable],
    responses: List[str],
    system_prompt: str = None
) -> List[float]:
    """
    Apply a reward function to a list of text responses.
    
    Args:
        reward_function: RewardFunction or callable
        responses: List of model response strings
        system_prompt: Optional system prompt to include
        
    Returns:
        List of reward scores
    """
    # Convert responses to message format
    message_batches = [
        prepare_grpo_message_format(response, system_prompt) for response in responses
    ]
    
    # Check if we need to get the adapter
    if isinstance(reward_function, RewardFunction):
        adapter = reward_function.get_trl_adapter()
    else:
        # Assume it's already an adapter
        adapter = reward_function
    
    # Apply the adapter
    return adapter(message_batches)


# Example usage
if __name__ == "__main__":
    # Test the functions with a simple example
    from reward_kit.rewards.length import length_reward
    
    # Create a length reward function
    length_rf = RewardFunction(func=length_reward)
    
    # Create a format reward function
    format_rf = RewardFunction(func=grpo_format_reward)
    
    # Combine them
    combined_reward = create_combined_reward(
        reward_functions=[format_rf, length_rf],
        weights=[0.4, 0.6]
    )
    
    # Create a GRPO-style reward
    grpo_reward = create_grpo_reward(length_rf)
    
    # Test with some example responses
    test_responses = [
        "<think>This is my reasoning</think><answer>This is my answer</answer>",
        "This is a response without tags",
        "<answer>Answer first</answer><think>Think second</think>"
    ]
    
    # Apply rewards
    format_scores = apply_reward_to_responses(format_rf, test_responses)
    length_scores = apply_reward_to_responses(length_rf, test_responses)
    combined_scores = apply_reward_to_responses(combined_reward, test_responses)
    grpo_scores = apply_reward_to_responses(grpo_reward, test_responses)
    
    # Print results
    print("Format scores:", format_scores)
    print("Length scores:", length_scores)
    print("Combined scores:", combined_scores)
    print("GRPO scores:", grpo_scores)