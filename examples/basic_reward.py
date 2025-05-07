"""
Example of a basic reward function using the Reward Kit.
"""

from typing import List, Dict, Optional, Any
from reward_kit import reward_function
from reward_kit.models import Message, EvaluateResult


@reward_function
def calculate_base_score(messages: List[Message], **kwargs) -> EvaluateResult:
    """
    Calculates a basic score based on length and keywords.

    Args:
        messages: List of conversation messages
        **kwargs: Additional arguments

    Returns:
        RewardOutput with score and metrics
    """
    # Get the last response (the one we're evaluating)
    last_response = messages[-1].content.lower()
    metrics = {}

    # Evaluate helpfulness
    is_helpful = "helpful" in last_response
    metrics["helpfulness"] = {
        "score": 0.5 if is_helpful else 0.0,
        "reason": (
            "Contains helpful keyword"
            if is_helpful
            else "Missing helpful keyword"
        ),
    }

    # Evaluate length
    is_long_enough = len(last_response) > 50
    metrics["length_bonus"] = {
        "score": 0.5 if is_long_enough else 0.0,
        "reason": (
            "Response length sufficient"
            if is_long_enough
            else "Response too short"
        ),
    }

    # Calculate final score (sum of component scores)
    final_score = sum(metric["score"] for metric in metrics.values())

    return EvaluateResult(score=final_score, metrics=metrics)


@reward_function
def calculate_safety_score(messages: List[Message], **kwargs) -> EvaluateResult:
    """
    Calculates a safety score (penalizes forbidden words).

    Args:
        messages: List of conversation messages
        **kwargs: Additional arguments

    Returns:
        RewardOutput with score and metrics
    """
    last_response = messages[-1].content.lower()
    metrics = {}

    # List of unsafe words (this would be much more comprehensive in practice)
    unsafe_words = ["harmful", "illegal", "dangerous", "unsafe", "violent"]

    # Check for unsafe words
    found_unsafe = [word for word in unsafe_words if word in last_response]

    if found_unsafe:
        reason = f"Contains unsafe word(s): {', '.join(found_unsafe)}"
        penalty = -1.0
    else:
        reason = "No unsafe content detected"
        penalty = 0.0

    metrics["safety_penalty"] = {"score": penalty, "reason": reason}

    # Safety score is just the penalty
    final_score = penalty

    return EvaluateResult(score=final_score, metrics=metrics)


@reward_function
def combined_reward(
    messages: List[Message], metadata: Optional[Dict[str, Any]] = None, **kwargs
) -> EvaluateResult:
    """
    Combines base score and safety score.

    Args:
        messages: List of conversation messages
        metadata: Optional metadata for customizing the reward
        **kwargs: Additional arguments

    Returns:
        RewardOutput with score and metrics
    """
    # Get component rewards
    base_output_dict = calculate_base_score(messages=messages, **kwargs)

    safety_output_dict = calculate_safety_score(messages=messages, **kwargs)

    # Extract scores and metrics
    base_score = base_output_dict["score"]
    base_metrics = base_output_dict["metrics"]

    safety_score = safety_output_dict["score"]
    safety_metrics = safety_output_dict["metrics"]

    # Combine metrics
    all_metrics = {**base_metrics, **safety_metrics}

    # Calculate final score (add base score and safety penalty)
    final_score = base_score + safety_score

    # Apply metadata modifiers if available
    if metadata and "boost_factor" in metadata:
        boost = float(metadata["boost_factor"])
        final_score *= boost
        all_metrics["boost_applied"] = {
            "score": 0.0,  # This doesn't affect the score, just documents the boost
            "reason": f"Applied boost factor of {boost}",
        }

    return EvaluateResult(score=final_score, metrics=all_metrics)


if __name__ == "__main__":
    # Example usage
    test_messages = [
        {"role": "user", "content": "Can you explain how to make a cake?"},
        {
            "role": "assistant",
            "content": "Sure, I'd be happy to explain how to make a basic cake! First, you'll need flour, sugar, eggs, butter, and baking powder. Mix the dry ingredients, then add the wet ingredients and mix until smooth. Pour into a greased pan and bake at 350°F for about 30 minutes.",
        },
    ]

    # Test the base reward
    base_result = calculate_base_score(messages=test_messages)
    print("Base Reward Result:")
    print(f"Score: {base_result['score']}")
    print("Metrics:")
    for name, metric in base_result["metrics"].items():
        print(f"  {name}: {metric['score']} - {metric['reason']}")
    print()

    # Test the safety reward
    safety_result = calculate_safety_score(messages=test_messages)
    print("Safety Reward Result:")
    print(f"Score: {safety_result['score']}")
    print("Metrics:")
    for name, metric in safety_result["metrics"].items():
        print(f"  {name}: {metric['score']} - {metric['reason']}")
    print()

    # Test the combined reward
    combined_result = combined_reward(
        messages=test_messages, metadata={"boost_factor": 1.2}
    )
    print("Combined Reward Result (with boost):")
    print(f"Score: {combined_result['score']}")
    print("Metrics:")
    for name, metric in combined_result["metrics"].items():
        print(f"  {name}: {metric['score']} - {metric['reason']}")
    print()

    # Deploy example (commented out in the example)
    # print("Deploying reward function...")
    # deployment_id = combined_reward.deploy(name="my-reward-function")
    # print(f"Deployed with ID: {deployment_id}")
