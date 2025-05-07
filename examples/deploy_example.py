"""
Example of deploying a reward function to Fireworks.

This example demonstrates how to create and deploy a reward function
that evaluates the informativeness of an assistant's response.
"""

import os
import sys
from typing import List, Dict

# Ensure reward-kit is in the path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# Check for required environment variables
if not os.environ.get("FIREWORKS_API_KEY"):
    print("Warning: FIREWORKS_API_KEY environment variable is not set.")
    print(
        "Either set this variable or provide an auth_token when calling deploy()."
    )
    print(
        "Example: FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY python examples/deploy_example.py"
    )

from reward_kit import legacy_reward_function, RewardOutput, MetricRewardOutput
from reward_kit.auth import get_authentication


@legacy_reward_function
def informativeness_reward(
    messages: List[Dict[str, str]],
    original_messages: List[Dict[str, str]],
    **kwargs,
) -> RewardOutput:
    """
    Evaluates the informativeness of an assistant response based on
    specificity markers and content density.
    """
    # Get the assistant's response
    if not messages or messages[-1].get("role") != "assistant":
        return RewardOutput(
            score=0.0,
            metrics={
                "error": MetricRewardOutput(
                    score=0.0, reason="No assistant response found"
                )
            },
        )

    response = messages[-1].get("content", "")
    metrics = {}

    # 1. Length check - reward concise but informative responses
    length = len(response)
    length_score = min(length / 1000.0, 1.0)  # Cap at 1000 chars
    metrics["length"] = MetricRewardOutput(
        score=length_score * 0.2,  # 20% weight
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
        1
        for marker in specificity_markers
        if marker.lower() in response.lower()
    )
    marker_score = min(marker_count / 2.0, 1.0)  # Cap at 2 markers
    metrics["specificity"] = MetricRewardOutput(
        score=marker_score * 0.3,  # 30% weight
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

    metrics["content_density"] = MetricRewardOutput(
        score=density_score * 0.5,  # 50% weight
        reason=f"Content density: {content_word_count} content words in {word_count} total words",
    )

    # Calculate final score as weighted sum of metrics
    final_score = sum(metric.score for metric in metrics.values())

    return RewardOutput(score=final_score, metrics=metrics)


# Test the reward function with example messages
def test_reward_function():
    # Example messages
    test_messages = [
        {"role": "user", "content": "Can you explain machine learning?"},
        {
            "role": "assistant",
            "content": "Machine learning is a method of data analysis that automates analytical model building. Specifically, it uses algorithms that iteratively learn from data, allowing computers to find hidden insights without being explicitly programmed where to look. For example, deep learning is a type of machine learning that uses neural networks with many layers. Such approaches have revolutionized fields like computer vision and natural language processing.",
        },
    ]

    # Test the reward function
    result = informativeness_reward(
        messages=test_messages, original_messages=[test_messages[0]]
    )
    print("Informativeness Reward Result:")
    print(f"Score: {result.score}")
    print("Metrics:")
    for name, metric in result.metrics.items():
        print(f"  {name}: {metric.score} - {metric.reason}")
    print()

    return result


# Deploy the reward function to Fireworks
def deploy_to_fireworks():
    try:
        # Get authentication from the auth module
        account_id, auth_token = get_authentication()

        # Display info about what we're using
        print(f"Using account ID: {account_id}")
        print(f"Using auth token (first 10 chars): {auth_token[:10]}...")

        # Deploy the reward function with force=True to overwrite if it exists
        evaluation_id = informativeness_reward.deploy(
            name="informativeness-v1",
            description="Evaluates response informativeness based on specificity and content density",
            account_id=account_id,
            auth_token=auth_token,
            force=True,  # Overwrite if already exists
        )
        print(f"Deployed evaluation with ID: {evaluation_id}")

        # Example of deploying with a custom provider
        custom_evaluation_id = informativeness_reward.deploy(
            name="informativeness-v1-anthropic",
            description="Informativeness evaluation using Claude model",
            account_id=account_id,
            auth_token=auth_token,
            force=True,  # Overwrite if already exists
            providers=[
                {
                    "providerType": "anthropic",
                    "modelId": "claude-3-sonnet-20240229",
                }
            ],
        )
        print(
            f"Deployed evaluation with custom provider: {custom_evaluation_id}"
        )

        # Show how to use the evaluation ID in a training job
        print("Use this in your RL training job:")
        print(
            f'firectl create rl-job --reward-endpoint "https://api.fireworks.ai/v1/evaluations/{evaluation_id}"'
        )

        return evaluation_id

    except ValueError as e:
        print(f"Authentication error: {str(e)}")
        print("Make sure you have proper Fireworks API credentials set up.")
        return None


if __name__ == "__main__":
    # First test the reward function locally
    print("Testing reward function locally...")
    test_reward_function()

    # Deploy to Fireworks
    print("\nDeploying to Fireworks...")
    deploy_to_fireworks()
