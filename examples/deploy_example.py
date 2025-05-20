"""
Example of deploying a reward function to Fireworks.

This example demonstrates how to create and deploy a reward function
that evaluates the informativeness of an assistant's response.
"""

import os
import sys
from typing import Dict, List

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Check for required environment variables
if not os.environ.get("FIREWORKS_API_KEY"):
    print("Warning: FIREWORKS_API_KEY environment variable is not set.")
    print("Either set this variable or provide an auth_token when calling deploy().")
    print(
        "Example: FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY python examples/deploy_example.py"
    )

from reward_kit import EvaluateResult, Message, MetricResult, reward_function

# get_authentication is removed, new auth functions are used by create_evaluation internally.
# from reward_kit.auth import get_fireworks_api_key, get_fireworks_account_id
from reward_kit.evaluation import create_evaluation
import tempfile
import shutil

# The reward function code as a string to write to a temporary file
INFORMATIVENESS_REWARD_CODE = """
from reward_kit import EvaluateResult, Message, MetricResult, reward_function
from typing import List # Ensure List is imported for the type hint

@reward_function
def evaluate( # Renamed from informativeness_reward to evaluate
    messages: List[Message],
    **kwargs,
) -> EvaluateResult:
    # This function is now named 'evaluate' as expected by the Evaluator class
    if not messages or messages[-1].role != "assistant":
        return EvaluateResult(
            score=0.0,
            reason="No assistant response found",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    success=False,
                    reason="No assistant response found",
                )
            },
        )

    response = (
        messages[-1].content if messages[-1].content is not None else ""
    )
    metrics = {}

    length = len(response)
    length_score = min(length / 1000.0, 1.0)
    metrics["length"] = MetricResult(
        score=length_score * 0.2,
        success=length_score > 0,
        reason=f"Response length: {length} chars",
    )

    specificity_markers = [
        "specifically", "in particular", "for example", "such as",
        "notably", "precisely", "exactly",
    ]
    marker_count = sum(
        1 for marker in specificity_markers if marker.lower() in response.lower()
    )
    marker_score = min(marker_count / 2.0, 1.0)
    metrics["specificity"] = MetricResult(
        score=marker_score * 0.3,
        success=marker_count > 0,
        reason=f"Found {marker_count} specificity markers",
    )

    content_words = [
        "information", "data", "analysis", "recommend", "solution",
        "approach", "technique", "method",
    ]
    word_count = len(response.split())
    content_word_count = sum(
        1 for word in content_words if word.lower() in response.lower()
    )

    if word_count > 0:
        density_score = min(
            content_word_count / (word_count / 20), 1.0
        )
    else:
        density_score = 0.0

    metrics["content_density"] = MetricResult(
        score=density_score * 0.5,
        success=density_score > 0.1,
        reason=f"Content density: {content_word_count} content words in {word_count} total words",
    )

    final_score = sum(metric.score for metric in metrics.values())
    overall_reason = "Evaluation based on length, specificity, and content density."
    if final_score > 0.7:
        overall_reason = "Response is informative."
    elif final_score < 0.3:
        overall_reason = "Response lacks informativeness."

    return EvaluateResult(score=final_score, reason=overall_reason, metrics=metrics)
"""

# Define the reward function here so it can be tested locally if needed,
# but for deployment, we'll use its code string.
# This requires importing List from typing for the annotation.
from typing import List


@reward_function
def informativeness_reward(
    messages: List[Message],
    **kwargs,
) -> EvaluateResult:
    """
    Evaluates the informativeness of an assistant response based on
    specificity markers and content density.
    """
    # Get the assistant's response
    # messages are List[Dict[str, str]] as per type hint, but decorator converts to List[Message]
    # However, the decorator in typed_interface.py passes List[Message] to the wrapped function.
    # So, messages[-1] here will be a Message object.
    if not messages or messages[-1].role != "assistant":  # Use attribute access
        return EvaluateResult(
            score=0.0,
            reason="No assistant response found",  # Added reason for EvaluateResult
            is_score_valid=False,
            metrics={
                "error": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="No assistant response found",  # success added
                )
            },
        )

    response = (
        messages[-1].content if messages[-1].content is not None else ""
    )  # Use attribute access
    metrics = {}

    # 1. Length check - reward concise but informative responses
    length = len(response)
    length_score = min(length / 1000.0, 1.0)  # Cap at 1000 chars
    metrics["length"] = MetricResult(
        score=length_score * 0.2,  # 20% weight
        is_score_valid=length_score > 0,  # Basic success if length > 0
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
        1 for marker in specificity_markers if marker.lower() in response.lower()
    )
    marker_score = min(marker_count / 2.0, 1.0)  # Cap at 2 markers
    metrics["specificity"] = MetricResult(
        score=marker_score * 0.3,  # 30% weight
        is_score_valid=marker_count > 0,  # Basic success if markers found
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

    metrics["content_density"] = MetricResult(
        score=density_score * 0.5,  # 50% weight
        is_score_valid=density_score > 0.1,  # Basic success if density is somewhat reasonable
        reason=f"Content density: {content_word_count} content words in {word_count} total words",
    )

    # Calculate final score as weighted sum of metrics
    final_score = sum(metric.score for metric in metrics.values())
    # Determine overall reason based on score
    overall_reason = "Evaluation based on length, specificity, and content density."
    if final_score > 0.7:
        overall_reason = "Response is informative."
    elif final_score < 0.3:
        overall_reason = "Response lacks informativeness."

    return EvaluateResult(score=final_score, reason=overall_reason, metrics=metrics, is_score_valid=final_score > 0.0)


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
    result = informativeness_reward(messages=test_messages)
    print("Informativeness Reward Result:")
    print(f"Score: {result.score}")
    print("Metrics:")
    for name, metric in result.metrics.items():
        print(f"  {name}: {metric.score} - {metric.reason}")
    print()

    return result


# Deploy the reward function to Fireworks
def deploy_to_fireworks():
    temp_metric_dir = None
    try:
        # Create a temporary directory for the metric
        temp_metric_dir = tempfile.mkdtemp(prefix="reward_kit_deploy_example_")
        metric_main_py = os.path.join(temp_metric_dir, "main.py")

        with open(metric_main_py, "w") as f:
            # Write the reward function code (defined above as a string) to main.py
            # Need to adjust imports within the string if they rely on relative paths
            # For this example, assume INFORMATIVENESS_REWARD_CODE is self-contained or uses absolute imports
            f.write(INFORMATIVENESS_REWARD_CODE)

        print(f"Created temporary metric file at {metric_main_py}")

        # Deploy using create_evaluation
        # Authentication is handled internally by create_evaluation
        print("Deploying 'informativeness-v1'...")
        evaluator_details = create_evaluation(
            evaluator_id="informativeness-v1",
            metric_folders=[f"informativeness={temp_metric_dir}"],
            display_name="Informativeness Reward (v1)",
            description="Evaluates response informativeness based on specificity and content density",
            force=True,  # Overwrite if already exists
        )

        if evaluator_details and "name" in evaluator_details:
            evaluation_id = evaluator_details[
                "name"
            ]  # The 'name' field usually contains the full path
            # Extract the simple ID if it's part of a path like 'accounts/.../evaluators/ID'
            if "/evaluators/" in evaluation_id:
                evaluation_id = evaluation_id.split("/evaluators/")[-1]

            print(f"Successfully deployed evaluator. Full details: {evaluator_details}")
            print(f"Evaluator ID: {evaluation_id}")

            # Show how to use the evaluation ID in a training job
            # Note: The endpoint structure might differ slightly based on API version.
            # This is a generic example.
            print("\nUse this in your RL training job (example endpoint):")
            print(
                f'firectl create rl-job --reward-endpoint "https://api.fireworks.ai/v1/accounts/YOUR_ACCOUNT_ID/evaluators/{evaluation_id}"'
            )
            return evaluation_id
        else:
            print(
                f"Failed to deploy evaluator or extract ID. Response: {evaluator_details}"
            )
            return None

    except Exception as e:
        print(f"Deployment error: {str(e)}")
        # Consider re-raising or logging more details if needed
        return None
    finally:
        if temp_metric_dir and os.path.exists(temp_metric_dir):
            print(f"Cleaning up temporary metric directory: {temp_metric_dir}")
            shutil.rmtree(temp_metric_dir)


if __name__ == "__main__":
    # First test the reward function locally
    print("Testing reward function locally...")
    # To test locally, the informativeness_reward function needs to be defined in this scope
    # The string version is for deployment. We'll use the directly defined one for local test.

    # Re-define or ensure informativeness_reward is available for local test
    # For simplicity, we assume the @reward_function decorated one above is used.
    # If INFORMATIVENESS_REWARD_CODE was the sole source, we'd need to exec it or similar.
    test_reward_function()

    # Deploy to Fireworks
    print("\nDeploying to Fireworks...")
    deploy_to_fireworks()
