# pylint: disable=all
"""
Example of deploying a reward function to Fireworks.

This example demonstrates how to create and deploy a reward function
that evaluates the informativeness of an assistant's response.
"""

import os
import sys
from typing import List

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Check for required environment variables
if not os.environ.get("FIREWORKS_API_KEY"):
    print("Warning: FIREWORKS_API_KEY environment variable is not set.")
    print("Either set this variable or provide an auth_token when calling deploy().")
    print("Example: FIREWORKS_API_KEY=... python examples/deploy_example.py")

# Import the evaluate function from the new location for local testing
from examples.informativeness_metric.main import (
    evaluate as informativeness_evaluate_function,
)

# Import Message for type hinting if needed, and other necessary components
from reward_kit import EvaluateResult, Message, MetricResult  # Ensure all are imported

# Import the deployment function
from reward_kit.evaluation import deploy_folder_evaluation


def test_reward_function():
    # Example messages - convert to Message objects for the test
    test_messages_data = [
        {"role": "user", "content": "Can you explain machine learning?"},
        {
            "role": "assistant",
            "content": (
                "Machine learning is a method of data analysis that automates "
                "analytical model building. Specifically, it uses algorithms that "
                "iteratively learn from data, allowing computers to find hidden "
                "insights without being explicitly programmed where to look. "
                "For example, deep learning is a type of machine learning that "
                "uses neural networks with many layers. Such approaches have "
                "revolutionized fields like computer vision and natural language "
                "processing."
            ),
        },
    ]
    test_messages_objects = [Message(**msg) for msg in test_messages_data]

    # Test the reward function
    # The imported function `informativeness_evaluate_function` is already decorated.
    result = informativeness_evaluate_function(messages=test_messages_objects)
    print("Informativeness Reward Result (Local Test):")
    print(f"Score: {result.score}")  # Access score via attribute
    print("Metrics:")
    # Access metrics via attribute, assuming MetricResult has score and reason
    if result.metrics:  # Check if metrics exist
        for (
            name,
            metric_obj,
        ) in result.metrics.items():  # result.metrics should be a dict of MetricResult
            print(f"  {name}: {metric_obj.score} - {metric_obj.reason}")
    print()
    return result


def deploy_to_fireworks():
    """Deploy the informativeness reward function to Fireworks AI."""
    try:
        # Authentication is typically handled by environment variables or ~/.fireworks/auth.ini
        print(
            "Attempting to deploy using credentials from environment or auth files..."
        )

        evaluator_folder_path = os.path.join(
            os.path.dirname(__file__), "informativeness_metric"
        )
        evaluator_id_name = "informativeness-metric-example-v1"

        # Deploy the reward function using deploy_folder_evaluation
        deployment_result = deploy_folder_evaluation(
            evaluator_id=evaluator_id_name,
            evaluator_folder=evaluator_folder_path,
            display_name="Informativeness Metric (Example V1)",
            description="Evaluates response informativeness based on specificity and content density.",
            force=True,  # Overwrite if already exists
        )

        deployed_evaluator_name = deployment_result.get("evaluator", {}).get(
            "name", evaluator_id_name
        )
        if (
            isinstance(deployed_evaluator_name, str)
            and "/evaluators/" in deployed_evaluator_name
        ):
            actual_id_to_use = deployed_evaluator_name.split("/evaluators/")[-1]
        else:
            actual_id_to_use = evaluator_id_name

        print(f"Deployment successful. Evaluator ID: {actual_id_to_use}")
        print(f"Full deployment result: {deployment_result}")

        print("\nUse this in your RL training job (example):")
        print(
            f'firectl create rl-job --reward-endpoint "https://api.fireworks.ai/v1/evaluations/{actual_id_to_use}"'
        )

        return actual_id_to_use

    except ValueError as e:
        print(f"Authentication or setup error: {str(e)}")
        print("Make sure you have proper Fireworks API credentials set up.")
        return None
    except ImportError as e:
        print(
            f"Import error: {str(e)}. Make sure reward-kit and its dependencies are installed."
        )
        return None
    except Exception as e:
        print(f"An unexpected error occurred during deployment: {str(e)}")
        return None


if __name__ == "__main__":
    print("Testing reward function locally...")
    test_reward_function()
    print("\nDeploying to Fireworks...")
    deploy_to_fireworks()
