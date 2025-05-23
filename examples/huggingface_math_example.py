"""
Example demonstrating integration with HuggingFace datasets for math evaluation.
"""

import os
import sys
from pathlib import Path

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Check for required environment variables
if not os.environ.get("FIREWORKS_API_KEY"):
    print("Warning: FIREWORKS_API_KEY environment variable is not set.")
    print(
        "Either set this variable or provide an auth_token when calling create_evaluation()."
    )
    print(
        "Example: FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY python examples/huggingface_math_example.py"
    )

# Import the evaluation functions
from reward_kit.evaluation import huggingface_dataset_to_jsonl, preview_evaluation
from reward_kit.rewards.length import length_reward
from reward_kit.rewards.math import math_reward


def main():
    # Example 1: Convert a HuggingFace dataset to JSONL for manual inspection
    print("Converting OpenR1-Math-220k dataset to JSONL...")
    jsonl_file = huggingface_dataset_to_jsonl(
        dataset_name="open-r1/OpenR1-Math-220k",
        split="train",
        max_samples=5,
        prompt_key="problem",
        response_key="solution",
    )
    print(f"Dataset converted to JSONL file: {jsonl_file}")

    # Example 2: Evaluate a custom response against the OpenR1-Math-220k dataset using math_reward and length_reward
    print(
        "\nEvaluating a custom response against OpenR1-Math-220k problem using math_reward and length_reward..."
    )

    # Get a problem from the dataset for demonstration
    try:
        from datasets import load_dataset

        dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train")
        problem = dataset[0]["problem"]

        # Sample custom solution (simplified for illustration)
        custom_solution = "2 + 2 = 4"

        # Evaluate the solution using math_reward and length_reward
        messages = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": custom_solution},
        ]
        # The ground_truth for "2 + 2 = 4" is "4"
        math_result = math_reward(messages=messages, ground_truth="4")
        length_result = length_reward(messages=messages)

        print(f"Problem: {problem}")
        print(f"Problem: {problem}")
        print(f"Custom solution math score: {math_result['score']}")
        print(f"Custom solution length score: {length_result['score']}")
        if math_result.get("metrics"):
            print("Detailed math metrics:")
            for metric_name, metric_value in math_result["metrics"].items():
                print(f"  {metric_name}: {metric_value.score} - {metric_value.reason}")
        if length_result.get("metrics"):
            print("Detailed length metrics:")
            for metric_name, metric_value in length_result["metrics"].items():
                print(f"  {metric_name}: {metric_value.score} - {metric_value.reason}")

    except ImportError:
        print("Could not load datasets package. Install with: pip install 'datasets'")


if __name__ == "__main__":
    main()
