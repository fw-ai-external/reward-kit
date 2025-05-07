"""
Example demonstrating integration with HuggingFace datasets for evaluation preview.
"""

import os
import sys

# Ensure reward-kit is in the path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# Check for required environment variables
if not os.environ.get("FIREWORKS_API_KEY"):
    print("Warning: FIREWORKS_API_KEY environment variable is not set.")
    print(
        "Either set this variable or provide an auth_token when calling create_evaluation()."
    )
    print(
        "Example: FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY python examples/huggingface_dataset_example.py"
    )

# Import the evaluation functions
from reward_kit.evaluation import (
    huggingface_dataset_to_jsonl,
)
# Unused reward functions were removed


def main():
    # Example 1: Convert a HuggingFace dataset to JSONL for manual inspection
    print("Converting DeepSeek-ProverBench dataset to JSONL...")

    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "deepseek-ai/DeepSeek-ProverBench", split="train"
        )
        print(f"First sample: {dataset[0]}")
        prompt_key = "statement"
        response_key = "reference_solution"
    except ImportError:
        print(
            "Could not load datasets package. Install with: pip install 'reward-kit[deepseek]'"
        )
        return

    jsonl_file = huggingface_dataset_to_jsonl(
        dataset_name="deepseek-ai/DeepSeek-ProverBench",
        split="train",
        max_samples=5,
        prompt_key=prompt_key,
        response_key=response_key,
    )
    print(f"Dataset converted to JSONL file: {jsonl_file}")


if __name__ == "__main__":
    main()
