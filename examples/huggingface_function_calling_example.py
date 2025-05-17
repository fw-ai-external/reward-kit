"""
Example demonstrating integration with HuggingFace datasets for function calling evaluation.
"""

import os
import sys

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Check for required environment variables
if not os.environ.get("FIREWORKS_API_KEY"):
    print("Warning: FIREWORKS_API_KEY environment variable is not set.")
    print(
        "Either set this variable or provide an auth_token when calling create_evaluation()."
    )
    print(
        "Example: FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY python examples/huggingface_function_calling_example.py"
    )

# Import the evaluation functions
# Unused evaluation functions were removed
from reward_kit.rewards.function_calling import composite_function_call_reward


def main():
    # Example 1: Convert a HuggingFace dataset to JSONL for manual inspection
    print("Converting Glaive-FC dataset to JSONL...")
    # Note: Replace with the actual dataset name, split, prompt_key, and response_key
    # jsonl_file = huggingface_dataset_to_jsonl(
    #     dataset_name="glaive-ai/glaive-function-calling",
    #     split="train",
    #     max_samples=5,
    #     prompt_key="prompt",
    #     response_key="response"
    # )
    # print(f"Dataset converted to JSONL file: {jsonl_file}")

    # Example 2: Evaluate a custom response against the Glaive-FC dataset using composite_function_call_reward
    print(
        "\nEvaluating a custom response against Glaive-FC example using composite_function_call_reward..."
    )

    # Replace with actual dataset loading and function call extraction logic
    # For demonstration purposes, we'll use a dummy function call and schema
    function_call = {
        "name": "get_weather",
        "arguments": '{"location": "San Francisco", "unit": "celsius"}',
    }
    expected_schema = {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "arguments": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
    }
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco?",
        },
        {
            "role": "assistant",
            "content": "I can help with that.",
            "function_call": function_call,
        },
    ]

    # Evaluate the function call using composite_function_call_reward
    result = composite_function_call_reward(
        messages=messages, expected_schema=expected_schema
    )

    print(f"Function call score: {result.score}")
    if result.metrics:
        print("Detailed metrics:")
        for metric_name, metric_value in result.metrics.items():
            print(f"  {metric_name}: {metric_value.score} - {metric_value.reason}")


if __name__ == "__main__":
    main()
