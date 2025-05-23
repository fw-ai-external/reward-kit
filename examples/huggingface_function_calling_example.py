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

import json  # Added import
from typing import Any, Dict, List, cast  # Moved imports to top

# Import the evaluation functions
from reward_kit.rewards.function_calling import (  # Changed import
    exact_tool_match_reward,
)


def main():
    # Example 1: Convert a HuggingFace dataset to JSONL for manual inspection
    # This part remains commented out as it's beyond the scope of this update
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

    # Example 2: Evaluate a custom response using exact_tool_match_reward
    print("\nEvaluating a custom response using exact_tool_match_reward...")

    # For demonstration purposes, we'll use a dummy tool call
    # and construct the messages and ground_truth accordingly.

    # Define the tool call the assistant is supposed to make
    assistant_tool_call_function = {
        "name": "get_weather",
        "arguments": json.dumps({"location": "San Francisco", "unit": "celsius"}),
    }

    assistant_tool_call = {
        "id": "call_sfo_weather_123",  # Example ID
        "type": "function",
        "function": assistant_tool_call_function,
    }

    messages = [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco?",
        },
        {
            "role": "assistant",
            "content": None,  # Content can be None if only tool_calls are present
            "tool_calls": [assistant_tool_call],  # Use tool_calls list
        },
    ]

    # Define the ground truth for the expected tool call
    ground_truth_data = {
        "tool_calls": [
            {
                # Note: The 'id' field in tool_calls is not compared by exact_tool_match_reward,
                # so it doesn't strictly need to match the one in the assistant's message.
                # However, 'type' and 'function' (including name and arguments) must match.
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps(
                        {"location": "San Francisco", "unit": "celsius"}
                    ),
                },
            }
        ]
    }

    # Evaluate the tool call using exact_tool_match_reward
    result = exact_tool_match_reward(
        messages=cast(List[Dict[str, Any]], messages), ground_truth=ground_truth_data
    )

    print(f"Tool call evaluation score: {result.score}")
    if result.metrics:
        print("Detailed metrics:")
        for metric_name, metric_value in result.metrics.items():
            print(f"  {metric_name}: {metric_value.score} - {metric_value.reason}")


if __name__ == "__main__":
    main()
