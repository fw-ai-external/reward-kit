"""
Example of regenerating responses using Fireworks Qwen3 model and evaluating them
for the function calling task.
"""

import json
import os
import sys
import time  # For potential rate limiting

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from reward_kit.evaluation import (  # Assuming this is the entry point for single evaluations
    evaluate_chat_messages,
)
from reward_kit.rewards.function_calling import (  # Our updated reward function
    composite_function_call_reward,
)

# We might need a Fireworks client or a generic HTTP client
# For now, let's define a placeholder for the generation logic
# from fireworks.client import Fireworks # Hypothetical client


# Placeholder for actual Fireworks API call for generation
def generate_with_fireworks_qwen3(prompt_messages: list, api_key: str) -> dict:
    """
    Placeholder function to simulate a call to Fireworks Qwen3 model.
    In a real scenario, this would use the Fireworks SDK or an HTTP client.
    """
    print(f"Simulating generation for: {prompt_messages[-1]['content'][:50]}...")
    # This should return a dictionary similar to an assistant's message,
    # potentially including 'tool_calls' or 'content' with tool calls.
    # For testing, let's return a dummy response that might include a tool call.
    # This dummy response should be structured like the 'assistant' message in the dataset.

    # Example: if the last user message asks for weather, simulate a weather tool call
    if "weather" in prompt_messages[-1]["content"].lower():
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps(
                            {"location": "San Francisco, CA", "unit": "celsius"}
                        ),
                    },
                }
            ],
        }
    # Default dummy response if no specific trigger
    return {
        "role": "assistant",
        "content": "This is a regenerated response from Qwen3 (simulated). No tool calls invoked for this query.",
        "tool_calls": [],
    }


def load_dataset(file_path: str) -> list:
    """Loads the dataset from a .jsonl file."""
    dataset = []
    with open(file_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def main():
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        print("Error: FIREWORKS_API_KEY environment variable is not set.")
        print(
            "This script requires FIREWORKS_API_KEY for regeneration with Fireworks Qwen3 model."
        )
        sys.exit(1)

    dataset_path = "examples/tool_calling_example/dataset.jsonl"
    dataset = load_dataset(dataset_path)

    total_evaluated = 0
    total_passed_regenerated = 0
    cumulative_score_regenerated = 0.0

    print(
        f"Regenerating responses using (simulated) Fireworks Qwen3 and evaluating dataset: {dataset_path}\n"
    )

    for i, item in enumerate(dataset):
        original_messages = item.get("messages", [])
        ground_truth_data = item.get("ground_truth")

        if not original_messages or original_messages[-1].get("role") == "assistant":
            print(
                f"Skipping item {i+1}: No user prompt to regenerate from or last message is assistant."
            )
            continue

        # Prepare prompt for Qwen3 (typically all messages up to the user's last turn)
        prompt_for_qwen3 = [
            msg for msg in original_messages if msg.get("role") != "assistant"
        ]
        if not prompt_for_qwen3 or prompt_for_qwen3[-1].get("role") != "user":
            print(f"Skipping item {i+1}: Last message in prompt is not from user.")
            continue

        print(f"--- Test Case {i+1} ---")
        print(f"Original User Query: {prompt_for_qwen3[-1].get('content')}")

        # Regenerate response using Fireworks Qwen3 (simulated)
        # In a real scenario, you'd handle potential API errors, rate limits, etc.
        try:
            regenerated_assistant_message = generate_with_fireworks_qwen3(
                prompt_for_qwen3, api_key
            )
            time.sleep(0.1)  # Small delay to simulate API call
        except Exception as e:
            print(f"Error during generation for item {i+1}: {e}")
            continue

        print(f"Regenerated Assistant: {regenerated_assistant_message}")

        # Combine original user messages with the new assistant generation for evaluation
        messages_for_evaluation = prompt_for_qwen3 + [regenerated_assistant_message]

        # Evaluate the regenerated response
        # The `composite_function_call_reward` will be used here.
        eval_result = composite_function_call_reward(
            messages=messages_for_evaluation,  # Contains the new regenerated message at the end
            ground_truth=ground_truth_data,
        )

        total_evaluated += 1
        cumulative_score_regenerated += eval_result.score
        if eval_result.score == 1.0:
            total_passed_regenerated += 1

        print(f"Regenerated Score: {eval_result.score}")
        print(f"Regenerated Reason: {eval_result.reason}")
        print("---------------------\n")

    if total_evaluated > 0:
        average_score = cumulative_score_regenerated / total_evaluated
        pass_rate = (total_passed_regenerated / total_evaluated) * 100
        print(f"\n--- Regeneration Evaluation Summary ---")
        print(f"Total Test Cases Regenerated & Evaluated: {total_evaluated}")
        print(f"Total Passed (Score 1.0): {total_passed_regenerated}")
        print(f"Pass Rate for Regenerated: {pass_rate:.2f}%")
        print(f"Average Score for Regenerated: {average_score:.4f}")
        print(f"------------------------------------")
    else:
        print("No test cases were regenerated and evaluated.")


if __name__ == "__main__":
    main()
