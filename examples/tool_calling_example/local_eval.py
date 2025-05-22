import json

# from reward_kit.models import Message
from typing import Any, Dict, List, Union

from reward_kit.rewards.function_calling import (  # Changed to direct import
    exact_tool_match_reward,
)


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Loads the dataset from a .jsonl file."""
    dataset = []
    with open(file_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def main():
    # Path relative to the root of the reward-kit-2 directory
    dataset_path = "examples/tool_calling_example/dataset.jsonl"
    dataset = load_dataset(dataset_path)

    total_evaluated = 0
    total_passed = 0
    cumulative_score = 0.0

    print(f"Evaluating dataset: {dataset_path} using exact_tool_match_reward\n")

    for i, item in enumerate(dataset):
        messages_data = item.get("messages", [])
        # ground_truth is expected to be a dict by exact_tool_match_reward
        ground_truth_data = item.get("ground_truth")

        if not messages_data:
            print(f"Skipping item {i+1}: No messages found.")
            continue

        # The composite_function_call_reward (now delegating to exact_tool_match_reward)
        # expects a list of messages and the ground_truth dict.

        print(f"--- Test Case {i+1} ---")
        eval_result = exact_tool_match_reward(  # Changed to direct call
            messages=messages_data,  # List of message dicts
            ground_truth=ground_truth_data,  # The ground_truth dict from dataset
        )

        total_evaluated += 1
        cumulative_score += eval_result.score
        if eval_result.score == 1.0:
            total_passed += 1

        print(f"Score: {eval_result.score}")
        print(f"Reason: {eval_result.reason}")
        print("---------------------\n")

    if total_evaluated > 0:
        average_score = cumulative_score / total_evaluated
        pass_rate = (total_passed / total_evaluated) * 100
        print(f"\n--- Evaluation Summary ---")
        print(f"Total Test Cases Evaluated: {total_evaluated}")
        print(f"Total Passed (Score 1.0): {total_passed}")
        print(f"Pass Rate: {pass_rate:.2f}%")
        print(f"Average Score: {average_score:.4f}")
        print(f"------------------------")
    else:
        print("No test cases were evaluated.")


if __name__ == "__main__":
    main()
