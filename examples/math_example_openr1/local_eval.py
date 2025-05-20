import json
import os
import sys

# Ensure reward-kit is in the path
# This assumes the script is run from the 'examples/math_example_openr1/' directory or reward-kit is installed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from reward_kit.rewards.math import math_reward
from reward_kit.models import Message


def load_dataset(file_path: str):
    """Loads a JSONL dataset."""
    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def main():
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.jsonl")

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    dataset = load_dataset(dataset_path)
    all_passed = True
    total_samples = len(dataset)
    passed_samples = 0

    print(
        f"Starting local evaluation for OpenR1 Math Example using {dataset_path}...\n"
    )  # Modified print statement

    for i, item in enumerate(dataset):
        messages_data = item.get("messages")
        if not messages_data:
            print(f"Sample {i+1}: Skipping, 'messages' field is missing or empty.")
            all_passed = False
            continue

        # Convert message dicts to Message objects
        try:
            messages = [Message(**msg) for msg in messages_data]
        except TypeError as e:
            print(f"Sample {i+1}: Error creating Message objects - {e}. Skipping.")
            all_passed = False
            continue

        # The math_reward expects the full conversation history.
        # The last message is the assistant's response to be evaluated.
        # The ground_truth for math_reward should be the expected answer string.
        # For this example, the dataset's assistant message content is used as the ground_truth
        # to verify self-consistency of the dataset with the reward function.
        assistant_response_content = next(
            (m.content for m in messages if m.role == "assistant"), None
        )
        if assistant_response_content is None:
            print(
                f"Sample {i+1}: Skipping, no assistant message found for ground_truth."
            )
            all_passed = False
            print("---------------------\n")
            continue

        try:
            # Evaluate the assistant's message against itself (as ground_truth)
            result = math_reward(
                messages=messages,  # Contains the assistant message to be evaluated
                ground_truth=assistant_response_content,  # The content of that same assistant message
            )

            print(f"--- Sample {i+1} ---")
            user_query = next((m.content for m in messages if m.role == "user"), "N/A")
            assistant_response = assistant_response_content  # Already extracted
            print(f"User: {user_query}")
            print(f"Assistant: {assistant_response}")
            print(f"Score: {result.score}")
            print(f"Reason: {result.reason}")
            if result.metrics:
                print("Metrics:")
                for metric_name, metric_detail in result.metrics.items():
                    print(
                        f"  {metric_name}: Score={metric_detail.score}, Success={metric_detail.is_score_valid}, Reason='{metric_detail.reason}'"
                    )

            if result.score == 1.0:
                print("Status: PASSED")
                passed_samples += 1
            else:
                print("Status: FAILED")
                all_passed = False
            print("---------------------\n")

        except Exception as e:
            print(f"Sample {i+1}: Error during evaluation - {e}")
            all_passed = False
            print("---------------------\n")

    print("\n--- Evaluation Summary ---")
    print(f"Total samples evaluated: {total_samples}")
    print(f"Samples passed: {passed_samples}")
    print(f"Samples failed: {total_samples - passed_samples}")

    if all_passed and total_samples > 0:
        print("\nAll samples passed successfully!")
    else:
        print("\nSome samples failed or an error occurred.")


if __name__ == "__main__":
    main()
