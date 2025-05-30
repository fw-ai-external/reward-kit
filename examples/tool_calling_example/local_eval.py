import json
import logging
import os

# from reward_kit.models import Message
from typing import Any, Dict, List, Union

import hydra
from omegaconf import DictConfig, OmegaConf

from reward_kit.rewards.function_calling import (  # Changed to direct import
    exact_tool_match_reward,
)

logger = logging.getLogger(__name__)


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Loads the dataset from a .jsonl file."""
    dataset = []
    if not os.path.exists(file_path):
        # Use print here as logger might not be configured if script is run directly without Hydra context
        print(f"Error: Dataset file not found at {file_path}")
        return []
    with open(file_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


@hydra.main(config_path="conf", config_name="local_eval_config", version_base=None)
def main(cfg: DictConfig) -> None:
    dataset_path = hydra.utils.to_absolute_path(cfg.dataset_file_path)

    # Using print for initial messages as Hydra logger might not be fully set up or for emphasis
    print(f"Hydra configuration:\n{OmegaConf.to_yaml(cfg)}")
    print(f"Resolved dataset path: {dataset_path}")

    dataset = load_dataset(dataset_path)
    if (
        not dataset
    ):  # Check if dataset loading failed (e.g. file not found by load_dataset)
        print(
            f"Failed to load dataset from {dataset_path} or dataset is empty. Exiting."
        )
        return

    total_evaluated = 0
    total_passed = 0
    cumulative_score = 0.0

    print(f"Evaluating dataset: {dataset_path} using exact_tool_match_reward\n")
    # logger.info(f"Evaluating dataset: {dataset_path} using exact_tool_match_reward")

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
