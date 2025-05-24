import logging
import os
import shutil
import sys
import tempfile
import types  # Import the types module

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from reward_kit.common_utils import load_jsonl
from reward_kit.evaluation import preview_evaluation

# math_reward will be imported inside the dynamically created main.py
from reward_kit.models import Message

# Configure basic logging if you want to see logs from load_jsonl
# logging.basicConfig(level=logging.INFO)

# Removed local load_dataset_for_preview function


def main():
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.jsonl")

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    if (
        not os.environ.get("FIREWORKS_API_KEY")
        and os.environ.get("TEST_MOCK_FIREWORKS_PREVIEW") != "true"
    ):
        print("Error: FIREWORKS_API_KEY environment variable is not set.")
        print("Please set this variable to your Fireworks API key or run in mock mode.")
        print("Example: export FIREWORKS_API_KEY='your_api_key_here'")
        print("To run in mock mode: export TEST_MOCK_FIREWORKS_PREVIEW='true'")
        return

    dataset = load_jsonl(dataset_path)

    print(
        f"Starting Fireworks Preview API evaluation for OpenR1 Math Example using {dataset_path}...\n"  # Modified
    )

    try:
        if os.environ.get("TEST_MOCK_FIREWORKS_PREVIEW") == "true":
            print("Mocking Fireworks Preview API call in test mode.")
            # Simulate a successful response structure that the script expects
            from reward_kit.evaluation import EvaluatorPreviewResult  # For structure

            results = EvaluatorPreviewResult()
            results.total_samples = len(dataset)
            results.total_runtime_ms = 42  # Mocked runtime

            for i, sample_data in enumerate(dataset):
                mock_metric_detail = types.SimpleNamespace(
                    score=1.0, reason="Mocked metric success", is_score_valid=True
                )
                # The key for per_metric_evals should match the metric_name used below
                mock_per_metric_evals = {
                    "openr1_math_example_metric": mock_metric_detail
                }
                sample_result_obj = types.SimpleNamespace(
                    is_score_valid=True,
                    score=1.0,
                    reason="Mocked sample success via env var",
                    per_metric_evals=mock_per_metric_evals,
                )
                results.results.append(sample_result_obj)
        else:
            # Original logic: Create a temporary directory for the math_reward metric
            with tempfile.TemporaryDirectory() as temp_metric_dir:
                metric_main_py_content = """
import sys
import os
# Adjust path to import from reward_kit correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from reward_kit.rewards.math import math_reward
from reward_kit.models import Message, EvaluateResult # Ensure EvaluateResult is imported
from typing import List, Dict, Any, Optional, Union # Added Optional, Union

def evaluate(messages: List[Dict[str, Any]], ground_truth: Optional[Union[str, List[Dict[str, Any]]]] = None, tools: List[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    typed_messages = [Message(**msg) for msg in messages]
    # typed_original_messages is no longer needed as a separate variable

    assistant_content_for_gt = next((m.content for m in typed_messages if m.role == 'assistant'), None)
    if assistant_content_for_gt is None:
        # Return as dict matching EvaluateResult structure
        return {"score": 0.0, "reason": "No assistant message content to use as ground_truth for math_reward", "metrics": {}}

    # Call the actual math_reward function
    # The `ground_truth` parameter for `math_reward` is `assistant_content_for_gt`.
    # Context will be derived from `typed_messages[:-1]` by `math_reward` if needed.
    result_obj = math_reward(
        messages=typed_messages,
        ground_truth=assistant_content_for_gt, # Use assistant's response as GT for this example
        **kwargs
    )
    return result_obj.model_dump() # Return as dict
"""
            metric_name = "openr1_math_example_metric"  # Make metric name specific
            metric_folder_path = os.path.join(temp_metric_dir, metric_name)
            os.makedirs(metric_folder_path)
            with open(os.path.join(metric_folder_path, "main.py"), "w") as f:
                f.write(metric_main_py_content)

            results = preview_evaluation(
                metric_folders=[f"{metric_name}={metric_folder_path}"],
                sample_file=dataset_path,
                max_samples=len(dataset),
            )

        print("--- Fireworks Preview API Evaluation Summary ---")
        print(f"Total Samples Processed by API: {results.total_samples}")
        print(f"Total API Runtime (ms): {results.total_runtime_ms}")

        all_passed_api = True
        passed_api_samples = 0

        if results.results:
            for i, sample_result in enumerate(results.results):
                print(f"\n--- Sample {i+1} (from API) ---")
                print(f"Success: {sample_result.is_score_valid}")
                print(f"Score: {sample_result.score}")
                print(f"Reason: {sample_result.reason}")
                if sample_result.per_metric_evals:
                    print("Per-Metric Evals (from API):")
                    for (
                        metric_eval_name,  # Changed variable name for clarity
                        eval_detail,
                    ) in sample_result.per_metric_evals.items():
                        print(
                            f"  Metric '{metric_eval_name}':"
                        )  # Use the actual metric name from results
                        print(f"    Score: {eval_detail.score}")
                        print(f"    Reason: {eval_detail.reason}")

                if sample_result.score == 1.0 and sample_result.is_score_valid:
                    print("Status: PASSED (according to API)")
                    passed_api_samples += 1
                else:
                    print("Status: FAILED (according to API)")
                    all_passed_api = False
        else:
            print("No individual sample results returned from API.")
            all_passed_api = False

        print("\n--------------------------------------------")
        print(f"Total samples in dataset: {len(dataset)}")
        print(f"Samples passed (according to API): {passed_api_samples}")
        print(f"Samples failed (according to API): {len(dataset) - passed_api_samples}")

        if all_passed_api and len(dataset) > 0 and passed_api_samples == len(dataset):
            print("\nAll samples passed successfully via Fireworks Preview API!")
        else:
            print("\nSome samples failed or an error occurred during API evaluation.")

    except Exception as e:
        print(f"Error during Fireworks Preview API evaluation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
