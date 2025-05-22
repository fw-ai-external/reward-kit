import os
import shutil
import sys
import tempfile
import types  # Import the types module
import logging  # Added for new utility

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from reward_kit.evaluation import preview_evaluation
from reward_kit.common_utils import load_jsonl  # Import the new utility

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

    if not os.environ.get("FIREWORKS_API_KEY"):
        print("Error: FIREWORKS_API_KEY environment variable is not set.")
        print("Please set this variable to your Fireworks API key.")
        print("Example: export FIREWORKS_API_KEY='your_api_key_here'")
        return

    dataset = load_jsonl(dataset_path)

    print(
        f"Starting Fireworks Preview API evaluation for Math Example using {dataset_path}...\n"
    )

    try:
        if os.environ.get("TEST_MOCK_FIREWORKS_PREVIEW") == "true":
            print("Mocking Fireworks Preview API call in test mode.")
            # Simulate a successful response structure that the script expects
            # This should align with what reward_kit.evaluation.EvaluatorPreviewResult would hold
            # if it were populated by Pydantic models from a real API call.
            # For simplicity, we'll construct a mock results object directly here.
            # This bypasses the actual preview_evaluation call.

            from reward_kit.evaluation import EvaluatorPreviewResult  # For structure

            results = EvaluatorPreviewResult()
            results.total_samples = len(dataset)
            results.total_runtime_ms = 42  # Mocked runtime

            for i, sample_data in enumerate(dataset):
                # Simulate what the API and subsequent parsing might produce
                # The key is that sample_result items should allow attribute access
                # for .success, .score, .reason, .per_metric_evals
                mock_metric_detail = types.SimpleNamespace(
                    score=1.0, reason="Mocked metric success", is_score_valid=True
                )
                mock_per_metric_evals = {
                    "math_example_metric": mock_metric_detail
                }  # if using the temp folder approach
                # or "math_reward" if that's the key

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
# Adjust path to import from reward_kit correctly if this temp script is run elsewhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from reward_kit.rewards.math import math_reward
from reward_kit.models import Message, EvaluateResult
from typing import List, Dict, Any

def evaluate(messages: List[Dict[str, Any]], original_messages: List[Dict[str, Any]] = None, tools: List[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    # Convert dict messages to Message objects for math_reward
    typed_messages = [Message(**msg) for msg in messages]
    typed_original_messages = [Message(**msg) for msg in original_messages] if original_messages else typed_messages

    # math_reward expects ground_truth. For this setup, we assume ground_truth is passed in kwargs
    # or derived if not. For the example dataset, assistant's response is the ground_truth.
    # The preview_evaluation sends each sample's **kwargs.
    # We need to ensure 'ground_truth' is in kwargs for each sample.
    # The dataset itself has messages, not direct ground_truth.
    # The preview_evaluation will pass the 'assistant' message content as ground_truth if we structure the call right.
    # However, the `evaluate` signature here is generic.
    # A robust way is to expect ground_truth in kwargs from the sample data.
    # For this example, we'll assume the dataset.jsonl provides the assistant message,
    # and math_reward will use that. The `preview_evaluation` function
    # might need to be called in a way that it knows how to extract ground_truth for math_reward.
    # The `preview_evaluation` in `reward_kit.evaluation.py` doesn't seem to have a direct way
    # to pass `ground_truth` per sample to a custom metric's `evaluate` function other than via `**kwargs`
    # from the sample file itself.
    # The `math_reward` function expects `ground_truth` as a direct parameter.

    # Simplification: Assume the dataset's assistant message is the ground_truth.
    # The `preview_evaluation` function in `reward_kit.evaluation.py` does not directly support
    # passing a list of reward functions. It expects metric folders.
    # The `evaluate` function in this temp main.py needs to conform to what `preview_evaluation` expects
    # for custom metrics. The `math_reward` function needs `ground_truth`.
    # The `dataset.jsonl` has `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`.
    # The `preview_evaluation` sends the whole sample dict as kwargs to `evaluate`.
    # So, `kwargs` will contain `messages`.

    assistant_content = next((m['content'] for m in typed_messages if m.role == 'assistant'), None)
    if assistant_content is None:
        return EvaluateResult(score=0.0, reason="No assistant message for ground truth").model_dump()

    # Call the actual math_reward function
    result = math_reward(
        messages=typed_messages,
        original_messages=typed_original_messages,
        ground_truth=assistant_content, # Use assistant's response as GT for this example
        **kwargs # Pass other potential args like tolerance
    )
    return result.model_dump() # Return as dict
"""
            metric_name = "math_example_metric"
            metric_folder_path = os.path.join(temp_metric_dir, metric_name)
            os.makedirs(metric_folder_path)
            with open(os.path.join(metric_folder_path, "main.py"), "w") as f:
                f.write(metric_main_py_content)

            # Now call preview_evaluation with metric_folders
            # The `dataset` for preview_evaluation should be a list of dictionaries,
            # where each dictionary is a sample that will be passed as **kwargs to the evaluate function.
            # Our dataset is already in the format: [{"messages": [...]}, ...]
            # The `evaluate` function in our temp main.py will receive `messages` via kwargs.
            results = preview_evaluation(
                metric_folders=[f"{metric_name}={metric_folder_path}"],
                sample_file=dataset_path,  # Pass the path to the JSONL file
                max_samples=len(dataset),  # Process all samples from the file
            )

        print("--- Fireworks Preview API Evaluation Summary ---")
        print(f"Total Samples Processed by API: {results.total_samples}")
        print(f"Total API Runtime (ms): {results.total_runtime_ms}")

        all_passed_api = True
        passed_api_samples = 0

        if results.results:
            for i, sample_result in enumerate(results.results):
                print(f"\n--- Sample {i+1} (from API) ---")
                # We don't have original messages here directly, but can show API result
                print(f"Success: {sample_result.is_score_valid}")
                print(f"Score: {sample_result.score}")
                print(f"Reason: {sample_result.reason}")  # Top-level reason from API
                if sample_result.per_metric_evals:
                    print("Per-Metric Evals (from API):")
                    for (
                        metric_name,
                        eval_detail,
                    ) in sample_result.per_metric_evals.items():
                        print(f"  Metric '{metric_name}':")
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
