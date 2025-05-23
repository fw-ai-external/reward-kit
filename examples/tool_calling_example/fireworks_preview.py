"""
Example of previewing a function calling evaluation using Fireworks API.
"""

import json
import logging
import os
import sys
from typing import List, cast  # Added cast and List

# Configure logging
log_file_path = os.path.join(os.path.dirname(__file__), "fireworks_preview_debug.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout),  # Keep logging to console as well
    ],
)
logger = logging.getLogger(__name__)

# Ensure reward-kit is in the path
# The fireworks_preview.py script is in examples/tool_calling_example/
# The reward_kit module is at the root. So, path should be ../../reward_kit
# The metrics folder is ./metrics/ within examples/tool_calling_example/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# Check for required environment variables
if not os.environ.get("FIREWORKS_API_KEY"):
    print("Warning: FIREWORKS_API_KEY environment variable is not set.")
    print(
        "This script requires FIREWORKS_API_KEY for interacting with the Fireworks API."
    )
    print(
        "Example: FIREWORKS_API_KEY=$YOUR_FIREWORKS_API_KEY python examples/tool_calling_example/fireworks_preview.py"
    )
    # sys.exit(1) # Optionally exit if key is mandatory for any run

from reward_kit.evaluation import preview_evaluation

# We don't need to import composite_function_call_reward here anymore,
# as it's called by the script in the metric folder.


def main():
    # Paths should be relative to this script's location if it's intended to be run from its own directory
    # or if tests run it from its directory.
    script_dir = os.path.dirname(__file__)
    sample_file_path = os.path.join(script_dir, "dataset.jsonl")
    metric_folder_path = os.path.join(script_dir, "metrics/exact_match_metric")

    # Check if the sample file exists
    if not os.path.exists(sample_file_path):
        print(f"Error: Sample file not found at {sample_file_path}")
        print("Please ensure the dataset.jsonl is in the correct location.")
        sys.exit(1)

    logger.info("Starting function calling evaluation preview using Fireworks API...")
    logger.debug(f"Sample file path: {sample_file_path}")
    logger.debug(f"Metric folder path: {metric_folder_path}")

    try:
        # Attempt to specify the reward function.
        # The exact parameter name for specifying a built-in reward function might vary.
        # Common patterns include 'reward_function_name', 'evaluator', or 'metric_name'.
        # Based on the general structure, 'reward_config' or 'evaluator_config' might also be options.
        # We are using 'reward_function_name' as a plausible guess.
        # If this is incorrect, the error message from preview_evaluation should guide us.
        # Using metric_folders to point to our wrapper.
        # The path to the metric folder is relative to where the script is run.
        # If running from project root: "examples/tool_calling_example/metrics/exact_match_metric"
        # If running from examples/tool_calling_example/: "./metrics/exact_match_metric"
        # The test runner in test_readiness.py runs scripts from their own directory.
        # So, the path for metric_folders in the script should be relative to the script's location.

        eval_params = {
            "sample_file": sample_file_path,
            "metric_folders": [f"exact_match_metric={metric_folder_path}"],
            "max_samples": 5,
        }
        logger.info(
            f"Preparing to call preview_evaluation with parameters: {json.dumps(eval_params, indent=2)}"
        )
        logger.debug("Attempting to call preview_evaluation...")

        preview_result = preview_evaluation(
            sample_file=cast(str, sample_file_path),
            metric_folders=cast(
                List[str], [f"exact_match_metric={metric_folder_path}"]
            ),
            max_samples=cast(int, 5),
            # Other parameters will use their defaults from the function signature
        )

        logger.debug("preview_evaluation call completed.")
        logger.info("preview_evaluation call successful.")
        # Attempt to serialize preview_result if it's complex, otherwise convert to string
        try:
            result_str = json.dumps(
                preview_result.__dict__, indent=2, default=str
            )  # Attempt to serialize if it has __dict__
        except AttributeError:
            result_str = str(preview_result)
        except TypeError:  # Handle non-serializable parts
            result_str = f"Preview result object (type: {type(preview_result)}) could not be fully serialized. Displaying basic string representation: {str(preview_result)}"

        logger.debug(f"Preview Result object: {result_str}")

        print("\nPreview Result:")
        preview_result.display()

        # Check if fallback mode was used (if applicable, similar to evaluation_preview_example.py)
        import reward_kit.evaluation as evaluation_module

        if (
            hasattr(evaluation_module, "used_preview_api")
            and not evaluation_module.used_preview_api
        ):
            logger.warning(
                "The preview may have used fallback mode due to server issues."
            )
            print(
                "\nNote: The preview may have used fallback mode due to server issues."
            )

    except Exception as e:
        logger.error(f"Error during preview_evaluation: {str(e)}", exc_info=True)
        print(f"\nError during preview_evaluation: {str(e)}")
        print(
            "This might be due to an incorrect parameter for specifying the reward function,"
        )
        print("or issues with API connectivity or authentication.")
        print("Please check the Fireworks API documentation for `preview_evaluation`.")
        logger.info(f"Debug logs have been saved to: {log_file_path}")


if __name__ == "__main__":
    main()
