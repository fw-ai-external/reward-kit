"""
Example of previewing an evaluation before creation.
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
        "Example: FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY python examples/evaluation_preview_example.py"
    )

# No example mode - will use real authentication

from reward_kit.evaluation import preview_evaluation, create_evaluation


def main():
    # Preview the evaluation using metrics folder and samples file
    print("Previewing evaluation...")
    preview_result = preview_evaluation(
        metric_folders=["word_count=./examples/metrics/word_count"],
        sample_file="./examples/samples/samples.jsonl",
        max_samples=2,
    )

    preview_result.display()

    # Modified approach - add a flag to reward_kit.evaluation that we'll check
    # to determine if the preview API was successfully used
    import reward_kit.evaluation as evaluation_module

    # Check if 'used_preview_api' attribute exists and is True
    # This attribute would be set to True when the preview API is used
    # and False when fallback mode is used
    if (
        hasattr(evaluation_module, "used_preview_api")
        and not evaluation_module.used_preview_api
    ):
        print("Note: The preview used fallback mode due to server issues.")
        proceed = input(
            "The server might be having connectivity issues. Do you want to try creating the evaluator anyway? (y/n): "
        )
        if proceed.lower() != "y":
            print("Skipping evaluator creation.")
            sys.exit(0)

    print("\nCreating evaluation...")
    try:
        evaluator = create_evaluation(
            evaluator_id="word-count-eval",
            metric_folders=["word_count=./examples/metrics/word_count"],
            display_name="Word Count Evaluator",
            description="Evaluates responses based on word count",
            force=True,  # Update the evaluator if it already exists
        )
        print(f"Created evaluator: {evaluator['name']}")
    except Exception as e:
        print(f"Error creating evaluator: {str(e)}")
        print("Make sure you have proper Fireworks API credentials set up.")


if __name__ == "__main__":
    main()
