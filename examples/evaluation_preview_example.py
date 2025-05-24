"""
Example of previewing an evaluation before creation.
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
        "Example: FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY python examples/evaluation_preview_example.py"
    )


from reward_kit.evaluation import create_evaluation, preview_evaluation


def main():
    # Preview the evaluation using metrics folder and samples file
    print("Previewing evaluation...")
    preview_result = preview_evaluation(
        metric_folders=["word_count=./examples/metrics/word_count"],
        sample_file="./examples/samples/samples.jsonl",
        max_samples=2,
    )

    preview_result.display()

    # The preview_evaluation function might use a local fallback if the API is unavailable.
    # This example checks if fallback mode was used and skips evaluator creation in that case
    # to avoid unintended behavior in a non-interactive script.
    import reward_kit.evaluation as evaluation_module

    if (
        hasattr(evaluation_module, "used_preview_api")  # Check if the fallback detection flag is present
        and not evaluation_module.used_preview_api  # Check if fallback mode was used
    ):
        print("Note: The preview used fallback mode due to server issues.")
        # Default to not creating the evaluator in non-interactive mode if fallback was used.
        print(
            "Skipping evaluator creation as fallback mode was used and this is a non-interactive run."
        )
        sys.exit(0)  # Exit gracefully

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
