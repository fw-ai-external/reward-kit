"""
CLI command for previewing an evaluator.
"""
import json
from pathlib import Path
from reward_kit.evaluation import preview_evaluation
from .common import check_environment

def preview_command(args):
    """Preview an evaluator with sample data"""

    # Check environment variables
    if not check_environment():
        return 1

    # Validate paths
    if args.metrics_folders:
        for folder in args.metrics_folders:
            if "=" not in folder:
                print(
                    f"Error: Metric folder format should be 'name=path', got '{folder}'"
                )
                return 1

    # Ensure either samples or huggingface_dataset is provided
    if not args.samples and not args.huggingface_dataset:
        print(
            "Error: Either sample file (--samples) or HuggingFace dataset (--huggingface-dataset) is required"
        )
        return 1

    # If using samples file, verify it exists
    if args.samples and not Path(args.samples).exists():
        print(f"Error: Sample file '{args.samples}' not found")
        return 1

    # Process HuggingFace key mapping if provided
    huggingface_message_key_map = None
    if args.huggingface_key_map:
        try:
            huggingface_message_key_map = json.loads(args.huggingface_key_map)
        except json.JSONDecodeError:
            print("Error: Invalid JSON format for --huggingface-key-map")
            return 1

    # Run preview
    try:
        preview_result = preview_evaluation(
            metric_folders=args.metrics_folders,
            sample_file=args.samples if args.samples else None,
            max_samples=args.max_samples,
            huggingface_dataset=args.huggingface_dataset,
            huggingface_split=args.huggingface_split,
            huggingface_prompt_key=args.huggingface_prompt_key,
            huggingface_response_key=args.huggingface_response_key,
            huggingface_message_key_map=huggingface_message_key_map,
        )

        preview_result.display()
        return 0
    except Exception as e:
        print(f"Error previewing evaluator: {str(e)}")
        # For more detailed debugging if needed:
        # import traceback
        # traceback.print_exc()
        return 1
