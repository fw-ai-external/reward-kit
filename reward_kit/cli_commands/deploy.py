"""
CLI command for creating and deploying an evaluator.
"""

import json

from reward_kit.evaluation import create_evaluation

from .common import check_environment


def deploy_command(args):
    """Create and deploy an evaluator"""

    # Check environment variables
    if not check_environment():
        return 1

    # Validate paths for metrics_folders (though create_evaluation might also do this)
    if args.metrics_folders:
        for folder_spec in args.metrics_folders:
            if "=" not in folder_spec:
                print(
                    f"Error: Metric folder format should be 'name=path', got '{folder_spec}'"
                )
                return 1

    if not args.id:
        print("Error: Evaluator ID (--id) is required for deployment")
        return 1

    # Process HuggingFace key mapping if provided
    huggingface_message_key_map = None
    if args.huggingface_key_map:
        try:
            huggingface_message_key_map = json.loads(args.huggingface_key_map)
        except json.JSONDecodeError:
            print("Error: Invalid JSON format for --huggingface-key-map")
            return 1

    # Create the evaluator
    try:
        evaluator = create_evaluation(
            evaluator_id=args.id,
            metric_folders=args.metrics_folders,
            display_name=args.display_name or args.id,
            description=args.description or f"Evaluator: {args.id}",
            force=args.force,
            huggingface_dataset=args.huggingface_dataset,
            huggingface_split=args.huggingface_split,
            huggingface_message_key_map=huggingface_message_key_map,
            huggingface_prompt_key=args.huggingface_prompt_key,
            huggingface_response_key=args.huggingface_response_key,
        )

        print(f"Successfully created evaluator: {evaluator['name']}")
        return 0
    except Exception as e:
        print(f"Error creating evaluator: {str(e)}")
        # For more detailed debugging if needed:
        # import traceback
        # traceback.print_exc()
        return 1
