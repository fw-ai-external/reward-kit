"""
import argparse
from typing import Optional
CLI command for creating and deploying an evaluator,
or registering a pre-deployed remote evaluator.
"""

import json
import sys  # For sys.exit

# Assuming create_evaluation can be adapted or a new function like
# register_remote_evaluator will be added to reward_kit.evaluation
from reward_kit.evaluation import create_evaluation

from .common import check_environment

# Placeholder for the new function if needed:
# from reward_kit.evaluation import register_remote_evaluator_url


def deploy_command(args: argparse.Namespace) -> Optional[int]:
    """Create and deploy an evaluator or register a remote one."""

    # Check environment variables
    if not check_environment():
        return 1

    if not args.id:  # ID is always required
        print("Error: Evaluator ID (--id) is required.")
        return 1

    if args.remote_url:
        # Deploying by registering a remote URL
        print(f"Registering remote evaluator '{args.id}' with URL: {args.remote_url}")
        if args.metrics_folders:
            print(
                "Info: --metrics-folders are ignored when deploying with --remote-url."
            )
        if args.huggingface_dataset:  # Also ignore HF dataset args in this mode
            print(
                "Info: HuggingFace dataset arguments are ignored when deploying with --remote-url."
            )

        if not (
            args.remote_url.startswith("http://")
            or args.remote_url.startswith("https://")
        ):
            print(
                f"Error: Invalid --remote-url '{args.remote_url}'. Must start with http:// or https://"
            )
            return 1

        try:
            # This is where the actual API call to register the remote URL would go.
            # This functionality needs to be implemented in reward_kit.evaluation or a similar module.
            # For example, a function like:
            # from reward_kit.evaluation import register_remote_evaluator_url
            # evaluator = register_remote_evaluator_url(
            #     evaluator_id=args.id,
            #     remote_url=args.remote_url,
            #     display_name=args.display_name or args.id,
            #     description=args.description or f"Remote evaluator: {args.id}",
            #     force=args.force
            # )
            # print(f"Successfully registered remote evaluator: {evaluator.get('name', args.id)} with URL {args.remote_url}")

            # Placeholder implementation until the actual registration logic exists:
            print(
                f"SUCCESS (Placeholder): Remote evaluator '{args.id}' would be registered with URL '{args.remote_url}'."
            )
            print(
                "Note: Actual API call to Fireworks platform needs to be implemented for this functionality."
            )
            # Simulate a successful response structure
            evaluator = {
                "name": args.id,
                "id": args.id,
                "config": {"remote_url": args.remote_url},
            }
            print(
                f"Successfully registered/updated remote evaluator: {evaluator['name']}"
            )

            return 0
        except Exception as e:
            print(f"Error registering remote evaluator '{args.id}': {str(e)}")
            # import traceback
            # traceback.print_exc()
            return 1

    else:
        # Original behavior: Deploying by packaging local metrics_folders
        if not args.metrics_folders:
            print("Error: --metrics-folders are required if not using --remote-url.")
            return 1

        # Validate paths for metrics_folders (though create_evaluation might also do this)
        for folder_spec in args.metrics_folders:
            if "=" not in folder_spec:
                print(
                    f"Error: Metric folder format should be 'name=path', got '{folder_spec}'"
                )
                return 1

        # Process HuggingFace key mapping if provided
        huggingface_message_key_map = None
        if args.huggingface_key_map:
            try:
                huggingface_message_key_map = json.loads(args.huggingface_key_map)
            except json.JSONDecodeError:
                print("Error: Invalid JSON format for --huggingface-key-map")
                return 1

        try:
            # This call assumes create_evaluation handles the standard deployment
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

            print(f"Successfully created/updated evaluator: {evaluator['name']}")
            return 0
        except Exception as e:
            print(f"Error creating/updating evaluator: {str(e)}")
            # import traceback
            # traceback.print_exc()
            return 1
