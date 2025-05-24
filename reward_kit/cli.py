"""
Command-line interface for reward-kit.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
import uuid
from pathlib import Path

from reward_kit.evaluation import create_evaluation, preview_evaluation

from .cli_commands.agent_eval_cmd import (  # Now points to the V2 logic
    agent_eval_command,
)
from .cli_commands.common import (
    check_agent_environment,
    check_environment,
    setup_logging,
)
from .cli_commands.deploy import deploy_command
from .cli_commands.preview import preview_command

# importlib.util was unused


# Note: validate_task_bundle, find_task_dataset, get_toolset_config, export_tool_specs
# were helpers for the old agent_eval_command and are now moved into agent_eval_cmd.py
# or will be part of the new agent_eval_v2_command logic.
# For now, they are removed from cli.py as agent_eval_command is imported.


def parse_args(args=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="reward-kit: Tools for evaluation and reward modeling"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Preview command
    preview_parser = subparsers.add_parser(
        "preview", help="Preview an evaluator with sample data"
    )
    preview_parser.add_argument(
        "--metrics-folders",
        "-m",
        nargs="+",
        help="Metric folders in format 'name=path', e.g., 'clarity=./metrics/clarity'",
    )

    # Make samples optional to allow HF dataset option
    preview_parser.add_argument(
        "--samples",
        "-s",
        required=False,
        help="Path to JSONL file containing sample data",
    )
    preview_parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Maximum number of samples to process (default: 5)",
    )

    # Add HuggingFace dataset options
    hf_group = preview_parser.add_argument_group("HuggingFace Dataset Options")
    hf_group.add_argument(
        "--huggingface-dataset",
        "--hf",
        help="HuggingFace dataset name (e.g., 'deepseek-ai/DeepSeek-ProverBench')",
    )
    hf_group.add_argument(
        "--huggingface-split",
        default="train",
        help="Dataset split to use (default: 'train')",
    )
    hf_group.add_argument(
        "--huggingface-prompt-key",
        default="prompt",
        help="Key in the dataset containing the prompt text (default: 'prompt')",
    )
    hf_group.add_argument(
        "--huggingface-response-key",
        default="response",
        help="Key in the dataset containing the response text (default: 'response')",
    )
    hf_group.add_argument(
        "--huggingface-key-map",
        help="JSON mapping of dataset keys to reward-kit message keys",
    )
    preview_parser.add_argument(
        "--remote-url",
        help="URL of a remote reward function endpoint to preview against. If provided, metrics-folders might be ignored.",
    )

    # Deploy command
    deploy_parser = subparsers.add_parser(
        "deploy", help="Create and deploy an evaluator, or register a remote one"
    )
    deploy_parser.add_argument("--id", required=True, help="ID for the evaluator")
    deploy_parser.add_argument(
        "--metrics-folders",
        "-m",
        nargs="+",
        required=False,  # No longer strictly required if --remote-url is used
        help="Metric folders in format 'name=path', e.g., 'clarity=./metrics/clarity'. Required if not using --remote-url.",
    )
    deploy_parser.add_argument(
        "--display-name",
        help="Display name for the evaluator (defaults to ID if not provided)",
    )
    deploy_parser.add_argument("--description", help="Description for the evaluator")
    deploy_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force update if evaluator already exists",
    )

    # Add HuggingFace dataset options to deploy command
    hf_deploy_group = deploy_parser.add_argument_group("HuggingFace Dataset Options")
    hf_deploy_group.add_argument(
        "--huggingface-dataset",
        "--hf",
        help="HuggingFace dataset name (e.g., 'deepseek-ai/DeepSeek-ProverBench')",
    )
    hf_deploy_group.add_argument(
        "--huggingface-split",
        default="train",
        help="Dataset split to use (default: 'train')",
    )
    hf_deploy_group.add_argument(
        "--huggingface-prompt-key",
        default="prompt",
        help="Key in the dataset containing the prompt text (default: 'prompt')",
    )
    hf_deploy_group.add_argument(
        "--huggingface-response-key",
        default="response",
        help="Key in the dataset containing the response text (default: 'response')",
    )
    hf_deploy_group.add_argument(
        "--huggingface-key-map",
        help="JSON mapping of dataset keys to reward-kit message keys",
    )
    deploy_parser.add_argument(
        "--remote-url",
        help="URL of a pre-deployed remote reward function. If provided, deploys by registering this URL with Fireworks AI.",
    )

    # GCP deployment options
    gcp_group = deploy_parser.add_argument_group(
        "GCP Cloud Run Deployment Options (used if --target is gcp-cloud-run)"
    )
    gcp_group.add_argument(
        "--target",
        choices=["fireworks", "gcp-cloud-run"],
        default="fireworks",
        help="Deployment target. 'fireworks' for standard deployment, 'gcp-cloud-run' for deploying to Google Cloud Run.",
    )
    gcp_group.add_argument(
        "--function-ref",
        help="Reference to the reward function to deploy (e.g., 'examples.gcp_cloud_run_deployment_example.dummy_rewards.hello_world_reward'). Required for gcp-cloud-run target.",
    )
    gcp_group.add_argument(
        "--gcp-project",
        required=False,
        help="Google Cloud Project ID. Must be provided via CLI or rewardkit.yaml.",
    )
    gcp_group.add_argument(
        "--gcp-region",
        required=False,
        help="Google Cloud Region for deployment (e.g., 'us-central1'). Must be provided via CLI or rewardkit.yaml.",
    )
    gcp_group.add_argument(
        "--gcp-ar-repo",
        required=False,
        help="Google Artifact Registry repository name. Optional, defaults to value in rewardkit.yaml or 'reward-kit-evaluators' if not specified.",
    )
    gcp_group.add_argument(
        "--service-account",
        help="Email of the GCP service account to run the Cloud Run service. Optional.",
    )
    gcp_group.add_argument(
        "--entry-point",
        default="reward_function",
        help="The name of the entry point function within your --function-ref module (default: reward_function). Only for gcp-cloud-run.",
    )
    gcp_group.add_argument(
        "--runtime",
        default="python311",  # Or a sensible default
        help="The Cloud Functions/Run runtime (e.g., python311). Only for gcp-cloud-run.",
    )
    gcp_group.add_argument(
        "--gcp-auth-mode",
        choices=["open", "api-key"],  # Add 'iam' later
        default=None,  # Default will be resolved in deploy_command
        help="Authentication mode for the deployed GCP Cloud Run service. "
        "'open': Publicly accessible. "
        "'api-key': Service is publicly accessible but requires an API key in requests (handled by the application). "
        "If not specified, defaults to value in rewardkit.yaml or 'api-key'. Optional.",
    )

    # Agent-eval command
    agent_eval_parser = subparsers.add_parser(
        "agent-eval", help="Run agent evaluation using the ForkableResource framework."
    )
    agent_eval_parser.add_argument(
        "--task-def",
        required=True,
        help="Path to task definition file or directory containing task definitions.",
    )
    agent_eval_parser.add_argument(
        "--parallel",
        action="store_true",
        help="Execute tasks in parallel when multiple tasks are specified.",
    )
    agent_eval_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=3,
        help="Maximum number of tasks to execute in parallel (default: 3).",
    )
    agent_eval_parser.add_argument(
        "--filter",
        nargs="+",
        help="Run only tasks matching the specified task IDs.",
    )
    agent_eval_parser.add_argument(
        "--output-dir",
        default="./agent_runs",
        help="Directory to store agent evaluation run results (default: ./agent_runs).",
    )
    agent_eval_parser.add_argument(
        "--model",
        help="Override MODEL_AGENT environment variable (format: provider/model_name).",
    )

    return parser.parse_args(args)


def main():
    """Main entry point for the CLI"""
    try:
        from dotenv import load_dotenv

        load_dotenv(override=True)  # Load .env file, overriding existing shell env vars
    except ImportError:
        # python-dotenv not installed, proceed without it.
        # Consider logging a warning if .env support is considered core.
        pass

    args = parse_args()
    # Setup logging based on global verbose/debug flags if they exist on args,
    # or command-specific if not. getattr is good for this.
    setup_logging(args.verbose, getattr(args, "debug", False))

    if args.command == "preview":
        return preview_command(args)
    elif args.command == "deploy":
        return deploy_command(args)
    elif args.command == "agent-eval":
        return agent_eval_command(args)
    else:
        # No command provided, show help
        # This case should ideally not be reached if subparsers are required.
        # If a command is not matched, argparse usually shows help or an error.
        # Keeping this for safety or if top-level `reward-kit` without command is allowed.
        parser = (
            argparse.ArgumentParser()
        )  # This might need to be the main parser instance
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
