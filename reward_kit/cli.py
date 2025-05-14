"""
Command-line interface for reward-kit.
"""

import argparse
import sys
import os
import json
import logging
import asyncio
import traceback
import uuid
from pathlib import Path

# importlib.util was unused

from reward_kit.evaluation import preview_evaluation, create_evaluation
from .cli_commands.common import setup_logging, check_environment, check_agent_environment
from .cli_commands.preview import preview_command
from .cli_commands.deploy import deploy_command
from .cli_commands.agent_eval_cmd import agent_eval_command # Now points to the V2 logic

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

    # Deploy command
    deploy_parser = subparsers.add_parser(
        "deploy", help="Create and deploy an evaluator"
    )
    deploy_parser.add_argument(
        "--id", required=True, help="ID for the evaluator"
    )
    deploy_parser.add_argument(
        "--metrics-folders",
        "-m",
        nargs="+",
        required=True,
        help="Metric folders in format 'name=path', e.g., 'clarity=./metrics/clarity'",
    )
    deploy_parser.add_argument(
        "--display-name",
        help="Display name for the evaluator (defaults to ID if not provided)",
    )
    deploy_parser.add_argument(
        "--description", help="Description for the evaluator"
    )
    deploy_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force update if evaluator already exists",
    )

    # Add HuggingFace dataset options to deploy command
    hf_deploy_group = deploy_parser.add_argument_group(
        "HuggingFace Dataset Options"
    )
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

    # Agent-eval command
    agent_eval_parser = subparsers.add_parser(
        "agent-eval", help="Run agent evaluation using the ForkableResource framework."
    )
    agent_eval_parser.add_argument(
        "--task-def",
        required=True,
        help="Path to the task definition YAML/JSON file for the agent evaluation.",
    )
    # Add other relevant arguments for agent-eval (formerly v2) if needed,
    # e.g., output_dir, model overrides, etc.
    # For PoC, --task-def is the main one.
    # Re-use verbose and debug from the main parser if they are global.
    # agent_eval_parser.add_argument(
    #     "--output-dir", # Example, if Orchestrator needs it and it's not in task_def
    #     default="./agent_runs", # Updated default dir name
    #     help="Directory to store agent evaluation runs (default: ./agent_runs)",
    # )
    # Arguments like --debug are handled by the main parser.

    return parser.parse_args(args)


def main():
    """Main entry point for the CLI"""
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
        parser = argparse.ArgumentParser() # This might need to be the main parser instance
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
