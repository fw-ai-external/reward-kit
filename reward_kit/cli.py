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
from .cli_commands.agent_eval_cmd import agent_eval_command # For the old agent-eval
from .cli_commands.agent_eval_v2_cmd import agent_eval_v2_command # For the new agent-eval-v2

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
        "agent-eval", help="Run agent evaluation on a task dataset"
    )

    # Task specification (mutually exclusive)
    task_group = agent_eval_parser.add_argument_group("Task Specification")
    task_group.add_argument(
        "--task-dir",
        help="Path to task bundle directory containing reward.py, tools.py, etc.",
    )
    task_group.add_argument(
        "--dataset", "-d", help="Path to JSONL file containing task dataset"
    )

    # Output and models
    output_group = agent_eval_parser.add_argument_group("Output and Models")
    output_group.add_argument(
        "--output-dir",
        "-o",
        default="./runs",
        help="Directory to store evaluation runs (default: ./runs)",
    )
    output_group.add_argument(
        "--model", help="Override MODEL_AGENT environment variable"
    )
    output_group.add_argument(
        "--sim-model",
        help="Override MODEL_SIM environment variable for simulated user",
    )

    # Test and debug options
    debug_group = agent_eval_parser.add_argument_group("Testing and Debugging")
    debug_group.add_argument(
        "--no-sim-user",
        action="store_true",
        help="Disable simulated user (use static initial messages only)",
    )
    debug_group.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode without requiring API keys",
    )
    debug_group.add_argument(
        "--mock-response",
        action="store_true",
        help="Use a mock agent response (works with --test-mode)",
    )
    debug_group.add_argument(
        "--debug", action="store_true", help="Enable detailed debug logging"
    )
    debug_group.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate task bundle structure without running evaluation",
    )
    debug_group.add_argument(
        "--export-tools",
        metavar="DIR",
        help="Export tool specifications to directory for manual testing",
    )

    # Advanced options
    advanced_group = agent_eval_parser.add_argument_group("Advanced Options")
    advanced_group.add_argument(
        "--task-ids", help="Comma-separated list of task IDs to run"
    )
    advanced_group.add_argument(
        "--max-tasks", type=int, help="Maximum number of tasks to evaluate"
    )
    advanced_group.add_argument(
        "--registries",
        nargs="+",
        help="Custom tool registries in format 'name=path'",
    )
    advanced_group.add_argument(
        "--registry-override",
        help="Override all toolset paths with this registry path",
    )
    advanced_group.add_argument(
        "--evaluator", help="Custom evaluator module path (overrides default)"
    )

    # Agent-eval-v2 command (New Framework)
    agent_eval_v2_parser = subparsers.add_parser(
        "agent-eval-v2", help="Run agent evaluation using the new V2 (ForkableResource) framework."
    )
    agent_eval_v2_parser.add_argument(
        "--task-def",
        required=True,
        help="Path to the task definition JSON file for the V2 framework.",
    )
    # Add other relevant arguments for V2 if needed, e.g., output_dir, model overrides, etc.
    # For PoC, --task-def is the main one.
    # Re-use verbose and debug from the main parser if they are global.
    # agent_eval_v2_parser.add_argument(
    #     "--output-dir", # Example, if Orchestrator needs it and it's not in task_def
    #     default="./agent_v2_runs",
    #     help="Directory to store V2 evaluation runs (default: ./agent_v2_runs)",
    # )


    return parser.parse_args(args)


def main():
    """Main entry point for the CLI"""
    args = parse_args()
    # Setup logging based on global verbose/debug flags if they exist on args,
    # or command-specific if not. getattr is good for this.
    # The setup_logging in agent_eval_v2_cmd.py might be redundant if global flags are used.
    # For now, main cli.py sets it up.
    setup_logging(args.verbose, getattr(args, "debug", False))

    if args.command == "preview":
        return preview_command(args)
    elif args.command == "deploy":
        return deploy_command(args)
    elif args.command == "agent-eval":
        return agent_eval_command(args)
    elif args.command == "agent-eval-v2":
        return agent_eval_v2_command(args)
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
