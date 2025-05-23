"""
CLI command for running agent evaluations using the ForkableResource framework.
"""

import asyncio

try:
    import yaml
except ImportError:
    import sys
    import types

    # Create a stub module if yaml is not installed
    yaml = types.ModuleType("yaml")

    def dummy_safe_load(x):
        return None

    def dummy_dump(x, **kwargs):
        return None

    yaml.safe_load = dummy_safe_load  # type: ignore[assignment]
    yaml.dump = dummy_dump  # type: ignore[assignment]

import json  # Fallback or for explicit JSON files
import logging  # For logger instance
import os  # For environment variables
from pathlib import Path

from pydantic import ValidationError

from reward_kit.agent import Orchestrator
from reward_kit.agent.task_manager import TaskManager
from reward_kit.models import (
    TaskDefinitionModel,
)  # Import the new Pydantic model

# setup_logging is already called in cli.py's main, but good for standalone use if any
# from .common import setup_logging


def agent_eval_command(args):
    """
    Run agent evaluation using the Orchestrator and ForkableResource framework.
    """
    logger = logging.getLogger("agent_eval")  # Use a specific logger
    logger.info("Starting agent-eval command.")

    # Create a new TaskManager instance
    task_manager = TaskManager()

    # Handle task definition file or directory
    if not args.task_def:
        logger.error(
            "Error: --task-def (path to task definition YAML file or directory) is required."
        )
        return 1

    task_def_path = Path(args.task_def)

    # Register tasks based on whether it's a file or directory
    registered_task_ids = []
    if task_def_path.is_file():
        # Single task definition file
        task_def = task_manager._load_task_from_file(str(task_def_path))
        if task_def:
            task_id = task_manager.register_task(task_def)
            registered_task_ids.append(task_id)
        else:
            logger.error(f"Failed to load task definition from {task_def_path}")
            return 1
    elif task_def_path.is_dir():
        # Directory of task definitions
        registered_task_ids = task_manager.register_tasks_from_directory(
            str(task_def_path)
        )
        if not registered_task_ids:
            logger.error(
                f"No valid task definitions found in directory: {task_def_path}"
            )
            return 1
    else:
        logger.error(
            f"Task definition path not found or invalid: {task_def_path}"
        )
        return 1

    logger.info(
        f"Registered {len(registered_task_ids)} tasks: {registered_task_ids}"
    )

    # Run tasks with asyncio
    try:

        async def main_flow():
            # Handle model override
            if getattr(args, "model", None):
                original_model = os.environ.get("MODEL_AGENT")
                os.environ["MODEL_AGENT"] = args.model
                logger.info(f"Model overridden to: {args.model}")

            # Execute tasks based on CLI options
            parallel = getattr(args, "parallel", False)
            max_concurrency = getattr(args, "max_concurrency", 3)
            filter_tasks = getattr(args, "filter", None)

            # Filter tasks if specified
            tasks_to_run = registered_task_ids
            if filter_tasks:
                tasks_to_run = [
                    tid for tid in registered_task_ids if tid in filter_tasks
                ]
                if not tasks_to_run:
                    logger.warning(
                        f"No tasks match the specified filter: {filter_tasks}"
                    )
                    return

            # Execute tasks (parallel or sequential)
            try:
                results = await task_manager.execute_tasks(
                    task_ids=tasks_to_run,
                    parallel=parallel,
                    max_concurrency=max_concurrency,
                )

                # Log results summary
                logger.info(f"Execution completed for {len(results)} tasks")
                for task_id, result in results.items():
                    if isinstance(result, dict) and "error" in result:
                        logger.error(
                            f"Task '{task_id}' failed: {result['error']}"
                        )
                    elif isinstance(result, dict) and "score" in result:
                        logger.info(
                            f"Task '{task_id}' score: {result['score']}"
                        )
                    else:
                        logger.info(f"Task '{task_id}' completed")
            finally:
                # Clean up resources regardless of execution outcome
                await task_manager.cleanup()

        asyncio.run(main_flow())
        logger.info("agent-eval command finished successfully.")
        return 0
    except Exception as e:
        logger.error(f"Error during agent-eval execution: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return 1


def bfcl_eval_command(args):
    """
    Run BFCL agent evaluations using the refactored framework.
    This command specifically manages BFCL task evaluation.
    """
    logger = logging.getLogger("bfcl_eval")
    logger.info("Starting BFCL evaluation command.")

    # Create a TaskManager instance
    task_manager = TaskManager()

    # Get task directory
    task_dir = Path(args.task_dir)
    if not task_dir.is_dir():
        logger.error(f"Task directory not found: {task_dir}")
        return 1

    # Load all tasks from the directory
    logger.info(f"Registering BFCL tasks from {task_dir}")

    try:
        registered_task_ids = []

        # Handle specific task_id vs. all tasks
        if args.task_id:
            # Look for a specific task
            task_path = task_dir / f"{args.task_id}.yaml"
            if not task_path.exists():
                logger.error(f"Task file not found: {task_path}")
                return 1

            task_def = task_manager._load_task_from_file(str(task_path))
            if task_def:
                task_id = task_manager.register_task(task_def)
                registered_task_ids.append(task_id)
                logger.info(f"Registered task: {task_id}")
            else:
                logger.error(f"Failed to load task from {task_path}")
                return 1
        else:
            # Register all tasks in the directory
            registered_task_ids = task_manager.register_tasks_from_directory(
                str(task_dir)
            )
            if not registered_task_ids:
                logger.error(
                    f"No valid BFCL tasks found in directory: {task_dir}"
                )
                return 1
            logger.info(f"Registered {len(registered_task_ids)} BFCL tasks")

        # Run tasks with asyncio
        async def main_flow():
            # Handle model override
            if args.model:
                original_model = os.environ.get("MODEL_AGENT")
                os.environ["MODEL_AGENT"] = args.model
                logger.info(f"Model overridden to: {args.model}")

            # Create output directory if needed
            if args.output_dir:
                output_path = Path(args.output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Results will be saved to {output_path}")

            # Execute tasks based on CLI options
            try:
                results = await task_manager.execute_tasks(
                    task_ids=registered_task_ids,
                    parallel=args.parallel,
                    max_concurrency=args.max_concurrency,
                )

                # Log results summary
                logger.info(
                    f"BFCL evaluation completed for {len(results)} tasks"
                )
                for task_id, result in results.items():
                    if isinstance(result, dict) and "error" in result:
                        logger.error(
                            f"Task '{task_id}' failed: {result['error']}"
                        )
                    elif isinstance(result, dict) and "score" in result:
                        logger.info(
                            f"Task '{task_id}' score: {result['score']}"
                        )

                        # More detailed results for BFCL
                        if "format_score" in result:
                            logger.info(
                                f"Task '{task_id}' format score: {result['format_score']}"
                            )
                        if "state_match" in result:
                            logger.info(
                                f"Task '{task_id}' state match: {result['state_match']}"
                            )
                    else:
                        logger.info(
                            f"Task '{task_id}' completed with result: {result}"
                        )

                # Save results to output directory if specified
                if args.output_dir:
                    results_file = Path(args.output_dir) / "bfcl_results.json"

                    # Convert results to JSON-serializable format
                    serializable_results = {}
                    for task_id, result in results.items():
                        if hasattr(result, "dict"):
                            # Handle Pydantic models
                            serializable_results[task_id] = result.dict()
                        elif isinstance(result, dict):
                            # Handle dictionaries with potentially non-serializable values
                            serializable_dict = {}
                            for k, v in result.items():
                                if hasattr(v, "dict"):
                                    serializable_dict[k] = v.dict()
                                elif hasattr(v, "__dict__"):
                                    serializable_dict[k] = str(v)
                                else:
                                    serializable_dict[k] = v
                            serializable_results[task_id] = serializable_dict
                        else:
                            # Handle other objects by converting to string
                            serializable_results[task_id] = str(result)

                    with open(results_file, "w") as f:
                        json.dump(serializable_results, f, indent=2)
                    logger.info(f"Results saved to {results_file}")

            finally:
                # Clean up resources
                await task_manager.cleanup()

        asyncio.run(main_flow())
        logger.info("BFCL evaluation completed successfully.")
        return 0

    except Exception as e:
        logger.error(f"Error during BFCL evaluation: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return 1
