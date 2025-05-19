#!/usr/bin/env python
"""
BFCL Evaluation Example

This script demonstrates how to use the refactored agent evaluation framework
to run BFCL (Berkeley Function Call Leaderboard) tasks.

This uses the integrated BFCL environment implementations in reward_kit/agent/resources/bfcl_envs,
removing the need for external verifiers.envs dependencies.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add the project root to the Python path
project_root = Path(__file__).parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# NOTE: No longer need to add test mocks to Python path
# The BFCL environments are now part of the main codebase in reward_kit/agent/resources/bfcl_envs

from reward_kit.agent.task_manager import TaskManager
from reward_kit.models import TaskDefinitionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("bfcl_eval")


async def run_bfcl_evaluation(
    task_path: Union[str, Path],
    parallel: bool = False,
    max_concurrency: int = 3,
    filter_tasks: Optional[List[str]] = None,
    model: Optional[str] = None,
    verbose: bool = False,
    test_mode: bool = False,  # Run actual agent execution by default
) -> Dict[str, Any]:
    """
    Run BFCL evaluation tasks using the refactored agent evaluation framework.

    Args:
        task_path: Path to a YAML task definition file or directory of tasks
        parallel: Whether to run tasks in parallel
        max_concurrency: Maximum number of tasks to run in parallel
        filter_tasks: List of task IDs to run (if None, runs all tasks)
        model: Model to use for agent (format: provider/model_name)
        verbose: Whether to show verbose logging
        test_mode: If True, run in simplified test mode without actual agent execution

    Returns:
        Dictionary mapping task IDs to evaluation results
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Set model if provided
    if model:
        logger.info(f"Setting model to {model}")
        os.environ["MODEL_AGENT"] = model
    elif "MODEL_AGENT" not in os.environ:
        # Default to GPT-4 if no model is specified
        logger.info("No model specified. Defaulting to openai/gpt-4")
        os.environ["MODEL_AGENT"] = "openai/gpt-4.1-2025-04-14"
    else:
        logger.info(f"Using model from environment: {os.environ['MODEL_AGENT']}")

    # Create task manager
    task_manager = TaskManager()
    logger.info(f"Created TaskManager")

    # Load tasks
    task_path = Path(task_path)

    if task_path.is_file():
        # Single task
        logger.info(f"Loading task from {task_path}")
        task = task_manager._load_task_from_file(str(task_path))
        if not task:
            logger.error(f"Failed to load task from {task_path}")
            return {}

        task_id = task_manager.register_task(task)
        registered_task_ids = [task_id]
    elif task_path.is_dir():
        # Directory of tasks
        logger.info(f"Loading tasks from directory: {task_path}")
        registered_task_ids = task_manager.register_tasks_from_directory(str(task_path))
        if not registered_task_ids:
            logger.error(f"No valid task definitions found in {task_path}")
            return {}
    else:
        logger.error(f"Task path not found or invalid: {task_path}")
        return {}

    logger.info(f"Registered {len(registered_task_ids)} tasks: {registered_task_ids}")

    # Filter tasks if specified
    task_ids_to_run = registered_task_ids
    if filter_tasks:
        task_ids_to_run = [tid for tid in registered_task_ids if tid in filter_tasks]
        if not task_ids_to_run:
            logger.warning(f"No tasks match the specified filters: {filter_tasks}")
            return {}
        logger.info(f"Filtered to {len(task_ids_to_run)} tasks: {task_ids_to_run}")

    # Execute tasks
    try:
        logger.info(
            f"Executing tasks with parallel={parallel}, max_concurrency={max_concurrency}"
        )

        # Only prepare tasks in test_mode - don't try to run them with an actual agent
        if test_mode:
            logger.info(
                "Running in test mode - only preparing tasks, not executing them"
            )
            results = {}
            for task_id in task_ids_to_run:
                prepared = await task_manager.prepare_task(task_id)
                orchestrator = task_manager.orchestrators.get(task_id)
                if prepared and orchestrator:
                    # Ensure the reward function is correctly loaded
                    import importlib

                    reward_function_path = task_manager.tasks[
                        task_id
                    ].reward_function_path

                    # If the Orchestrator failed to load the reward function, try manually loading it
                    if not orchestrator.reward_function:
                        logger.debug(
                            f"Orchestrator failed to load reward function, trying manually..."
                        )
                        try:
                            if "." in reward_function_path:
                                module_path, function_name = (
                                    reward_function_path.rsplit(".", 1)
                                )
                                module = importlib.import_module(module_path)
                                if hasattr(module, function_name):
                                    reward_function = getattr(module, function_name)
                                    if callable(reward_function):
                                        orchestrator.reward_function = reward_function
                                        logger.debug(
                                            f"Successfully loaded reward function manually"
                                        )

                            # Try importing from rewards package as a last resort
                            if not orchestrator.reward_function:
                                import reward_kit.rewards as rw

                                function_name = reward_function_path.split(".")[-1]
                                if hasattr(rw, function_name):
                                    orchestrator.reward_function = getattr(
                                        rw, function_name
                                    )
                                    logger.debug(
                                        f"Loaded reward function from rewards package"
                                    )
                        except Exception as e:
                            logger.error(
                                f"Error attempting to manually load reward function: {e}"
                            )

                    results[task_id] = {
                        "status": "prepared",
                        "has_resource": orchestrator.base_resource is not None,
                        "has_reward_function": orchestrator.reward_function is not None,
                        "resource_type": task_manager.tasks[task_id].resource_type,
                        "reward_function_path": task_manager.tasks[
                            task_id
                        ].reward_function_path,
                    }
                else:
                    results[task_id] = {"error": "Failed to prepare task"}
        else:
            # Normal execution
            results = await task_manager.execute_tasks(
                task_ids=task_ids_to_run,
                parallel=parallel,
                max_concurrency=max_concurrency,
            )

        # Log results
        logger.info(f"Completed execution for {len(results)} tasks")
        for task_id, result in results.items():
            if isinstance(result, dict) and "error" in result:
                logger.error(f"Task '{task_id}' failed: {result['error']}")
            elif isinstance(result, dict) and "score" in result:
                logger.info(f"Task '{task_id}' score: {result['score']}")
                if "reason" in result:
                    logger.info(f"Task '{task_id}' reason: {result['reason']}")
            else:
                logger.info(f"Task '{task_id}' result: {result}")

        return results
    except Exception as e:
        logger.error(f"Error during task execution: {e}", exc_info=True)
        return {"error": str(e)}
    finally:
        # Clean up
        await task_manager.cleanup()
        logger.info("Cleaned up resources")


async def main():
    """Main function to demonstrate BFCL evaluation."""
    # Set up command line arguments using argparse
    import argparse

    parser = argparse.ArgumentParser(description="Run BFCL evaluation tasks")
    parser.add_argument(
        "--task", "-t", required=True, help="Path to task definition file or directory"
    )
    parser.add_argument(
        "--parallel", "-p", action="store_true", help="Run tasks in parallel"
    )
    parser.add_argument(
        "--max-concurrency",
        "-c",
        type=int,
        default=3,
        help="Maximum number of tasks to run in parallel",
    )
    parser.add_argument("--filter", "-f", nargs="+", help="Filter tasks by ID")
    parser.add_argument(
        "--model", "-m", help="Model to use (format: provider/model_name)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument("--output", "-o", help="Path to save results JSON")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode without agent execution",
    )

    args = parser.parse_args()

    # Run evaluation
    results = await run_bfcl_evaluation(
        task_path=args.task,
        parallel=args.parallel,
        max_concurrency=args.max_concurrency,
        filter_tasks=args.filter,
        model=args.model,
        verbose=args.verbose,
        test_mode=args.test_mode,
    )

    # Save results if output path is specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    # Print summary
    print("\n--- BFCL Evaluation Summary ---")
    for task_id, result in results.items():
        if isinstance(result, dict) and "error" in result:
            print(f"Task '{task_id}': FAILED - {result['error']}")
        elif isinstance(result, dict) and "score" in result:
            print(f"Task '{task_id}': Score {result['score']}")
        else:
            print(f"Task '{task_id}': {result}")


if __name__ == "__main__":
    asyncio.run(main())
