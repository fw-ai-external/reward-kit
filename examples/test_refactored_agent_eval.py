#!/usr/bin/env python
"""
Test script for the refactored agent evaluation framework.
This script demonstrates the use of the TaskManager to run multiple tasks.
"""

import asyncio
import logging
import os
from pathlib import Path

from reward_kit.agent.task_manager import TaskManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("test_refactored_agent_eval")


async def run_test():
    """Run test tasks using the refactored agent evaluation framework."""
    logger.info("Testing refactored agent evaluation framework...")

    # Create TaskManager
    task_manager = TaskManager()

    # Register tasks from directory
    task_dir = Path(__file__).parent / "test_tasks"
    logger.info(f"Registering tasks from {task_dir}")
    task_ids = task_manager.register_tasks_from_directory(str(task_dir))
    logger.info(f"Registered {len(task_ids)} tasks: {task_ids}")

    # Set model environment variable if not already set
    if not os.environ.get("MODEL_AGENT"):
        # Default to OpenAI GPT-4
        os.environ["MODEL_AGENT"] = "openai/gpt-4"
        logger.info(f"Set MODEL_AGENT to {os.environ['MODEL_AGENT']}")

    # Run tasks sequentially
    try:
        logger.info("Running tasks sequentially...")
        seq_results = await task_manager.execute_tasks(parallel=False)

        logger.info("Sequential execution results:")
        for task_id, result in seq_results.items():
            if isinstance(result, dict) and "score" in result:
                logger.info(
                    f"Task {task_id}: Score: {result['score']}, Reason: {result.get('reason')}"
                )
            else:
                logger.info(f"Task {task_id}: Result: {result}")

        # Run tasks in parallel
        logger.info("\nRunning tasks in parallel...")
        par_results = await task_manager.execute_tasks(parallel=True, max_concurrency=2)

        logger.info("Parallel execution results:")
        for task_id, result in par_results.items():
            if isinstance(result, dict) and "score" in result:
                logger.info(
                    f"Task {task_id}: Score: {result['score']}, Reason: {result.get('reason')}"
                )
            else:
                logger.info(f"Task {task_id}: Result: {result}")

    finally:
        # Clean up
        await task_manager.cleanup()
        logger.info("Test completed.")


def main():
    """Main entry point."""
    asyncio.run(run_test())


if __name__ == "__main__":
    main()
