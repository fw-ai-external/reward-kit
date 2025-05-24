#!/usr/bin/env python
"""
Test script for BFCL task evaluation using the refactored agent evaluation framework.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path if it's not already there
project_root = Path(__file__).parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from reward_kit.agent.task_manager import TaskManager
from reward_kit.models import TaskDefinitionModel
from reward_kit.rewards import bfcl_reward

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_bfcl_task")


async def main():
    """Run the BFCL task evaluation."""
    logger.info("Testing BFCL task with refactored agent evaluation framework...")

    # Check that BFCL modules are accessible
    try:
        from reward_kit.agent.resources.bfcl_envs import (
            gorilla_file_system,
            posting_api,
        )

        logger.info("BFCL modules loaded successfully")
    except ImportError as e:
        logger.error(f"Failed to load BFCL modules: {e}")
        return

    # Set the model to use for agent
    os.environ["MODEL_AGENT"] = "openai/gpt-4"
    logger.info(f"Using model: {os.environ['MODEL_AGENT']}")

    # Create a TaskManager
    task_manager = TaskManager()

    # Path to a BFCL task
    task_path = str(
        project_root / "evaluations" / "bfcl" / "tasks" / "multi_turn_base_0.yaml"
    )
    logger.info(f"Loading BFCL task from: {task_path}")

    # Load and register the task
    task = task_manager._load_task_from_file(task_path)
    if task:
        # Modify the reward function path to use the imported bfcl_reward directly
        # This avoids the import error in the orchestrator
        task.reward_function_path = "examples.test_bfcl_task.get_bfcl_reward"

        task_id = task_manager.register_task(task)
        logger.info(f"Registered task: {task_id}")

        # Execute the task
        try:
            logger.info("Executing BFCL task...")
            results = await task_manager.execute_tasks([task_id])

            # Display results
            logger.info(f"Task execution completed.")
            for task_id, result in results.items():
                logger.info(f"Task {task_id} result: {result}")
        except Exception as e:
            logger.error(f"Error during task execution: {e}")
        finally:
            # Clean up resources
            await task_manager.cleanup()
    else:
        logger.error(f"Failed to load task from {task_path}")

    logger.info("Test completed.")


def get_bfcl_reward(*args, **kwargs):
    """Wrapper for bfcl_reward.bfcl_reward"""
    return bfcl_reward.bfcl_reward(*args, **kwargs)


if __name__ == "__main__":
    asyncio.run(main())
