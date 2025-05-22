"""
Task Manager for the Agent Evaluation Framework V2.
Coordinates multiple tasks and their associated resources.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import asyncio
import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..models import TaskDefinitionModel
from .orchestrator import Orchestrator
from .resource_abc import ForkableResource
from .resource_pool import ResourcePool


class TaskManager:
    """
    Manages the execution of multiple agent evaluation tasks.
    Coordinates resources, orchestrators, and execution flows.
    """

    def __init__(self) -> None:
        """Initialize the TaskManager with an empty task registry."""
        self.tasks: Dict[str, TaskDefinitionModel] = {}
        self.resource_pool = ResourcePool()
        self.logger = logging.getLogger("TaskManager")
        self.orchestrators: Dict[str, Orchestrator] = {}

    def register_task(self, task_definition: TaskDefinitionModel) -> str:
        """
        Register a task with the manager.

        Args:
            task_definition: A validated TaskDefinitionModel instance

        Returns:
            task_id: A unique identifier for the registered task
        """
        task_id = task_definition.name
        if task_id in self.tasks:
            self.logger.warning(f"Task '{task_id}' is already registered. Overwriting.")

        self.tasks[task_id] = task_definition
        self.logger.info(f"Registered task: {task_id}")
        return task_id

    def register_tasks_from_directory(self, directory_path: str) -> List[str]:
        """
        Register all task definition files from a directory.

        Args:
            directory_path: Path to directory containing task definition files

        Returns:
            task_ids: List of task IDs that were successfully registered
        """
        task_ids = []
        dir_path = Path(directory_path)

        if not dir_path.exists() or not dir_path.is_dir():
            self.logger.error(
                f"Directory not found or not a directory: {directory_path}"
            )
            return task_ids

        for file_path in dir_path.glob("*.y*ml"):
            try:
                task_def = self._load_task_from_file(str(file_path))
                if task_def:
                    task_id = self.register_task(task_def)
                    task_ids.append(task_id)
            except Exception as e:
                self.logger.error(f"Error loading task from {file_path}: {e}")

        for file_path in dir_path.glob("*.json"):
            try:
                task_def = self._load_task_from_file(str(file_path))
                if task_def:
                    task_id = self.register_task(task_def)
                    task_ids.append(task_id)
            except Exception as e:
                self.logger.error(f"Error loading task from {file_path}: {e}")

        self.logger.info(f"Registered {len(task_ids)} tasks from {directory_path}")
        return task_ids

    def _load_task_from_file(self, file_path: str) -> Optional[TaskDefinitionModel]:
        """
        Load and validate a task definition from a file.

        Args:
            file_path: Path to the task definition file

        Returns:
            task_def: A validated TaskDefinitionModel instance or None if loading fails
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists() or not file_path_obj.is_file():
            self.logger.error(f"File not found or not a file: {file_path}")
            return None

        try:
            # Try to load as YAML first
            try:
                import yaml

                with open(file_path, "r") as f:
                    task_data = yaml.safe_load(f)
            except ImportError:
                # If PyYAML is not available, try JSON
                with open(file_path, "r") as f:
                    task_data = json.load(f)
            except Exception:
                # If YAML loading fails, try JSON
                with open(file_path, "r") as f:
                    task_data = json.load(f)

            # Validate with Pydantic model
            task_def = TaskDefinitionModel.model_validate(task_data)
            return task_def
        except Exception as e:
            self.logger.error(f"Error loading task definition from {file_path}: {e}")
            return None

    async def prepare_task(self, task_id: str) -> bool:
        """
        Prepare a task for execution by setting up its resources.

        Args:
            task_id: Identifier of the task to prepare

        Returns:
            success: True if preparation was successful, False otherwise
        """
        if task_id not in self.tasks:
            self.logger.error(f"Task '{task_id}' is not registered.")
            return False

        task_def = self.tasks[task_id]

        # Create an orchestrator for this specific task
        orchestrator = Orchestrator(task_definition=task_def)
        self.orchestrators[task_id] = orchestrator

        # Prepare the resources for this task
        try:
            # Resource setup is handled by the orchestrator
            await orchestrator.setup_base_resource()
            return True
        except Exception as e:
            self.logger.error(f"Error preparing resources for task '{task_id}': {e}")
            return False

    async def execute_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Execute a registered task.

        Args:
            task_id: Identifier of the task to execute

        Returns:
            result: Dictionary containing execution results or None if execution fails
        """
        if task_id not in self.tasks:
            self.logger.error(f"Task '{task_id}' is not registered.")
            return None

        if task_id not in self.orchestrators:
            self.logger.info(
                f"Task '{task_id}' orchestrator not initialized. Preparing task..."
            )
            success = await self.prepare_task(task_id)
            if not success:
                self.logger.error(f"Failed to prepare task '{task_id}'.")
                return None

        orchestrator = self.orchestrators[task_id]

        try:
            self.logger.info(f"Executing task '{task_id}'...")
            result = await orchestrator.execute_task_poc()
            self.logger.info(f"Task '{task_id}' execution completed.")
            return result
        except Exception as e:
            self.logger.error(f"Error executing task '{task_id}': {e}", exc_info=True)
            return {"error": str(e)}

    async def execute_tasks(
        self,
        task_ids: Optional[List[str]] = None,
        parallel: bool = False,
        max_concurrency: int = 3,
    ) -> Dict[str, Any]:
        """
        Execute multiple tasks sequentially or in parallel.

        Args:
            task_ids: List of task IDs to execute. If None, execute all registered tasks.
            parallel: If True, execute tasks in parallel; otherwise, execute sequentially
            max_concurrency: Maximum number of tasks to execute in parallel

        Returns:
            results: Dictionary mapping task IDs to execution results
        """
        # If task_ids is None, use all registered tasks
        task_ids_to_execute = (
            task_ids if task_ids is not None else list(self.tasks.keys())
        )

        # Validate task IDs
        valid_task_ids = [tid for tid in task_ids_to_execute if tid in self.tasks]
        if len(valid_task_ids) != len(task_ids_to_execute):
            invalid_task_ids = set(task_ids_to_execute) - set(valid_task_ids)
            self.logger.warning(f"Some task IDs are not registered: {invalid_task_ids}")

        if not valid_task_ids:
            self.logger.error("No valid tasks to execute.")
            return {}

        results = {}

        if parallel and len(valid_task_ids) > 1:
            # Execute tasks in parallel with concurrency limit
            self.logger.info(
                f"Executing {len(valid_task_ids)} tasks in parallel with max concurrency {max_concurrency}."
            )

            # Prepare all tasks first
            prepare_tasks = [self.prepare_task(task_id) for task_id in valid_task_ids]
            prepare_results = await asyncio.gather(*prepare_tasks)

            # Filter out tasks that failed preparation
            prepared_task_ids = [
                tid for tid, success in zip(valid_task_ids, prepare_results) if success
            ]

            if not prepared_task_ids:
                self.logger.error("No tasks were successfully prepared.")
                return results

            # Execute tasks with semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrency)

            async def execute_with_semaphore(tid: str) -> Tuple[str, Any]:
                async with semaphore:
                    return tid, await self.execute_task(tid)

            execution_tasks = [execute_with_semaphore(tid) for tid in prepared_task_ids]
            execution_results = await asyncio.gather(*execution_tasks)

            # Collect results
            results = {tid: result for tid, result in execution_results}
        else:
            # Execute tasks sequentially
            self.logger.info(f"Executing {len(valid_task_ids)} tasks sequentially.")
            for task_id in valid_task_ids:
                if await self.prepare_task(task_id):
                    results[task_id] = await self.execute_task(task_id)
                else:
                    results[task_id] = {"error": "Task preparation failed"}

        return results

    async def cleanup(self, task_ids: Optional[List[str]] = None) -> None:
        """
        Clean up resources for specified tasks or all tasks.

        Args:
            task_ids: List of task IDs to clean up. If None, clean up all tasks.
        """
        task_ids_to_cleanup = (
            task_ids if task_ids is not None else list(self.orchestrators.keys())
        )

        for task_id in task_ids_to_cleanup:
            if task_id in self.orchestrators:
                orchestrator = self.orchestrators[task_id]
                if orchestrator.base_resource:
                    try:
                        await orchestrator.base_resource.close()
                        self.logger.info(f"Cleaned up resources for task '{task_id}'.")
                    except Exception as e:
                        self.logger.error(
                            f"Error cleaning up resources for task '{task_id}': {e}"
                        )
                del self.orchestrators[task_id]
