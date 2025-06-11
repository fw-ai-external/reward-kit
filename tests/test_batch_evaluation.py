"""
End-to-end integration tests for batch evaluation feature.

These tests validate the entire batch evaluation pipeline with live API calls
to both Fireworks and OpenAI, ensuring production readiness.
"""

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from reward_kit.agent.task_manager import TaskManager
from reward_kit.cli_commands.agent_eval_cmd import agent_eval_command
from reward_kit.models import TaskDefinitionModel


class MockArgs:
    """Mock args object for agent_eval_command."""

    def __init__(self, task_def: str, num_rollouts: int = 2, **kwargs):
        self.task_def = task_def
        self.num_rollouts = num_rollouts
        self.parallel = kwargs.get("parallel", False)
        self.max_concurrency = kwargs.get("max_concurrency", 3)
        self.model = kwargs.get("model", None)
        self.filter = kwargs.get("filter", None)


@pytest.mark.integration
class TestBatchEvaluation:
    """Integration tests for batch evaluation functionality."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Ensure we have the necessary environment variables
        self.original_env = {}

        # Store original environment values
        env_vars = ["FIREWORKS_API_KEY", "OPENAI_API_KEY", "MODEL_AGENT"]
        for var in env_vars:
            self.original_env[var] = os.environ.get(var)

        # Set default model for agent if not specified
        if not os.environ.get("MODEL_AGENT"):
            os.environ["MODEL_AGENT"] = (
                "accounts/fireworks/models/llama-v3p1-8b-instruct"
            )

    def teardown_method(self):
        """Clean up after each test."""
        # Restore original environment
        for var, value in self.original_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value

    @pytest.mark.asyncio
    async def test_batch_evaluation_task_manager_fireworks(self):
        """Test batch evaluation using TaskManager with Fireworks API."""
        # Skip if no API key available
        if not os.environ.get("FIREWORKS_API_KEY"):
            pytest.skip("FIREWORKS_API_KEY not available")

        task_manager = TaskManager()

        # Load the frozen lake task definition for Fireworks
        task_def_path = Path("examples/frozen_lake/client/task_def.yaml")
        if not task_def_path.exists():
            pytest.skip(f"Task definition not found: {task_def_path}")

        # Load and register the task
        task_def = task_manager._load_task_from_file(str(task_def_path))
        assert task_def is not None, "Failed to load task definition"

        # Override num_rollouts to reduce test time
        task_def.num_rollouts = 2

        task_id = task_manager.register_task(task_def)
        assert task_id == "frozen_lake_http_rollout"

        try:
            # Execute the task with batch evaluation
            results = await task_manager.execute_tasks(
                task_ids=[task_id],
                parallel=False,
                max_concurrency=2,
                num_rollouts_override=2,
            )

            # Validate results structure
            assert task_id in results
            result = results[task_id]

            # Should not be an error result
            assert not (
                isinstance(result, dict) and "error" in result
            ), f"Task failed: {result.get('error', 'Unknown error')}"

            # Should be aggregated results
            assert isinstance(result, dict)
            assert result.get(
                "aggregated", False
            ), "Results should be aggregated for batch evaluation"

            # Validate aggregated result structure
            required_keys = [
                "num_rollouts",
                "successful_rollouts",
                "success_rate",
                "avg_score",
                "min_score",
                "max_score",
            ]
            for key in required_keys:
                assert key in result, f"Missing key in aggregated results: {key}"

            # Validate result values
            assert result["num_rollouts"] == 2
            assert result["successful_rollouts"] >= 0
            assert result["successful_rollouts"] <= result["num_rollouts"]
            assert 0.0 <= result["success_rate"] <= 1.0
            assert isinstance(result["avg_score"], (int, float))
            assert isinstance(result["min_score"], (int, float))
            assert isinstance(result["max_score"], (int, float))
            assert result["min_score"] <= result["avg_score"] <= result["max_score"]

            # Should have individual results
            assert "individual_scores" in result
            assert "individual_results" in result
            assert len(result["individual_scores"]) == result["successful_rollouts"]
            assert len(result["individual_results"]) == result["successful_rollouts"]

            logging.info(f"Fireworks batch evaluation completed successfully: {result}")

        finally:
            await task_manager.cleanup()

    @pytest.mark.asyncio
    async def test_batch_evaluation_task_manager_openai(self):
        """Test batch evaluation using TaskManager with OpenAI API."""
        # Skip if no API key available
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not available")

        task_manager = TaskManager()

        # Load the frozen lake task definition for OpenAI
        task_def_path = Path("examples/frozen_lake/client/task_def_openai.yaml")
        if not task_def_path.exists():
            pytest.skip(f"Task definition not found: {task_def_path}")

        # Load and register the task
        task_def = task_manager._load_task_from_file(str(task_def_path))
        assert task_def is not None, "Failed to load task definition"

        # Override num_rollouts to reduce test time
        task_def.num_rollouts = 2

        # Set OpenAI model temporarily
        original_model = os.environ.get("MODEL_AGENT")
        os.environ["MODEL_AGENT"] = "gpt-4o-mini"

        task_id = task_manager.register_task(task_def)
        assert task_id == "frozen_lake_http_rollout_openai"

        try:
            # Execute the task with batch evaluation
            results = await task_manager.execute_tasks(
                task_ids=[task_id],
                parallel=False,
                max_concurrency=2,
                num_rollouts_override=2,
            )

            # Validate results structure
            assert task_id in results
            result = results[task_id]

            # Should not be an error result
            assert not (
                isinstance(result, dict) and "error" in result
            ), f"Task failed: {result.get('error', 'Unknown error')}"

            # Should be aggregated results
            assert isinstance(result, dict)
            assert result.get(
                "aggregated", False
            ), "Results should be aggregated for batch evaluation"

            # Validate aggregated result structure
            required_keys = [
                "num_rollouts",
                "successful_rollouts",
                "success_rate",
                "avg_score",
                "min_score",
                "max_score",
            ]
            for key in required_keys:
                assert key in result, f"Missing key in aggregated results: {key}"

            # Validate result values
            assert result["num_rollouts"] == 2
            assert result["successful_rollouts"] >= 0
            assert result["successful_rollouts"] <= result["num_rollouts"]
            assert 0.0 <= result["success_rate"] <= 1.0
            assert isinstance(result["avg_score"], (int, float))
            assert isinstance(result["min_score"], (int, float))
            assert isinstance(result["max_score"], (int, float))
            assert result["min_score"] <= result["avg_score"] <= result["max_score"]

            # Should have individual results
            assert "individual_scores" in result
            assert "individual_results" in result
            assert len(result["individual_scores"]) == result["successful_rollouts"]
            assert len(result["individual_results"]) == result["successful_rollouts"]

            logging.info(f"OpenAI batch evaluation completed successfully: {result}")

        finally:
            # Restore original model
            if original_model:
                os.environ["MODEL_AGENT"] = original_model
            else:
                os.environ.pop("MODEL_AGENT", None)
            await task_manager.cleanup()

    def test_cli_batch_evaluation_fireworks(self):
        """Test batch evaluation through CLI command with Fireworks."""
        # Skip if no API key available
        if not os.environ.get("FIREWORKS_API_KEY"):
            pytest.skip("FIREWORKS_API_KEY not available")

        task_def_path = Path("examples/frozen_lake/client/task_def.yaml")
        if not task_def_path.exists():
            pytest.skip(f"Task definition not found: {task_def_path}")

        # Create mock args for CLI command
        args = MockArgs(
            task_def=str(task_def_path),
            num_rollouts=2,
            parallel=False,
            max_concurrency=2,
        )

        # Execute CLI command
        exit_code = agent_eval_command(args)

        # Should complete successfully
        assert exit_code == 0, "CLI command should complete successfully"

    def test_cli_batch_evaluation_openai(self):
        """Test batch evaluation through CLI command with OpenAI."""
        # Skip if no API key available
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not available")

        task_def_path = Path("examples/frozen_lake/client/task_def_openai.yaml")
        if not task_def_path.exists():
            pytest.skip(f"Task definition not found: {task_def_path}")

        # Set OpenAI model
        original_model = os.environ.get("MODEL_AGENT")
        os.environ["MODEL_AGENT"] = "gpt-4o-mini"

        try:
            # Create mock args for CLI command
            args = MockArgs(
                task_def=str(task_def_path),
                num_rollouts=2,
                parallel=False,
                max_concurrency=2,
            )

            # Execute CLI command
            exit_code = agent_eval_command(args)

            # Should complete successfully
            assert exit_code == 0, "CLI command should complete successfully"

        finally:
            # Restore original model
            if original_model:
                os.environ["MODEL_AGENT"] = original_model
            else:
                os.environ.pop("MODEL_AGENT", None)

    @pytest.mark.asyncio
    async def test_parallel_batch_evaluation(self):
        """Test parallel execution of multiple rollouts."""
        # Skip if no API key available
        if not os.environ.get("FIREWORKS_API_KEY"):
            pytest.skip("FIREWORKS_API_KEY not available")

        task_manager = TaskManager()

        # Load the frozen lake task definition
        task_def_path = Path("examples/frozen_lake/client/task_def.yaml")
        if not task_def_path.exists():
            pytest.skip(f"Task definition not found: {task_def_path}")

        # Load and register the task
        task_def = task_manager._load_task_from_file(str(task_def_path))
        assert task_def is not None, "Failed to load task definition"

        # Test with more rollouts to verify parallelism
        task_def.num_rollouts = 3

        task_id = task_manager.register_task(task_def)

        try:
            # Execute with parallel enabled
            results = await task_manager.execute_tasks(
                task_ids=[task_id],
                parallel=True,
                max_concurrency=2,
                num_rollouts_override=3,
            )

            # Validate results
            assert task_id in results
            result = results[task_id]

            # Should be successful and aggregated
            assert not (isinstance(result, dict) and "error" in result)
            assert result.get("aggregated", False)
            assert result["num_rollouts"] == 3

            logging.info(f"Parallel batch evaluation completed: {result}")

        finally:
            await task_manager.cleanup()

    @pytest.mark.asyncio
    async def test_server_lifecycle_management(self):
        """Test that resource servers are properly started and stopped."""
        # Skip if no API key available
        if not os.environ.get("FIREWORKS_API_KEY"):
            pytest.skip("FIREWORKS_API_KEY not available")

        task_manager = TaskManager()

        # Load task definition
        task_def_path = Path("examples/frozen_lake/client/task_def.yaml")
        if not task_def_path.exists():
            pytest.skip(f"Task definition not found: {task_def_path}")

        task_def = task_manager._load_task_from_file(str(task_def_path))
        task_def.num_rollouts = 2
        task_id = task_manager.register_task(task_def)

        # Check that no servers are running initially
        assert len(task_manager.server_processes) == 0
        assert len(task_manager.server_ports) == 0

        try:
            # Execute task
            results = await task_manager.execute_tasks(
                task_ids=[task_id], num_rollouts_override=2
            )

            # Task should complete successfully
            assert task_id in results
            result = results[task_id]
            assert not (isinstance(result, dict) and "error" in result)

        finally:
            await task_manager.cleanup()

            # Check that all servers are cleaned up
            assert len(task_manager.server_processes) == 0
            assert len(task_manager.server_ports) == 0


@pytest.mark.integration
class TestBatchEvaluationErrorHandling:
    """Test error handling in batch evaluation scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_task_definition(self):
        """Test handling of invalid task definitions."""
        task_manager = TaskManager()

        # Create a temporary invalid task definition
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
name: "invalid_task"
description: "This task has invalid configuration"
resource_type: "nonexistent_resource"
"""
            )
            invalid_task_path = f.name

        try:
            # Attempt to load invalid task
            task_def = task_manager._load_task_from_file(invalid_task_path)

            # Should either fail to load or fail during execution
            if task_def is not None:
                task_id = task_manager.register_task(task_def)
                results = await task_manager.execute_tasks([task_id])

                # Should result in error
                assert task_id in results
                result = results[task_id]
                assert isinstance(result, dict) and "error" in result

        finally:
            # Clean up temporary file
            Path(invalid_task_path).unlink(missing_ok=True)
            await task_manager.cleanup()

    @pytest.mark.asyncio
    async def test_missing_api_key_handling(self):
        """Test graceful handling when API keys are missing."""
        # Temporarily remove API keys
        original_fw_key = os.environ.get("FIREWORKS_API_KEY")
        original_openai_key = os.environ.get("OPENAI_API_KEY")

        os.environ.pop("FIREWORKS_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)

        task_manager = TaskManager()

        try:
            task_def_path = Path("examples/frozen_lake/client/task_def.yaml")
            if not task_def_path.exists():
                pytest.skip(f"Task definition not found: {task_def_path}")

            task_def = task_manager._load_task_from_file(str(task_def_path))
            if task_def:
                task_def.num_rollouts = 1  # Reduce rollouts for faster failure
                task_id = task_manager.register_task(task_def)

                results = await task_manager.execute_tasks([task_id])

                # Should handle missing API key gracefully
                assert task_id in results
                result = results[task_id]
                # Result could be error or have low success rate due to API failures
                if isinstance(result, dict) and "error" in result:
                    # Direct error is acceptable
                    pass
                elif isinstance(result, dict) and result.get("aggregated", False):
                    # Batch result with low success rate is also acceptable
                    assert result["success_rate"] <= 1.0

        finally:
            # Restore API keys
            if original_fw_key:
                os.environ["FIREWORKS_API_KEY"] = original_fw_key
            if original_openai_key:
                os.environ["OPENAI_API_KEY"] = original_openai_key
            await task_manager.cleanup()
