#!/usr/bin/env python3
"""
Pytest for verifying that different seeds and map sizes produce correct environments.

This test validates:
1. Different seeds produce different environments
2. 4x4 rollouts create 4x4 grids
3. 8x8 rollouts create 8x8 grids
4. Map names are properly propagated through the MCP server
"""

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

import reward_kit as rk


def load_rollouts_data(rollouts_file: Path) -> List[Dict[str, Any]]:
    """Load rollouts data from a JSONL file."""
    rollouts = []
    with open(rollouts_file, "r") as f:
        for line in f:
            if line.strip():
                rollouts.append(json.loads(line.strip()))
    return rollouts


class MCPServerManager:
    """Manages MCP server lifecycle for testing."""

    def __init__(self, server_script: str, port: int = 8002):
        self.server_script = server_script
        self.port = port
        self.process = None
        self.base_dir = Path(__file__).parent.parent / "mcp_server"
        self.log_file = Path(__file__).parent / f"server_seed_test_{port}.log"

    def start(self) -> None:
        """Start the MCP server."""
        if self.process:
            return

        # Remove old log file
        if self.log_file.exists():
            self.log_file.unlink()

        # Set environment for server
        env = os.environ.copy()
        env["PORT"] = str(self.port)

        # Start server process
        with open(self.log_file, "w") as f:
            self.process = subprocess.Popen(
                ["python", self.server_script, "--port", str(self.port)],
                cwd=self.base_dir,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )

        # Wait for server to start
        time.sleep(5)

        if self.process.poll() is not None:
            # Server failed to start, read logs
            logs = ""
            if self.log_file.exists():
                with open(self.log_file) as f:
                    logs = f.read()
            raise RuntimeError(f"Server failed to start. Logs:\n{logs}")

    def stop(self) -> None:
        """Stop the MCP server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None

    def get_logs(self) -> str:
        """Get server logs."""
        if self.log_file.exists():
            with open(self.log_file) as f:
                return f.read()
        return "No logs available"


@pytest.fixture
def server_manager():
    """Provide a managed MCP server for testing."""
    manager = MCPServerManager("simulation_server.py", port=8002)
    manager.start()
    yield manager
    manager.stop()


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.slow
async def test_seed_and_map_propagation(server_manager):
    """Test that seeds and map sizes are properly propagated through the MCP server."""
    print("show test running")

    # Load rollouts data from both files
    base_dir = Path(__file__).parent.parent
    rollouts_4x4_file = base_dir / "shared_data" / "rollouts.jsonl"
    rollouts_8x8_file = base_dir / "shared_data" / "rollouts_8x8.jsonl"

    rollouts_4x4 = load_rollouts_data(rollouts_4x4_file)
    rollouts_8x8 = load_rollouts_data(rollouts_8x8_file)

    assert len(rollouts_4x4) > 0, "Should have 4x4 rollouts"
    assert len(rollouts_8x8) > 0, "Should have 8x8 rollouts"

    # Test with rollouts from both files
    all_rollouts = rollouts_4x4 + rollouts_8x8
    environments = {}
    print("show all rollouts length", len(all_rollouts))
    for rollout in all_rollouts:
        env_context = rollout["environment_context"]
        seed = env_context["seed"]
        map_name = env_context["map_name"]
        rollout_id = rollout["id"]

        # Use the rollout data directly
        dataset = [rollout]

        # Create environment with the dataset
        envs = rk.make(
            "http://localhost:8002/mcp/", dataset=dataset, model_id="test-model"
        )

        # Reset environment to get initial state
        observations, _, _ = await envs.reset()

        # Extract the grid layout from the observation
        obs = observations[0]
        assert isinstance(obs, dict), f"Observation should be dict for {rollout_id}"
        assert "grid" in obs, f"Observation should have 'grid' for {rollout_id}"

        grid_layout = obs["grid"]
        print("show grid layout", grid_layout, "for seed", seed)
        grid_lines = grid_layout.split("\n")
        actual_size = f"{len(grid_lines)}x{len(grid_lines[0])}"

        environments[rollout_id] = {
            "grid": grid_layout,
            "observation": obs,
            "expected_size": map_name,
            "actual_size": actual_size,
            "seed": seed,
        }

        # Clean up
        await envs.close()

        # Small delay between tests
        time.sleep(0.5)

    # Analyze results

    # Check if grid sizes match expectations
    size_mismatches = []
    for rollout_id, env_data in environments.items():
        expected = env_data["expected_size"]
        actual = env_data["actual_size"]
        if expected != actual:
            size_mismatches.append(f"{rollout_id}: expected {expected}, got {actual}")

    assert not size_mismatches, f"Grid size mismatches: {size_mismatches}"

    # Verify we have the expected number of each size
    size_counts = {}
    for env_data in environments.values():
        size = env_data["actual_size"]
        size_counts[size] = size_counts.get(size, 0) + 1

    assert "4x4" in size_counts, "Should have 4x4 environments"
    assert "8x8" in size_counts, "Should have 8x8 environments"
    assert size_counts["4x4"] == len(
        rollouts_4x4
    ), f"Should have {len(rollouts_4x4)} 4x4 environments"
    assert size_counts["8x8"] == len(
        rollouts_8x8
    ), f"Should have {len(rollouts_8x8)} 8x8 environments"

    # Check that different seeds with same map size can produce different environments
    # Group by map size and collect grids
    grids_by_size = {"4x4": [], "8x8": []}
    seeds_by_size = {"4x4": [], "8x8": []}

    for env_data in environments.values():
        size = env_data["actual_size"]
        if size in grids_by_size:
            grids_by_size[size].append(env_data["grid"])
            seeds_by_size[size].append(env_data["seed"])

    # Check that we have multiple seeds for each size
    assert len(seeds_by_size["4x4"]) > 1, "Should have multiple 4x4 seeds to test"
    assert len(seeds_by_size["8x8"]) > 1, "Should have multiple 8x8 seeds to test"

    # Check that different seeds produce different grids for 4x4
    unique_4x4_grids = set(grids_by_size["4x4"])
    unique_4x4_seeds = set(seeds_by_size["4x4"])

    print(
        f"4x4 grids: Found {len(unique_4x4_grids)} unique grids from {len(unique_4x4_seeds)} unique seeds"
    )
    print(f"4x4 seeds: {seeds_by_size['4x4']}")
    print(f"4x4 unique grids: {len(unique_4x4_grids)}")

    # Check that different seeds produce different grids for 8x8
    unique_8x8_grids = set(grids_by_size["8x8"])
    unique_8x8_seeds = set(seeds_by_size["8x8"])

    print(
        f"8x8 grids: Found {len(unique_8x8_grids)} unique grids from {len(unique_8x8_seeds)} unique seeds"
    )
    print(f"8x8 seeds: {seeds_by_size['8x8']}")
    print(f"8x8 unique grids: {len(unique_8x8_grids)}")

    # Assert that different seeds should produce different grids
    # This is the core test: if seeds are properly propagated, different seeds should create different environments
    assert len(unique_4x4_grids) > 1, (
        f"Expected different 4x4 grids from different seeds, but got {len(unique_4x4_grids)} unique grids "
        f"from seeds {seeds_by_size['4x4']}. This suggests seeds are not being properly propagated."
    )

    assert len(unique_8x8_grids) > 1, (
        f"Expected different 8x8 grids from different seeds, but got {len(unique_8x8_grids)} unique grids "
        f"from seeds {seeds_by_size['8x8']}. This suggests seeds are not being properly propagated."
    )

    # Additional validation: verify that the number of unique grids matches the number of unique seeds
    # (assuming each seed should produce a unique grid)
    if len(unique_4x4_seeds) > 1:
        assert len(unique_4x4_grids) == len(unique_4x4_seeds), (
            f"Expected {len(unique_4x4_seeds)} unique 4x4 grids for {len(unique_4x4_seeds)} unique seeds, "
            f"but got {len(unique_4x4_grids)} unique grids. Seeds may not be deterministically creating environments."
        )

    if len(unique_8x8_seeds) > 1:
        assert len(unique_8x8_grids) == len(unique_8x8_seeds), (
            f"Expected {len(unique_8x8_seeds)} unique 8x8 grids for {len(unique_8x8_seeds)} unique seeds, "
            f"but got {len(unique_8x8_grids)} unique grids. Seeds may not be deterministically creating environments."
        )

    print("✅ Seed propagation test passed: Different seeds produce different grids")

    # Note: Due to the nature of random generation, some seeds might produce identical
    # grids even with different seeds. This is expected behavior and not a failure.
    # The important thing is that the grid sizes are correct and the system can
    # handle different configurations properly.


@pytest.mark.asyncio
@pytest.mark.integration
async def test_4x4_rollout_creates_4x4_grid(server_manager):
    """Test that a 4x4 rollout specifically creates a 4x4 grid."""

    # Create a simple 4x4 test rollout
    test_rollout = {
        "id": "test_4x4_specific",
        "system_prompt": "Test 4x4 grid creation",
        "user_prompt_template": "Current state: {observation}",
        "environment_context": {"game": "FrozenLake", "map_name": "4x4", "seed": 42},
    }

    # Create environment with the dataset
    envs = rk.make(
        "http://localhost:8002/mcp/", dataset=[test_rollout], model_id="test-model"
    )

    # Reset environment to get initial state
    observations, _, _ = await envs.reset()

    # Extract the grid layout from the observation
    obs = observations[0]
    assert isinstance(obs, dict), "Observation should be dict"
    assert "grid" in obs, "Observation should have 'grid'"

    grid_layout = obs["grid"]
    grid_lines = grid_layout.split("\n")
    actual_size = f"{len(grid_lines)}x{len(grid_lines[0])}"

    assert actual_size == "4x4", f"Expected 4x4 grid, got {actual_size}"
    assert len(grid_lines) == 4, f"Expected 4 rows, got {len(grid_lines)}"
    assert all(
        len(line) == 4 for line in grid_lines
    ), "Expected all rows to have 4 columns"

    # Clean up
    await envs.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_8x8_rollout_creates_8x8_grid(server_manager):
    """Test that an 8x8 rollout specifically creates an 8x8 grid."""

    # Create a simple 8x8 test rollout
    test_rollout = {
        "id": "test_8x8_specific",
        "system_prompt": "Test 8x8 grid creation",
        "user_prompt_template": "Current state: {observation}",
        "environment_context": {"game": "FrozenLake", "map_name": "8x8", "seed": 42},
    }

    # Create environment with the dataset
    envs = rk.make(
        "http://localhost:8002/mcp/", dataset=[test_rollout], model_id="test-model"
    )

    # Reset environment to get initial state
    observations, _, _ = await envs.reset()

    # Extract the grid layout from the observation
    obs = observations[0]
    assert isinstance(obs, dict), "Observation should be dict"
    assert "grid" in obs, "Observation should have 'grid'"

    grid_layout = obs["grid"]
    grid_lines = grid_layout.split("\n")
    actual_size = f"{len(grid_lines)}x{len(grid_lines[0])}"

    assert actual_size == "8x8", f"Expected 8x8 grid, got {actual_size}"
    assert len(grid_lines) == 8, f"Expected 8 rows, got {len(grid_lines)}"
    assert all(
        len(line) == 8 for line in grid_lines
    ), "Expected all rows to have 8 columns"

    # Clean up
    await envs.close()


if __name__ == "__main__":
    # Allow running as script for debugging
    pytest.main([__file__, "-v", "-s"])
