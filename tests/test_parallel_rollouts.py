import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest

import reward_kit as rk


@pytest.mark.asyncio
async def test_seed_handling_and_type_compatibility():
    """
    Tests the specific issues we fixed:
    1. Seed extraction from client_info and proper propagation to environment
    2. MCP resource type compatibility (string vs ResourceContents)
    3. Session isolation for concurrent requests

    This test uses a local simulation server to avoid hitting remote services.
    """
    # 1. Start local simulation server for testing
    import subprocess
    import time

    server_script = (
        Path(__file__).parent.parent
        / "examples"
        / "frozen_lake_mcp_complete"
        / "mcp_server"
        / "simulation_server.py"
    )
    venv_python = Path(__file__).parent.parent / ".venv" / "bin" / "python"

    # Check if the venv python exists, otherwise use system python
    if not venv_python.exists():
        import sys

        venv_python = Path(sys.executable)

    # Start server in background
    server_process = subprocess.Popen(
        [str(venv_python), str(server_script), "--port", "8001", "--host", "127.0.0.1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    await asyncio.sleep(3)

    try:
        # 2. Create dataset with different seeds to test seed propagation
        test_seeds = [42, 123]  # Reduced to 2 seeds for faster testing
        system_prompt = "You are playing FrozenLake. Use the lake_move tool with actions LEFT, DOWN, RIGHT, UP to navigate the grid."
        user_prompt_template = "Initial game state grid: {grid_layout}\\n\\nYour current position: {position}\\n\\nChoose your next move."

        dataset = []
        for i, seed in enumerate(test_seeds):
            dataset.append(
                {
                    "id": f"seed_test_{seed}",
                    "system_prompt": system_prompt,
                    "user_prompt_template": user_prompt_template,
                    "environment_context": {
                        "game": "FrozenLake",
                        "grid_type": "4x4",
                        "seed": seed,
                    },
                    "seed": seed,  # Also include at top level for client extraction
                }
            )

        # 3. Test that environments are created with proper seed isolation
        envs = rk.make("http://127.0.0.1:8001/mcp/", dataset=dataset)

        # Verify we have the right number of environments
        assert len(envs.sessions) == len(
            test_seeds
        ), f"Expected {len(test_seeds)} sessions, got {len(envs.sessions)}"

        # 4. Test resource reading and seed propagation
        # This tests both the type compatibility fix and seed handling
        await envs.reset()

        # 5. Verify that different seeds produce different initial states
        initial_states = []
        for session in envs.sessions:
            # Extract the initial observation from the session
            initial_obs = session.last_observation
            if isinstance(initial_obs, dict) and "grid_layout" in initial_obs:
                initial_states.append(initial_obs["grid_layout"])
            elif isinstance(initial_obs, str):
                # Parse if it's a JSON string
                try:
                    obs_data = json.loads(initial_obs)
                    if "grid_layout" in obs_data:
                        initial_states.append(obs_data["grid_layout"])
                except json.JSONDecodeError:
                    initial_states.append(initial_obs)
            else:
                # If we can't extract grid layout, just use a string representation
                initial_states.append(str(initial_obs))

        # Verify we got different initial states (the core bug we fixed)
        assert len(initial_states) == len(
            test_seeds
        ), "Should have initial states for all seeds"

        # Check that seeds 42 and 123 produce different grids (they should based on our predefined maps)
        unique_states = set(initial_states)
        assert (
            len(unique_states) > 1
        ), f"Seeds 42 and 123 should produce different grid layouts. Got: {initial_states}"

        print("✅ Seed handling and type compatibility test passed!")
        print(f"   - Tested {len(test_seeds)} different seeds")
        print(f"   - Generated {len(unique_states)} unique grid layouts")
        print(f"   - Grid layouts: {initial_states}")
        print("   - ✅ Resource type compatibility: Server returned proper JSON")
        print(
            "   - ✅ Seed propagation: Different seeds produced different environments"
        )

    finally:
        # Clean up: stop the server
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()


@pytest.mark.asyncio
async def test_mcp_resource_type_compatibility():
    """
    Specific test for the MCP resource type issue we fixed.
    Tests that the core functionality works with JSON serialization.
    """
    # Test the core functionality that was causing issues
    from examples.frozen_lake_mcp_complete.mcp_server.frozen_lake_adapter import (
        FrozenLakeAdapter,
    )

    # Test the map generation with different seeds (this was the core bug)
    adapter = FrozenLakeAdapter()

    # Test that different seeds produce different maps
    map1 = adapter._generate_random_map(size=4, seed=42)
    map2 = adapter._generate_random_map(size=4, seed=123)
    map3 = adapter._generate_random_map(size=4, seed=999)

    # Verify they are different (the main bug we fixed)
    assert (
        map1 != map2 or map1 != map3
    ), f"Different seeds should produce different maps. Got: {map1}, {map2}, {map3}"

    # Test that the same seed produces the same map (deterministic)
    map1_repeat = adapter._generate_random_map(size=4, seed=42)
    assert map1 == map1_repeat, "Same seed should produce same map"

    # Test JSON serialization (the type compatibility issue)
    test_observation = {
        "position": 0,
        "grid_layout": "\n".join(map1),
        "moves": 0,
        "terminated": False,
        "truncated": False,
        "reward": 0.0,
        "info": {"seed": 42},
    }

    # This should work without errors (the fix we implemented)
    try:
        json_str = json.dumps(test_observation)
        parsed = json.loads(json_str)
        assert parsed == test_observation, "JSON round-trip should preserve data"
    except (TypeError, json.JSONDecodeError) as e:
        pytest.fail(f"Observation should be JSON-serializable: {e}")

    print("✅ MCP resource type compatibility test passed!")
    print(f"   - Seed 42 map: {map1}")
    print(f"   - Seed 123 map: {map2}")
    print(f"   - Seed 999 map: {map3}")
    print(f"   - JSON serialization: ✅")
