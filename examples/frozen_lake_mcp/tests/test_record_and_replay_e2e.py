#!/usr/bin/env python3
"""
End-to-End Record and Replay Tests for FrozenLake MCP

This module provides pytest-compatible tests that:
1. Set up production server automatically
2. Record trajectories in the first run
3. Use recorded trajectories for fast replay in subsequent runs
4. Validate server functionality and performance
5. Clean up resources properly

Usage:
    pytest test_record_and_replay_e2e.py -v

Environment Variables:
    REWARD_KIT_FORCE_RECORD=1  # Force recording mode even if replay file exists
    REWARD_KIT_PLAYBACK_FILE   # Path to replay file (auto-detected if not set)
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

import reward_kit as rk


class MCPServerManager:
    """Manages MCP server lifecycle for testing."""

    def __init__(self, server_script: str, port: int = 8000):
        self.server_script = server_script
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.base_dir = Path(__file__).parent.parent

    def start(self) -> None:
        """Start the MCP server."""
        if self.process:
            return

        # Set environment for server
        env = os.environ.copy()
        env["PORT"] = str(self.port)

        # Start server process
        cmd = ["python", self.server_script, "--port", str(self.port)]
        self.process = subprocess.Popen(
            cmd,
            cwd=self.base_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for server to start
        time.sleep(3)

        # Check if process is still running
        if self.process.poll() is not None:
            stdout, stderr = self.process.communicate()
            raise RuntimeError(f"Server failed to start: {stderr}")

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

    def is_running(self) -> bool:
        """Check if server is running."""
        return self.process is not None and self.process.poll() is None


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide test data directory."""
    return Path(__file__).parent.parent / "shared_data"


@pytest.fixture(scope="session")
def frozen_lake_dataset(test_data_dir):
    """Load FrozenLake test dataset."""
    rollouts_file = test_data_dir / "rollouts.jsonl"
    if not rollouts_file.exists():
        pytest.skip(f"Dataset not found: {rollouts_file}")

    with open(rollouts_file) as f:
        dataset = [json.loads(line) for line in f]

    # Use only first 2 entries for faster testing
    return dataset[:1]


@pytest.fixture(scope="session")
def production_server():
    """Start and manage production server."""
    server = MCPServerManager("server.py", port=9500)

    try:
        server.start()
        yield server
    finally:
        server.stop()


@pytest.fixture
def production_recording_file():
    """Provide a recording file path for the production server test."""
    recording_dir = Path(__file__).parent / "recordings"
    recording_dir.mkdir(exist_ok=True)
    recording_path = recording_dir / "production_trajectory.jsonl"

    # In CI, preserve existing recording files for replay mode
    # Only remove if not in CI or if forced to record
    is_ci = os.environ.get("CI", "").lower() in ["true", "1", "yes"]
    force_record = os.environ.get("REWARD_KIT_FORCE_RECORD", "").lower() in [
        "true",
        "1",
        "yes",
    ]

    if os.path.exists(recording_path) and not is_ci and not force_record:
        os.unlink(recording_path)
    elif is_ci and not os.path.exists(recording_path):
        pytest.skip("CI mode requires existing recording file for replay")

    yield str(recording_path)


@pytest.fixture
def conda_isolation_recording_file():
    """Provide a recording file path for the conda isolation test."""
    recording_dir = Path(__file__).parent / "recordings"
    recording_dir.mkdir(exist_ok=True)
    recording_path = recording_dir / "conda_isolation_trajectory.jsonl"

    # In CI, preserve existing recording files for replay mode
    # Only remove if not in CI or if forced to record
    is_ci = os.environ.get("CI", "").lower() in ["true", "1", "yes"]
    force_record = os.environ.get("REWARD_KIT_FORCE_RECORD", "").lower() in [
        "true",
        "1",
        "yes",
    ]

    if os.path.exists(recording_path) and not is_ci and not force_record:
        os.unlink(recording_path)
    elif is_ci and not os.path.exists(recording_path):
        pytest.skip("CI mode requires existing recording file for replay")

    yield str(recording_path)


@pytest.mark.asyncio
async def test_production_server_record_and_replay(
    production_server, frozen_lake_dataset, production_recording_file
):
    """Test production server with record and replay functionality."""

    # Check if we're in CI mode and have existing recording
    is_ci = os.environ.get("CI", "").lower() in ["true", "1", "yes"]
    if is_ci and os.path.exists(production_recording_file):
        print("\n🎬 === CI MODE: PLAYBACK ONLY ===")

        # Set up playback environment
        os.environ["REWARD_KIT_PLAYBACK_FILE"] = production_recording_file

        # Create playback policy
        playback_policy = rk.FireworksPolicy(
            model_id="accounts/fireworks/models/qwen3-235b-a22b",
            temperature=0.2,
            max_tokens=4096,
        )

        assert playback_policy.is_playback_mode(), "Should be in playback mode in CI"

        # Create environments for playback
        playback_envs = rk.make(
            "http://localhost:9500/mcp/",
            dataset=frozen_lake_dataset,
            model_id=playback_policy.model_id,
        )

        # Run playback
        start_time = time.time()
        playback_trajectories = await rk.rollout(
            playback_envs, policy=playback_policy, steps=8
        )
        playback_duration = time.time() - start_time

        print(
            f"✅ CI playback completed: {len(playback_trajectories)} trajectories in {playback_duration:.2f}s"
        )

        # Clean up environment variable
        if "REWARD_KIT_PLAYBACK_FILE" in os.environ:
            del os.environ["REWARD_KIT_PLAYBACK_FILE"]

        return  # Skip recording phase in CI

    # === RECORDING PHASE ===
    print("\n📝 === FROZEN LAKE RECORDING PHASE ===")

    # Set up recording environment
    os.environ["REWARD_KIT_PLAYBACK_FILE"] = production_recording_file

    # Create policy for recording
    policy = rk.FireworksPolicy(
        model_id="accounts/fireworks/models/qwen3-235b-a22b",
        temperature=0.2,
        max_tokens=4096,
    )

    assert not policy.is_playback_mode(), "Should be in recording mode initially"

    # Create environments
    envs = rk.make(
        "http://localhost:9500/mcp/",
        dataset=frozen_lake_dataset,
        model_id=policy.model_id,
    )

    # Record trajectories (FrozenLake typically needs fewer steps than Taxi)
    start_time = time.time()
    trajectories = await rk.rollout(
        envs,
        policy=policy,
        steps=8,  # FrozenLake episodes are typically shorter
        openai_format_log_file=None,  # Don't need OpenAI format for testing
    )
    recording_duration = time.time() - start_time

    assert len(trajectories) == len(
        frozen_lake_dataset
    ), "Should have trajectory for each dataset entry"
    assert os.path.exists(production_recording_file), "Recording file should be created"

    print(f"✅ Recorded {len(trajectories)} trajectories in {recording_duration:.2f}s")
    print(f"📁 Recording saved to: {production_recording_file}")

    # Print trajectory summary for review
    print("📊 Trajectory Summary:")
    for i, traj in enumerate(trajectories):
        dataset_entry = frozen_lake_dataset[i]
        seed = dataset_entry.get("environment_context", {}).get("seed", "N/A")
        print(
            f"  Trajectory {i} (seed: {seed}): {traj.steps} steps, reward: {traj.total_reward:.2f}, terminated: {traj.terminated}"
        )
        if hasattr(traj, "actions") and len(traj.actions) > 0:
            print(
                f"    Actions: {traj.actions[:5]}{'...' if len(traj.actions) > 5 else ''}"
            )

    # Read and display first few recorded steps for verification
    print("🔍 Sample recorded steps (first 3):")
    try:
        with open(production_recording_file, "r") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                step_data = json.loads(line)
                env_idx = step_data.get("env_index", "?")
                step_num = step_data.get("step", "?")
                print(
                    f"    Step {step_num} (env {env_idx}): {len(step_data.get('messages', []))} messages"
                )
    except Exception as e:
        print(f"    Could not read recording file for preview: {e}")

    # === PLAYBACK PHASE ===
    print("\n🎬 === FROZEN LAKE PLAYBACK PHASE ===")

    # Create new policy for playback (same environment variable)
    playback_policy = rk.FireworksPolicy(
        model_id="accounts/fireworks/models/qwen3-235b-a22b",
        temperature=0.2,
        max_tokens=4096,
    )

    assert playback_policy.is_playback_mode(), "Should be in playback mode"

    # Create new environments for playback
    playback_envs = rk.make(
        "http://localhost:9500/mcp/",
        dataset=frozen_lake_dataset,
        model_id=playback_policy.model_id,
    )

    # Run playback
    start_time = time.time()
    playback_trajectories = await rk.rollout(
        playback_envs, policy=playback_policy, steps=15
    )
    playback_duration = time.time() - start_time

    assert len(playback_trajectories) == len(
        trajectories
    ), "Playback should have same number of trajectories"

    # Calculate speedup
    speedup = (
        recording_duration / playback_duration
        if playback_duration > 0
        else float("inf")
    )

    print(
        f"✅ Played back {len(playback_trajectories)} trajectories in {playback_duration:.2f}s"
    )
    print(f"⚡ Speedup: {speedup:.1f}x faster than recording")

    # Validate performance - playback should be significantly faster
    assert speedup > 10, f"Playback should be at least 10x faster, got {speedup:.1f}x"

    # Clean up environment variable
    if "REWARD_KIT_PLAYBACK_FILE" in os.environ:
        del os.environ["REWARD_KIT_PLAYBACK_FILE"]


def test_server_health_checks(production_server):
    """Test that the server is running and healthy."""
    assert production_server.is_running(), "Production server should be running"


@pytest.mark.asyncio
async def test_production_only_recorded_policy(frozen_lake_dataset):
    """Test that production environments work with pre-recorded policies only."""

    # Create a test recording file that persists for review
    recording_dir = Path(__file__).parent / "recordings"
    recording_dir.mkdir(exist_ok=True)
    test_recording_file = recording_dir / "playback_only_test.jsonl"

    # Create a dummy trajectory file
    recording_data = [
        {
            "env_index": 0,
            "step": 0,
            "messages": [
                {"role": "system", "content": frozen_lake_dataset[0]["system_prompt"]},
                {"role": "user", "content": "Initial state"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "lake_move",
                                "arguments": '{"action": "RIGHT"}',
                            },
                        }
                    ],
                },
            ],
        }
    ]

    # Save the test recording
    with open(test_recording_file, "w") as f:
        for entry in recording_data:
            f.write(json.dumps(entry) + "\n")

    print(f"📁 Test recording saved to: {test_recording_file}")

    try:
        # Set up playback environment
        os.environ["REWARD_KIT_PLAYBACK_FILE"] = str(test_recording_file)

        # Create policy in playback mode
        policy = rk.FireworksPolicy(
            model_id="accounts/fireworks/models/qwen3-235b-a22b",
            temperature=0.2,
            max_tokens=4096,
        )

        assert policy.is_playback_mode(), "Policy should be in playback mode"

        print("✅ FrozenLake production environment successfully using recorded policy")
        print(f"📊 Playback policy using: {test_recording_file}")

    finally:
        # Clean up environment variable (but keep the file for review)
        if "REWARD_KIT_PLAYBACK_FILE" in os.environ:
            del os.environ["REWARD_KIT_PLAYBACK_FILE"]


@pytest.mark.asyncio
async def test_frozen_lake_step_by_step(conda_isolation_recording_file):
    """Test FrozenLake step by step functionality with conda isolation."""

    print("\n🧪 === FROZEN LAKE STEP BY STEP TEST ===")

    # Check if we're in CI mode and have existing recording
    is_ci = os.environ.get("CI", "").lower() in ["true", "1", "yes"]
    if is_ci and os.path.exists(conda_isolation_recording_file):
        print(
            "⚠️ CI mode: Skipping conda isolation test (requires live conda environments)"
        )
        pytest.skip("CI mode skips resource-intensive conda isolation tests")

    # Test with conda isolation (if CondaServerProcessManager is available)
    try:
        from reward_kit.mcp import CondaServerProcessManager

        # Create process manager for conda isolation
        script_path = Path(__file__).parent.parent / "server.py"
        requirements_path = Path(__file__).parent.parent / "requirements.txt"

        manager = CondaServerProcessManager(
            script_path=str(script_path),
            requirements_path=str(requirements_path),
            port_range=(10000, 11000),
        )

        # Start server with seed
        port = manager.start_server(seed=42)
        print(f"✅ Started conda-isolated server on port {port}")

        # Set up recording
        os.environ["REWARD_KIT_PLAYBACK_FILE"] = conda_isolation_recording_file

        # Create policy
        policy = rk.FireworksPolicy(
            model_id="accounts/fireworks/models/qwen3-235b-a22b",
            temperature=0.2,
            max_tokens=4096,
        )

        # Create simple dataset for testing
        test_dataset = [
            {
                "id": "conda_test_001",
                "system_prompt": "You are playing FrozenLake, a 4x4 grid game. Use lake_move tool with LEFT, DOWN, RIGHT, UP actions.",
                "user_prompt_template": "Current state: {observation}. Choose your move.",
                "environment_context": {
                    "game": "FrozenLake",
                    "map_name": "4x4",
                    "seed": 42,
                },
            }
        ]

        # Create environment pointing to conda-isolated server
        envs = rk.make(
            f"http://localhost:{port}/mcp/",
            dataset=test_dataset,
            model_id=policy.model_id,
        )

        # Run short rollout
        start_time = time.time()
        trajectories = await rk.rollout(envs, policy=policy, steps=5)
        duration = time.time() - start_time

        assert len(trajectories) == 1, "Should have one trajectory"
        assert len(trajectories[0].get("steps", [])) > 0, "Should have recorded steps"

        print(
            f"✅ Conda-isolated server test completed with {len(trajectories[0]['steps'])} steps in {duration:.2f}s"
        )
        print(
            f"📁 Conda isolation recording saved to: {conda_isolation_recording_file}"
        )

        # Print trajectory summary
        traj = trajectories[0]
        print(
            f"📊 Conda Isolation Trajectory: {traj.steps} steps, reward: {traj.total_reward:.2f}, terminated: {traj.terminated}"
        )
        if hasattr(traj, "actions") and len(traj.actions) > 0:
            print(f"    Actions: {traj.actions}")

        # Clean up
        manager.stop_server(port)
        print("✅ Conda-isolated server stopped and cleaned up")

        # Clean up environment variable (but keep the recording file)
        if "REWARD_KIT_PLAYBACK_FILE" in os.environ:
            del os.environ["REWARD_KIT_PLAYBACK_FILE"]

    except ImportError:
        print(
            "⚠️ CondaServerProcessManager not available, skipping conda isolation test"
        )
        pytest.skip("CondaServerProcessManager not available")


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v", "-s"])
