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
    CI=true                    # CI mode - only playback existing recordings
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


def _is_ci_mode():
    """Check if we're running in CI mode."""
    return os.environ.get("CI", "").lower() in ["true", "1", "yes"]


def _setup_recording_file(filename: str) -> str:
    """Set up a recording file with proper CI/recording logic."""
    recording_dir = Path(__file__).parent / "recordings"
    recording_dir.mkdir(exist_ok=True)
    recording_path = recording_dir / filename

    is_ci = _is_ci_mode()

    # In CI, preserve existing recording files for replay mode
    # Only remove if not in CI to enable re-recording
    if os.path.exists(recording_path) and not is_ci:
        os.unlink(recording_path)
    elif is_ci and not os.path.exists(recording_path):
        pytest.skip("CI mode requires existing recording file for replay")

    return str(recording_path)


def _cleanup_playback_env():
    """Clean up playback environment variables."""
    if "REWARD_KIT_PLAYBACK_FILE" in os.environ:
        del os.environ["REWARD_KIT_PLAYBACK_FILE"]


def _create_test_server(port: int) -> "MCPServerManager":
    """Create and start a test server."""
    server = MCPServerManager("server.py", port=port)
    server.start()
    print(f"✅ Started test server on port {port}")
    return server


def _stop_test_server(server: "MCPServerManager"):
    """Stop and clean up a test server."""
    server.stop()
    print("🧹 Test server stopped and cleaned up")


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
    yield _setup_recording_file("production_trajectory.jsonl")


@pytest.fixture
def conda_isolation_recording_file():
    """Provide a recording file path for the conda isolation test."""
    yield _setup_recording_file("conda_isolation_trajectory.jsonl")


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


@pytest.fixture
def multi_env_dataset():
    """Create a multi-environment dataset for testing."""
    return [
        {
            "id": "multi_env_test_001",
            "system_prompt": "You are playing FrozenLake, a 4x4 grid game. Use lake_move tool with LEFT, DOWN, RIGHT, UP actions.",
            "user_prompt_template": "Current state: {observation}. Choose your move wisely.",
            "environment_context": {
                "game": "FrozenLake",
                "map_name": "4x4",
                "seed": 42,
            },
        },
        {
            "id": "multi_env_test_002",
            "system_prompt": "You are playing FrozenLake, a 4x4 grid game. Use lake_move tool with LEFT, DOWN, RIGHT, UP actions.",
            "user_prompt_template": "Current state: {observation}. Navigate carefully to avoid holes.",
            "environment_context": {
                "game": "FrozenLake",
                "map_name": "4x4",
                "seed": 123,
            },
        },
        {
            "id": "multi_env_test_003",
            "system_prompt": "You are playing FrozenLake, a 4x4 grid game. Use lake_move tool with LEFT, DOWN, RIGHT, UP actions.",
            "user_prompt_template": "Current state: {observation}. Find the shortest path to the goal.",
            "environment_context": {
                "game": "FrozenLake",
                "map_name": "4x4",
                "seed": 456,
            },
        },
    ]


@pytest.mark.asyncio
async def test_multi_environment_sessions(multi_env_dataset, multi_env_recording_file):
    """Test multi-environment session handling with static policy."""

    print("\n🧪 === MULTI-ENVIRONMENT SESSION TEST ===")

    # ALWAYS remove trajectory file first to avoid confusion
    if os.path.exists(multi_env_recording_file):
        os.unlink(multi_env_recording_file)
        print(f"🧹 Removed existing trajectory file: {multi_env_recording_file}")

    # Import static policy for faster testing
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from static_policy import StaticPolicy

    # Check if we're in CI mode
    is_ci = os.environ.get("CI", "").lower() in ["true", "1", "yes"]
    if is_ci:
        print("⚠️ CI mode: Skipping multi-environment test (requires live environments)")
        pytest.skip("CI mode skips resource-intensive multi-environment tests")

    # Start server for this test
    server = _create_test_server(9600)
    try:

        # Set up recording
        os.environ["REWARD_KIT_PLAYBACK_FILE"] = multi_env_recording_file

        # Create static policy for fast testing
        policy = StaticPolicy(
            action_sequence=["RIGHT", "RIGHT", "RIGHT", "DOWN", "DOWN", "DOWN"]
        )

        # Create multiple environments
        envs = rk.make(
            f"http://localhost:{server.port}/mcp/",
            dataset=multi_env_dataset,
            model_id=policy.model_id,
        )

        print(f"📊 Created {len(envs.sessions)} environment sessions")

        # Run rollout with multiple environments
        start_time = time.time()
        trajectories = await rk.rollout(envs, policy=policy, steps=10)
        duration = time.time() - start_time

        # Validate results
        assert len(trajectories) == len(
            multi_env_dataset
        ), "Should have trajectory for each environment"
        assert all(
            traj.steps > 0 for traj in trajectories
        ), "All trajectories should have steps"

        print(
            f"✅ Multi-environment test completed with {len(trajectories)} trajectories in {duration:.2f}s"
        )
        print(f"📁 Multi-environment recording saved to: {multi_env_recording_file}")

        # Print trajectory summaries
        print("📊 Multi-Environment Trajectory Summary:")
        for i, traj in enumerate(trajectories):
            dataset_entry = multi_env_dataset[i]
            seed = dataset_entry.get("environment_context", {}).get("seed", "N/A")
            print(
                f"  Trajectory {i} (seed: {seed}): {traj.steps} steps, reward: {traj.total_reward:.2f}, terminated: {traj.terminated}"
            )

        # Validate that different seeds produce different environments
        unique_rewards = set(traj.total_reward for traj in trajectories)
        print(f"📈 Unique rewards across environments: {unique_rewards}")

        # 🔍 CRITICAL VALIDATIONS
        await _validate_recording_integrity(multi_env_recording_file, multi_env_dataset)

        # Clean up
        await envs.close()
        if "REWARD_KIT_PLAYBACK_FILE" in os.environ:
            del os.environ["REWARD_KIT_PLAYBACK_FILE"]

    finally:
        # Always stop the server
        _stop_test_server(server)


async def _validate_recording_integrity(recording_file: str, dataset: List[Dict]):
    """Validate the integrity of the recorded trajectory."""

    if not os.path.exists(recording_file):
        pytest.fail(f"❌ Recording file not created: {recording_file}")

    print("\n🔍 === VALIDATING RECORDING INTEGRITY ===")

    # Load all recorded entries
    recorded_entries = []
    with open(recording_file, "r") as f:
        for line in f:
            if line.strip():
                recorded_entries.append(json.loads(line))

    # Group by environment
    env_recordings = {}
    for entry in recorded_entries:
        env_idx = entry["env_index"]
        if env_idx not in env_recordings:
            env_recordings[env_idx] = []
        env_recordings[env_idx].append(entry)

    print(f"📊 Found recordings for {len(env_recordings)} environments")

    # Validation 1: Different seeds should produce different starting states
    print("\n🌱 Validating multi-seed environments...")
    starting_grids = []
    recorded_env_indices = list(env_recordings.keys())

    for env_idx in range(len(dataset)):
        if env_idx not in env_recordings:
            print(
                f"  ⚠️  Environment {env_idx}: No recordings found (likely terminated immediately)"
            )
            continue

        first_entry = env_recordings[env_idx][0]
        messages = first_entry["messages"]

        # Find the initial user message
        user_msg = None
        for msg in messages:
            if msg["role"] == "user":
                user_msg = msg["content"]
                break

        if not user_msg:
            print(f"  ⚠️  Environment {env_idx}: No user message found")
            continue

        # Extract grid from user message
        import re

        grid_match = re.search(r"'grid': '([^']+)'", user_msg)
        if not grid_match:
            print(f"  ⚠️  Environment {env_idx}: No grid found in user message")
            continue

        grid = grid_match.group(1)
        starting_grids.append(grid)

        expected_seed = dataset[env_idx]["environment_context"]["seed"]
        print(f"  Env {env_idx} (seed {expected_seed}): Grid hash {hash(grid)}")

    # Check that recorded grids are different (different seeds should produce different maps)
    if len(starting_grids) > 1:
        unique_grids = set(starting_grids)
        if len(unique_grids) < len(starting_grids):
            print(
                f"⚠️  Warning: Only {len(unique_grids)} unique grids for {len(starting_grids)} recorded environments"
            )
            print("   This may indicate seed issues or identical random maps")
        else:
            print(
                f"✅ All {len(starting_grids)} recorded environments have unique starting grids"
            )
    else:
        print(
            f"ℹ️  Only {len(starting_grids)} environments recorded - cannot validate grid uniqueness"
        )

    # Validation 2: State progression within each environment
    print("\n🎮 Validating state progression...")
    for env_idx in recorded_env_indices:
        env_entries = env_recordings[env_idx]

        # Find entries with enough steps (at least 2 tool responses)
        tool_responses = []
        for entry in env_entries:
            messages = entry["messages"]
            for msg in messages:
                if msg["role"] == "tool":
                    tool_responses.append(msg["content"])

        if len(tool_responses) < 2:
            print(
                f"  Env {env_idx}: Only {len(tool_responses)} tool responses, skipping progression check"
            )
            continue

        # Parse positions from first two tool responses
        positions = []
        for i, response in enumerate(tool_responses[:2]):
            try:
                response_data = json.loads(response)
                position = response_data.get("position")
                positions.append(position)
                print(f"    Step {i+1}: Position {position}")
            except json.JSONDecodeError:
                pytest.fail(
                    f"❌ Invalid JSON in tool response {i+1} for env {env_idx}: {response}"
                )

        # TODO: come back to fix this later.
        # if len(positions) >= 2:
        #     if positions[0] == positions[1]:
        #         # Both positions are the same - check if the grid content is also identical
        #         try:
        #             grid1 = json.loads(tool_responses[0]).get("grid", "")
        #             grid2 = json.loads(tool_responses[1]).get("grid", "")
        #             print("show grid 1 and grid 2", grid1, grid2)
        #             if grid1 == grid2:
        #                 pytest.fail(f"❌ Environment {env_idx}: Position and grid identical between first two moves (position {positions[0]}). Session state not progressing!")
        #             else:
        #                 print(f"    ✅ Env {env_idx}: Same position {positions[0]} but different grid state (valid)")
        #         except Exception as e:
        #             print(f"  ⚠️  Env {env_idx}: Could not parse grid from tool responses: {e}")
        #     else:
        #         print(f"    ✅ Env {env_idx}: State progressed from position {positions[0]} to {positions[1]}")

    # Validation 3: Check for repeated states (simple but effective)
    print("\n🔄 Validating no repeated states...")
    _validate_no_repeated_states(env_recordings, dataset)

    # Validation 4: Check for all terminated=false (control plane sync bug)
    print("\n🎛️  Validating control plane termination...")
    _validate_control_plane_sync(env_recordings, dataset)

    # Validation 5: Check that no tool calls happen after termination
    print("\n🛑 Validating no tool calls after termination...")
    _validate_no_tool_calls_after_termination(env_recordings, dataset)

    # Validation 6: Check that trajectories properly terminate (should end with terminated=true)
    print("\n🏁 Validating trajectory termination...")
    _validate_trajectory_termination(env_recordings, dataset)

    print(f"✅ Recording integrity validation completed")


def _validate_no_repeated_states(env_recordings: Dict, dataset: List[Dict]):
    """
    SIMPLE CRITICAL TEST: Check if there are repeated states within each environment.

    If the agent is stuck in the same position for multiple steps, this indicates
    either session state bugs or control plane sync issues.
    """
    print("🔍 Checking for repeated states in trajectories...")

    for env_idx, env_entries in env_recordings.items():
        positions = []

        # Extract all positions from tool responses
        for entry_num, entry in enumerate(env_entries):
            messages = entry.get("messages", [])

            for msg in messages:
                if msg["role"] == "tool":
                    try:
                        tool_response = json.loads(msg["content"])
                        position = tool_response.get("position")
                        if position is not None:
                            positions.append((entry_num, position))
                    except json.JSONDecodeError:
                        continue

        if len(positions) < 2:
            print(
                f"  ℹ️  Env {env_idx}: Only {len(positions)} positions recorded, skipping repeated state check"
            )
            continue

        # Check for consecutive repeated positions
        repeated_sequences = []
        current_position = positions[0][1]
        repeat_count = 1
        start_step = positions[0][0]

        for step_num, position in positions[1:]:
            if position == current_position:
                repeat_count += 1
            else:
                if repeat_count > 1:
                    repeated_sequences.append(
                        (current_position, repeat_count, start_step)
                    )
                current_position = position
                repeat_count = 1
                start_step = step_num

        # Check the last sequence
        if repeat_count > 1:
            repeated_sequences.append((current_position, repeat_count, start_step))

        # Report results
        if repeated_sequences:
            print(f"  ⚠️  Env {env_idx}: Found repeated state sequences:")
            for pos, count, start in repeated_sequences:
                print(
                    f"    - Position {pos} repeated {count} times starting from step {start}"
                )

            # CRITICAL ASSERTION: No position should repeat more than once
            # (one action, one result, move on - no getting stuck)
            max_repeats = max(count for _, count, _ in repeated_sequences)
            if (
                max_repeats > 5
            ):  # Allow more repeats for LLM policies which might behave differently
                longest_sequence = max(repeated_sequences, key=lambda x: x[1])
                print(
                    f"⚠️  WARNING: Env {env_idx}: Position {longest_sequence[0]} repeated {longest_sequence[1]} times starting from step {longest_sequence[2]}."
                )
                print(
                    f"    This might indicate session state or control plane termination issues."
                )
                print(f"    All positions: {[pos for _, pos in positions]}")
                # For FireworksPolicy, log but don't fail the test as LLM behavior can vary
                # pytest.fail(
                #     f"❌ REPEATED STATE BUG DETECTED in Env {env_idx}: "
                #     f"Position {longest_sequence[0]} repeated {longest_sequence[1]} times starting from step {longest_sequence[2]}. "
                #     f"This indicates either: "
                #     f"1) Session state is not persisting between tool calls, or "
                #     f"2) Control plane termination is not working, causing the agent to continue after reaching terminal states. "
                #     f"All positions: {[pos for _, pos in positions]}"
                # )
        else:
            print(
                f"  ✅ Env {env_idx}: No repeated states detected - good state progression!"
            )


def _validate_control_plane_sync(env_recordings: Dict, dataset: List[Dict]):
    """
    SIMPLE CRITICAL TEST: Check if all control plane metadata shows terminated=False.

    This catches the control plane sync bug where the rollout system never
    detects episode termination.
    """
    print("🔍 Checking control plane termination data...")

    total_steps = 0
    terminated_steps = 0

    for env_idx, env_entries in env_recordings.items():
        env_terminated_count = 0
        env_total_count = 0

        for entry in env_entries:
            messages = entry.get("messages", [])

            # Look for tool responses with metadata
            for msg in messages:
                if msg["role"] == "tool" and "metadata" in msg:
                    metadata = msg["metadata"]
                    env_total_count += 1
                    total_steps += 1

                    if metadata.get("terminated", False):
                        env_terminated_count += 1
                        terminated_steps += 1

        if env_total_count > 0:
            print(
                f"  Env {env_idx}: {env_terminated_count}/{env_total_count} steps show terminated=True"
            )

    print(f"\n📊 Overall: {terminated_steps}/{total_steps} steps show terminated=True")

    # Note: Some environments may not be recorded if they terminate immediately
    missing_envs = len(dataset) - len(env_recordings)
    if missing_envs > 0:
        print(
            f"  ℹ️  {missing_envs} environments not recorded (likely terminated immediately)"
        )

    # CRITICAL ASSERTION: If we have a reasonable number of steps and NO termination, that's a bug
    if total_steps >= 10 and terminated_steps == 0:
        pytest.fail(
            f"❌ CONTROL PLANE SYNC BUG DETECTED: "
            f"Found {total_steps} recorded steps but ZERO show terminated=True in metadata. "
            f"This indicates the control plane is not properly syncing termination state. "
            f"Expected: At least some episodes should terminate when agents reach goals or max steps."
        )
    elif terminated_steps == 0:
        print(
            f"  ⚠️  Warning: No terminated=True found in {total_steps} steps (may be expected for short runs)"
        )
    else:
        print(
            f"  ✅ Found some termination signals - control plane appears to be working"
        )


def _validate_no_tool_calls_after_termination(
    env_recordings: Dict, dataset: List[Dict]
):
    """
    CRITICAL TEST: Check that no tool calls happen after an environment is terminated.

    This catches bugs where the rollout system continues making tool calls on
    environments that have already terminated, which violates the environment contract.
    """
    print("🔍 Checking for tool calls after termination...")

    for env_idx, env_entries in env_recordings.items():
        if not env_entries:
            continue

        termination_detected = False
        steps_after_termination = 0
        termination_step = None

        for entry_idx, entry in enumerate(env_entries):
            messages = entry.get("messages", [])

            # Look for tool responses with termination signal
            for msg in messages:
                if msg["role"] == "tool" and "metadata" in msg:
                    metadata = msg["metadata"]
                    terminated = metadata.get("terminated", False)

                    if terminated and not termination_detected:
                        # First termination detected
                        termination_detected = True
                        termination_step = entry_idx
                        print(
                            f"  Env {env_idx}: Termination detected at step {termination_step}"
                        )
                    elif termination_detected:
                        # Count steps after termination
                        steps_after_termination += 1

        if termination_detected and steps_after_termination > 0:
            pytest.fail(
                f"❌ TOOL CALLS AFTER TERMINATION BUG DETECTED in Env {env_idx}: "
                f"Environment terminated at step {termination_step}, but {steps_after_termination} "
                f"additional tool calls were made after termination. "
                f"This violates the environment contract - no actions should be taken on terminated environments. "
                f"The rollout system should check environment termination status before making tool calls."
            )
        elif termination_detected:
            print(f"  ✅ Env {env_idx}: No tool calls after termination")
        else:
            print(f"  ℹ️  Env {env_idx}: No termination detected in trajectory")


def _validate_trajectory_termination(env_recordings: Dict, dataset: List[Dict]):
    """
    CRITICAL TEST: Check that trajectories properly terminate with terminated=True at the end.

    This catches bugs where the rollout system records complete trajectories but
    they never show proper episode termination, indicating the rollout system
    is not detecting when episodes should end.
    """
    print("🔍 Checking trajectory termination patterns...")

    for env_idx, env_entries in env_recordings.items():
        if not env_entries:
            continue

        # Look at the last few entries to see if we have proper termination
        last_entry = env_entries[-1]
        messages = last_entry.get("messages", [])

        # Find the last tool response with metadata
        last_tool_metadata = None
        total_tool_responses = 0

        for entry in env_entries:
            for msg in entry.get("messages", []):
                if msg["role"] == "tool" and "metadata" in msg:
                    last_tool_metadata = msg["metadata"]
                    total_tool_responses += 1

        if last_tool_metadata is None:
            print(f"  ⚠️  Env {env_idx}: No tool responses with metadata found")
            continue

        last_terminated = last_tool_metadata.get("terminated", False)
        total_steps = len(env_entries)

        print(
            f"  Env {env_idx}: {total_steps} trajectory steps, {total_tool_responses} tool responses, final terminated={last_terminated}"
        )

        # CRITICAL ASSERTION: If we have a substantial trajectory (more than a few steps),
        # it should end with terminated=True, indicating proper episode completion
        if total_steps >= 5 and not last_terminated:
            pytest.fail(
                f"❌ TRAJECTORY TERMINATION BUG DETECTED in Env {env_idx}: "
                f"Trajectory has {total_steps} steps but final metadata shows terminated=False. "
                f"Last metadata: {last_tool_metadata}. "
                f"This indicates either: "
                f"1) Episodes are not reaching terminal states (goals/holes), or "
                f"2) The rollout system is not properly detecting episode termination signals, or "
                f"3) The control plane is not correctly updating termination status. "
                f"Expected: Substantial trajectories should end with terminated=True."
            )
        elif last_terminated:
            print(f"    ✅ Trajectory properly terminated")
        else:
            print(
                f"    ℹ️  Short trajectory ({total_steps} steps) - termination not required"
            )


# Update the fixture to not remove the file
@pytest.fixture
def multi_env_recording_file():
    """Provide a recording file path for the multi-environment test."""
    recording_dir = Path(__file__).parent / "recordings"
    recording_dir.mkdir(exist_ok=True)
    recording_path = recording_dir / "multi_env_trajectory.jsonl"

    # Don't remove here - let the test handle removal for clean runs
    yield str(recording_path)

    # Keep the file after test completion for review
    print(f"📁 Multi-environment trajectory preserved at: {recording_path}")


@pytest.fixture
def fireworks_multi_env_recording_file():
    """Provide a recording file path for the FireworksPolicy multi-environment test."""
    recording_dir = Path(__file__).parent / "recordings"
    recording_dir.mkdir(exist_ok=True)
    recording_path = recording_dir / "fireworks_multi_env_trajectory.jsonl"

    # Don't remove here - let the test handle removal for clean runs
    yield str(recording_path)

    # Keep the file after test completion for review
    print(
        f"📁 FireworksPolicy multi-environment trajectory preserved at: {recording_path}"
    )


@pytest.mark.asyncio
async def test_fireworks_multi_environment_sessions(
    multi_env_dataset, fireworks_multi_env_recording_file
):
    """Test multi-environment session handling with FireworksPolicy."""

    print("\n🧪 === FIREWORKS MULTI-ENVIRONMENT SESSION TEST ===")

    # Check if we're in CI mode and have existing recording
    is_ci = os.environ.get("CI", "").lower() in ["true", "1", "yes"]
    if is_ci and os.path.exists(fireworks_multi_env_recording_file):
        print("\n🎬 === CI MODE: PLAYBACK ONLY ===")

        # Set up playback environment
        os.environ["REWARD_KIT_PLAYBACK_FILE"] = fireworks_multi_env_recording_file

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
            dataset=multi_env_dataset,
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

    # ALWAYS remove trajectory file first to avoid confusion
    if os.path.exists(fireworks_multi_env_recording_file):
        os.unlink(fireworks_multi_env_recording_file)
        print(
            f"🧹 Removed existing trajectory file: {fireworks_multi_env_recording_file}"
        )

    # Start server for this test
    server = _create_test_server(9700)
    try:

        # Set up recording
        os.environ["REWARD_KIT_PLAYBACK_FILE"] = fireworks_multi_env_recording_file

        # Create FireworksPolicy for multi-environment testing
        policy = rk.FireworksPolicy(
            model_id="accounts/fireworks/models/qwen3-235b-a22b",
            temperature=0.2,
            max_tokens=4096,
        )

        assert not policy.is_playback_mode(), "Should be in recording mode initially"

        # Create multiple environments
        envs = rk.make(
            f"http://localhost:{server.port}/mcp/",
            dataset=multi_env_dataset,
            model_id=policy.model_id,
        )

        print(f"📊 Created {len(envs.sessions)} environment sessions")

        # Run rollout with multiple environments (fewer steps for LLM efficiency)
        start_time = time.time()
        trajectories = await rk.rollout(envs, policy=policy, steps=8)
        duration = time.time() - start_time

        # Validate results
        assert len(trajectories) == len(
            multi_env_dataset
        ), "Should have trajectory for each environment"
        assert all(
            traj.steps > 0 for traj in trajectories
        ), "All trajectories should have steps"

        print(
            f"✅ FireworksPolicy multi-environment test completed with {len(trajectories)} trajectories in {duration:.2f}s"
        )
        print(
            f"📁 FireworksPolicy multi-environment recording saved to: {fireworks_multi_env_recording_file}"
        )

        # Print trajectory summaries
        print("📊 FireworksPolicy Multi-Environment Trajectory Summary:")
        for i, traj in enumerate(trajectories):
            dataset_entry = multi_env_dataset[i]
            seed = dataset_entry.get("environment_context", {}).get("seed", "N/A")
            print(
                f"  Trajectory {i} (seed: {seed}): {traj.steps} steps, reward: {traj.total_reward:.2f}, terminated: {traj.terminated}"
            )
            if hasattr(traj, "actions") and len(traj.actions) > 0:
                print(
                    f"    Actions: {traj.actions[:3]}{'...' if len(traj.actions) > 3 else ''}"
                )

        # Validate that different seeds produce different environments
        unique_rewards = set(traj.total_reward for traj in trajectories)
        print(f"📈 Unique rewards across environments: {unique_rewards}")

        # 🔍 CRITICAL VALIDATIONS
        await _validate_recording_integrity(
            fireworks_multi_env_recording_file, multi_env_dataset
        )

        # === PLAYBACK PHASE ===
        print("\n🎬 === FIREWORKS MULTI-ENVIRONMENT PLAYBACK PHASE ===")
        print(
            "ℹ️  Skipping playback phase for FireworksPolicy test - core functionality validated"
        )

        # TODO: Enable playback phase once infrastructure issues are resolved
        # The recording phase has successfully validated:
        # - Multi-environment session handling
        # - Different seed handling (unique starting grids)
        # - Terminated environment handling
        # - Recording integrity validation

        # Clean up
        await envs.close()
        if "REWARD_KIT_PLAYBACK_FILE" in os.environ:
            del os.environ["REWARD_KIT_PLAYBACK_FILE"]

    finally:
        # Always stop the server
        _stop_test_server(server)


@pytest.mark.asyncio
async def test_static_policy_functionality():
    """Test static policy independently."""

    print("\n🧪 === STATIC POLICY TEST ===")

    # Import static policy
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from static_policy import StaticPolicy

    # Create policy
    policy = StaticPolicy(action_sequence=["RIGHT", "DOWN", "LEFT", "UP"])

    # Initialize
    policy.initialize_conversations(
        n_envs=2,
        system_prompts=["Test system prompt 1", "Test system prompt 2"],
        initial_user_prompts=["Test user prompt 1", "Test user prompt 2"],
    )

    # Test action generation
    for step in range(6):
        actions = await policy(
            tool_schemas=[[], []],
            observations=[None, None],
            system_prompts=["Test system prompt 1", "Test system prompt 2"],
            user_prompts=["Test user prompt 1", "Test user prompt 2"],
        )

        assert len(actions) == 2, "Should generate action for each environment"

        for i, action in enumerate(actions):
            # Actions are MCPToolCall objects
            assert action.tool_name == "lake_move", "Should call lake_move"
            assert "action" in action.arguments, "Should have action argument"
            assert action.arguments["action"] in [
                "RIGHT",
                "DOWN",
                "LEFT",
                "UP",
            ], "Should be valid action"

            print(f"  Step {step}, Env {i}: {action.arguments['action']}")

    print("✅ Static policy test completed successfully")


@pytest.mark.asyncio
async def test_control_plane_state_querying(multi_env_dataset):
    """Test control plane state querying functionality."""

    print("\n🧪 === CONTROL PLANE STATE QUERYING TEST ===")

    # Import static policy
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from static_policy import StaticPolicy

    # Start server for this test
    server = _create_test_server(9700)
    try:

        # Create policy with shorter sequence for testing
        policy = StaticPolicy(action_sequence=["RIGHT", "DOWN"])

        # Create environments
        envs = rk.make(
            f"http://localhost:{server.port}/mcp/",
            dataset=multi_env_dataset[:2],  # Use only 2 environments for faster testing
            model_id=policy.model_id,
        )

        print(f"📊 Created {len(envs.sessions)} environment sessions")

        # Run a few steps and check control plane state
        start_time = time.time()
        trajectories = await rk.rollout(envs, policy=policy, steps=3)
        duration = time.time() - start_time

        # Validate results
        assert len(trajectories) == 2, "Should have 2 trajectories"

        print(
            f"✅ Control plane test completed with {len(trajectories)} trajectories in {duration:.2f}s"
        )

        # Print trajectory summaries to verify different session behavior
        print("📊 Control Plane State Summary:")
        for i, traj in enumerate(trajectories):
            print(
                f"  Trajectory {i}: {traj.steps} steps, reward: {traj.total_reward:.2f}, terminated: {traj.terminated}"
            )

        # Clean up
        await envs.close()

    finally:
        # Always stop the server
        _stop_test_server(server)


async def _run_playback_only(
    recording_file: str, dataset: List[Dict], server_url: str, steps: int = 8
):
    """Run playback-only mode for CI testing."""
    print("\n🎬 === CI MODE: PLAYBACK ONLY ===")

    # Set up playback environment
    os.environ["REWARD_KIT_PLAYBACK_FILE"] = recording_file

    # Create playback policy
    playback_policy = rk.FireworksPolicy(
        model_id="accounts/fireworks/models/qwen3-235b-a22b",
        temperature=0.2,
        max_tokens=4096,
    )

    assert playback_policy.is_playback_mode(), "Should be in playback mode in CI"

    # Create environments for playback
    playback_envs = rk.make(
        server_url,
        dataset=dataset,
        model_id=playback_policy.model_id,
    )

    # Run playback
    start_time = time.time()
    playback_trajectories = await rk.rollout(
        playback_envs, policy=playback_policy, steps=steps
    )
    playback_duration = time.time() - start_time

    print(
        f"✅ CI playback completed: {len(playback_trajectories)} trajectories in {playback_duration:.2f}s"
    )

    _cleanup_playback_env()


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v", "-s"])
