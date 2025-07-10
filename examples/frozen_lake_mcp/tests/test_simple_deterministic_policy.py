#!/usr/bin/env python3
"""
Simple Deterministic Policy Test for MCP-Gym Session Debugging

This test uses SimpleDeterministicPolicy to isolate the session management issue
from LLM complexity. It focuses on testing multi-step trajectories with a
predetermined action sequence (RIGHT, DOWN, RIGHT, DOWN).

The goal is to:
1. Test if we can get multi-step trajectories working
2. Identify the session management issue
3. Record and validate trajectory data
4. Compare with the existing test results

Usage:
    pytest test_simple_deterministic_policy.py::test_deterministic_policy_multi_step -v -s
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import pytest

import reward_kit as rk
from reward_kit.mcp.execution import SimpleDeterministicPolicy


def _validate_trajectory_format(recording_file: str):
    """
    Validate trajectory format to ensure proper conversation flow.
    """
    print("  Validating trajectory format...")

    with open(recording_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                step_data = json.loads(line)
                env_idx = step_data.get("env_index", "?")
                step_num = step_data.get("step", "?")
                messages = step_data.get("messages", [])

                # Validate that there are messages
                if not messages:
                    raise AssertionError(
                        f"Line {line_num}: No messages found for env {env_idx}, step {step_num}"
                    )

                # Validate first message is system
                first_msg = messages[0]
                if first_msg.get("role") != "system":
                    raise AssertionError(
                        f"Line {line_num}: First message should be 'system', got '{first_msg.get('role')}' for env {env_idx}, step {step_num}"
                    )

            except json.JSONDecodeError as e:
                raise AssertionError(f"Line {line_num}: Invalid JSON: {e}")

    print("  ✅ Trajectory format validation passed")


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


class MultiEnvironmentProxyManager:
    """Manages MultiEnvironmentProxy server lifecycle for testing."""

    def __init__(
        self,
        server_script: str,
        requirements_path: str,
        proxy_port: int = 8090,
        max_envs: int = 3,
    ):
        self.server_script = server_script
        self.requirements_path = requirements_path
        self.proxy_port = proxy_port
        self.max_envs = max_envs
        self.process: Optional[subprocess.Popen] = None
        self.base_dir = Path(__file__).parent.parent

    def start(self) -> None:
        """Start the multi-environment proxy server."""
        if self.process:
            return

        # Start proxy server process using the module
        cmd = [
            "python",
            "-m",
            "reward_kit.mcp.multi_environment_proxy",
            "--server-script",
            self.server_script,
            "--requirements",
            self.requirements_path,
            "--port",
            str(self.proxy_port),
            "--max-envs",
            str(self.max_envs),
        ]

        self.process = subprocess.Popen(
            cmd,
            cwd=self.base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for proxy server to start
        time.sleep(10)

        # Check if process is still running
        if self.process.poll() is not None:
            stdout, stderr = self.process.communicate()
            raise RuntimeError(
                f"Proxy server failed to start:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
            )

    def stop(self) -> None:
        """Stop the proxy server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None

    def is_running(self) -> bool:
        """Check if proxy server is running."""
        return self.process is not None and self.process.poll() is None


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide test data directory."""
    return Path(__file__).parent.parent / "shared_data"


@pytest.fixture(scope="session")
def simple_server():
    """Start and manage a simple MCP server for testing."""
    server = MCPServerManager("server.py", port=9600)

    try:
        server.start()
        yield server
    finally:
        server.stop()


@pytest.fixture(scope="session")
def multi_environment_proxy():
    """Start and manage multi-environment proxy server."""
    base_dir = Path(__file__).parent.parent
    server_script = str(base_dir / "server.py")
    requirements_path = str(base_dir / "requirements.txt")

    # Set environment variables for faster testing
    # SKIP_EAGER_TOOL_DISCOVERY = "true"  # Keep tool discovery but use simple process manager
    os.environ["FORCE_SIMPLE_PROCESS_MANAGER"] = "true"

    proxy = MultiEnvironmentProxyManager(
        server_script=server_script,
        requirements_path=requirements_path,
        proxy_port=8095,
        max_envs=3,
    )

    try:
        proxy.start()
        yield proxy
    finally:
        proxy.stop()
        # Clean up environment variables
        # if "SKIP_EAGER_TOOL_DISCOVERY" in os.environ:
        #     del os.environ["SKIP_EAGER_TOOL_DISCOVERY"]
        if "FORCE_SIMPLE_PROCESS_MANAGER" in os.environ:
            del os.environ["FORCE_SIMPLE_PROCESS_MANAGER"]


@pytest.fixture
def deterministic_recording_file():
    """Provide a recording file path for deterministic policy test."""
    recording_dir = Path(__file__).parent / "recordings"
    recording_dir.mkdir(exist_ok=True)
    recording_path = recording_dir / "deterministic_policy_trajectory.jsonl"

    # Clean up existing recording for fresh test
    if os.path.exists(recording_path):
        os.unlink(recording_path)

    yield str(recording_path)


@pytest.mark.asyncio
async def test_deterministic_policy_single_environment(
    simple_server, deterministic_recording_file
):
    """Test single environment with deterministic policy to isolate issues."""
    print("\n🎯 === DETERMINISTIC POLICY SINGLE ENVIRONMENT TEST ===")

    # Clean up recording file for fresh test (unless in CI)
    is_ci = os.environ.get("CI", "false").lower() == "true"
    if not is_ci and os.path.exists(deterministic_recording_file):
        os.unlink(deterministic_recording_file)
        print(f"🧹 Cleaned up existing recording file: {deterministic_recording_file}")

    # Create simple test dataset
    test_dataset = [
        {
            "id": "deterministic_test_001",
            "system_prompt": "You are playing FrozenLake, a 4x4 grid game. Use lake_move tool with LEFT, DOWN, RIGHT, UP actions to reach the goal 'G' while avoiding holes 'H'.",
            "user_intent": "Navigate safely to reach the goal 'G' while avoiding holes 'H'.",
            "user_prompt_template": "Current state: {observation}. Navigate to reach the goal 'G' while avoiding holes 'H'. Choose your next move wisely.",
            "environment_context": {"seed": 999, "size": 4, "p": 0.8},
        }
    ]

    # Set up recording
    os.environ["REWARD_KIT_PLAYBACK_FILE"] = deterministic_recording_file

    # Create deterministic policy
    policy = SimpleDeterministicPolicy(
        action_sequence=["RIGHT", "DOWN", "RIGHT", "DOWN"],
        model_id="simple-deterministic-policy",
    )

    assert not policy.is_playback_mode(), "Should be in recording mode initially"

    # Create environments
    envs = rk.make(
        f"http://localhost:{simple_server.port}/mcp/",
        dataset=test_dataset,
        model_id=policy.model_id,
    )

    assert envs.n == 1, f"Expected 1 environment, got {envs.n}"
    print(f"✅ Created {envs.n} environments")

    # Run rollout with more steps to test multi-step behavior
    print(f"🎮 Running rollout with deterministic policy...")
    start_time = time.time()

    trajectories = await rk.rollout(
        envs,
        policy=policy,
        steps=10,  # More steps to test multi-step behavior
        openai_format_log_file=None,
    )

    duration = time.time() - start_time

    # Validate results
    assert len(trajectories) == 1, "Should have 1 trajectory"
    assert os.path.exists(
        deterministic_recording_file
    ), "Recording file should be created"

    trajectory = trajectories[0]
    print(f"✅ Recorded trajectory in {duration:.2f}s")
    print(f"📊 Trajectory Summary:")
    print(f"  • Steps: {trajectory.steps}")
    print(f"  • Total reward: {trajectory.total_reward:.2f}")
    print(f"  • Terminated: {trajectory.terminated}")
    print(f"  • Actions: {trajectory.actions}")

    # Validate trajectory format
    _validate_trajectory_format(deterministic_recording_file)

    # Print some recorded steps for analysis
    print("🔍 Sample recorded steps:")
    with open(deterministic_recording_file, "r") as f:
        for i, line in enumerate(f):
            if i >= 3:  # Only show first 3 steps
                break
            step_data = json.loads(line)
            env_idx = step_data.get("env_index", "?")
            step_num = step_data.get("step", "?")
            messages = step_data.get("messages", [])
            print(f"    Step {step_num} (env {env_idx}): {len(messages)} messages")

    # Key validation: Check if we got multi-step execution
    print(f"\n🔍 Multi-step validation:")
    print(f"  • Expected: Multiple steps (> 1)")
    print(f"  • Actual: {trajectory.steps} steps")

    if trajectory.steps > 1:
        print(f"  • ✅ SUCCESS: Multi-step execution working!")
    else:
        print(f"  • ❌ FAILURE: Only got {trajectory.steps} step(s)")

    # Clean up environment variable
    if "REWARD_KIT_PLAYBACK_FILE" in os.environ:
        del os.environ["REWARD_KIT_PLAYBACK_FILE"]

    # Return results for further analysis
    return {
        "trajectory": trajectory,
        "recording_file": deterministic_recording_file,
        "multi_step_success": trajectory.steps > 1,
        "duration": duration,
    }


@pytest.mark.asyncio
async def test_deterministic_policy_multi_environment(
    multi_environment_proxy, deterministic_recording_file
):
    """Test multi-environment with deterministic policy to isolate session management issues."""
    print("\n🌟 === DETERMINISTIC POLICY MULTI-ENVIRONMENT TEST ===")

    # Clean up recording file for fresh test (unless in CI)
    multi_env_recording_file = deterministic_recording_file.replace(
        ".jsonl", "_multi_env.jsonl"
    )
    is_ci = os.environ.get("CI", "false").lower() == "true"
    if not is_ci and os.path.exists(multi_env_recording_file):
        os.unlink(multi_env_recording_file)
        print(f"🧹 Cleaned up existing recording file: {multi_env_recording_file}")

    # Create multi-environment test dataset
    multi_env_dataset = [
        {
            "id": "multi_deterministic_001",
            "system_prompt": "You are playing FrozenLake, a 4x4 grid game. Use lake_move tool with LEFT, DOWN, RIGHT, UP actions to reach the goal 'G' while avoiding holes 'H'.",
            "user_intent": "Navigate safely to reach the goal 'G' while avoiding holes 'H'. Environment 1.",
            "user_prompt_template": "Current state: {observation}. Navigate to reach the goal 'G' while avoiding holes 'H'. Choose your next move wisely.",
            "environment_context": {"seed": 42, "size": 4, "p": 0.8},
        },
        {
            "id": "multi_deterministic_002",
            "system_prompt": "You are playing FrozenLake, a 4x4 grid game. Use lake_move tool with LEFT, DOWN, RIGHT, UP actions to reach the goal 'G' while avoiding holes 'H'.",
            "user_intent": "Navigate safely to reach the goal 'G' while avoiding holes 'H'. Environment 2.",
            "user_prompt_template": "Current state: {observation}. Navigate to reach the goal 'G' while avoiding holes 'H'. Choose your next move wisely.",
            "environment_context": {"seed": 123, "size": 4, "p": 0.8},
        },
        {
            "id": "multi_deterministic_003",
            "system_prompt": "You are playing FrozenLake, a 4x4 grid game. Use lake_move tool with LEFT, DOWN, RIGHT, UP actions to reach the goal 'G' while avoiding holes 'H'.",
            "user_intent": "Navigate safely to reach the goal 'G' while avoiding holes 'H'. Environment 3.",
            "user_prompt_template": "Current state: {observation}. Navigate to reach the goal 'G' while avoiding holes 'H'. Choose your next move wisely.",
            "environment_context": {"seed": 456, "size": 4, "p": 0.8},
        },
    ]

    # Set up recording
    os.environ["REWARD_KIT_PLAYBACK_FILE"] = multi_env_recording_file

    # Create deterministic policy with different action sequence for variety
    policy = SimpleDeterministicPolicy(
        action_sequence=["RIGHT", "DOWN", "RIGHT", "DOWN", "RIGHT", "DOWN"],
        model_id="multi-deterministic-policy",
    )

    assert not policy.is_playback_mode(), "Should be in recording mode initially"

    # Create environments pointing to the proxy server
    print(
        f"🔗 Connecting to proxy server at http://localhost:{multi_environment_proxy.proxy_port}/mcp"
    )
    envs = rk.make(
        f"http://localhost:{multi_environment_proxy.proxy_port}/mcp",
        dataset=multi_env_dataset,
        model_id=policy.model_id,
    )

    assert envs.n == len(
        multi_env_dataset
    ), f"Expected {len(multi_env_dataset)} environments, got {envs.n}"
    print(f"✅ Created {envs.n} environments through proxy server")

    # Run rollout
    print(f"🎮 Running rollouts with {envs.n} isolated server instances...")
    start_time = time.time()

    trajectories = await rk.rollout(
        envs,
        policy=policy,
        steps=8,  # Enough steps to test multi-step behavior
        openai_format_log_file=None,
    )

    duration = time.time() - start_time

    # Validate results
    assert len(trajectories) == len(
        multi_env_dataset
    ), "Should have trajectory for each environment"
    assert os.path.exists(multi_env_recording_file), "Recording file should be created"

    print(f"✅ Recorded {len(trajectories)} trajectories in {duration:.2f}s")
    print(f"📁 Multi-environment recording saved to: {multi_env_recording_file}")

    # Analyze trajectories
    print("📊 Multi-Environment Trajectory Analysis:")
    unique_seeds = set()
    total_steps = 0
    successful_envs = 0
    multi_step_envs = 0

    for i, traj in enumerate(trajectories):
        dataset_entry = multi_env_dataset[i]
        seed = dataset_entry.get("environment_context", {}).get("seed", "N/A")
        unique_seeds.add(seed)
        total_steps += traj.steps

        if traj.total_reward > 0:
            successful_envs += 1

        if traj.steps > 1:
            multi_step_envs += 1

        print(
            f"  Environment {i} (seed: {seed}): {traj.steps} steps, reward: {traj.total_reward:.2f}, terminated: {traj.terminated}"
        )
        if hasattr(traj, "actions") and len(traj.actions) > 0:
            actions_preview = (
                traj.actions[:3] if len(traj.actions) > 3 else traj.actions
            )
            print(
                f"    Actions: {actions_preview}{'...' if len(traj.actions) > 3 else ''}"
            )

    print(f"\n🔍 Environment Isolation and Multi-Step Verification:")
    print(f"  • Total environments: {len(trajectories)}")
    print(
        f"  • Unique seeds: {len(unique_seeds)} (should equal {len(multi_env_dataset)})"
    )
    print(f"  • Seeds used: {sorted(unique_seeds)}")
    print(f"  • Multi-step environments: {multi_step_envs}/{len(trajectories)}")
    print(f"  • Successful environments: {successful_envs}/{len(trajectories)}")
    print(f"  • Average steps per environment: {total_steps/len(trajectories):.1f}")
    print(f"  • Total execution time: {duration:.2f}s")

    # Validate environment isolation
    assert len(unique_seeds) == len(
        multi_env_dataset
    ), f"Expected {len(multi_env_dataset)} unique seeds, got {len(unique_seeds)}"

    # Validate trajectory format
    _validate_trajectory_format(multi_env_recording_file)

    # Key validation: Check if we got multi-step execution
    print(f"\n🔍 Multi-step validation:")
    print(f"  • Expected: Multiple steps (> 1) per environment")
    print(
        f"  • Actual: {multi_step_envs}/{len(trajectories)} environments with multi-step"
    )

    if multi_step_envs > 0:
        print(
            f"  • ✅ SUCCESS: Multi-step execution working for {multi_step_envs} environments!"
        )
    else:
        print(f"  • ❌ FAILURE: No environments got multi-step execution")

    # Clean up environment variable
    if "REWARD_KIT_PLAYBACK_FILE" in os.environ:
        del os.environ["REWARD_KIT_PLAYBACK_FILE"]

    print("🎉 Multi-environment deterministic policy test completed!")

    return {
        "trajectories": trajectories,
        "recording_file": multi_env_recording_file,
        "multi_step_success": multi_step_envs > 0,
        "multi_step_count": multi_step_envs,
        "duration": duration,
    }


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v", "-s"])
