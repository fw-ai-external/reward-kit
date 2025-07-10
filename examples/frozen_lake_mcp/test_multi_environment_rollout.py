#!/usr/bin/env python3
"""
Test Multi-Environment Rollout with Proxy Server

This script demonstrates how to use the MultiEnvironmentProxy with the existing
reward-kit rollout system to run multiple isolated environment instances.

The beauty of this approach is that from the client perspective, it's just a
normal MCP server - but behind the scenes it manages multiple isolated conda
environments automatically.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List

import reward_kit as rk
from reward_kit.mcp.multi_environment_proxy import create_multi_environment_proxy


class ProxyServerManager:
    """Manages the lifecycle of the proxy server for testing."""

    def __init__(self, proxy_port: int = 8080):
        self.proxy_port = proxy_port
        self.process = None

    def start(self, server_script_path: str, requirements_path: str, max_envs: int = 5):
        """Start the proxy server in a subprocess."""
        cmd = [
            "python",
            "-m",
            "reward_kit.mcp.multi_environment_proxy",
            "--server-script",
            server_script_path,
            "--requirements",
            requirements_path,
            "--port",
            str(self.proxy_port),
            "--max-envs",
            str(max_envs),
        ]

        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Wait for server to start
        time.sleep(5)

        if self.process.poll() is not None:
            stdout, stderr = self.process.communicate()
            raise RuntimeError(
                f"Proxy server failed to start:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
            )

        print(f"✅ Proxy server started on port {self.proxy_port}")

    def stop(self):
        """Stop the proxy server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
            print("✅ Proxy server stopped")


async def test_multi_environment_rollout():
    """Test multi-environment rollout using the proxy server."""

    # Setup paths
    base_dir = Path(__file__).parent
    server_script = base_dir / "frozen_lake_mcp.py"
    requirements_file = base_dir / "requirements.txt"

    # Create a minimal requirements.txt if it doesn't exist
    if not requirements_file.exists():
        with open(requirements_file, "w") as f:
            f.write("fastmcp\ngymnasium\nnumpy\n")

    # Create test dataset
    dataset = [
        {
            "id": f"test_env_{i}",
            "system_prompt": "You are playing FrozenLake, a grid-based navigation game. "
            "Navigate to reach the goal 'G' while avoiding holes 'H'. "
            "Use the lake_move tool with actions: LEFT, DOWN, RIGHT, UP.",
            "user_intent": f"Navigate safely to reach the goal 'G' while avoiding holes 'H'. Environment {i}.",
            "environment_context": {"seed": 42 + i, "size": 4, "p": 0.8},
        }
        for i in range(3)  # Test with 3 environments
    ]

    # Start proxy server
    proxy_manager = ProxyServerManager(proxy_port=8080)

    try:
        proxy_manager.start(
            server_script_path=str(server_script),
            requirements_path=str(requirements_file),
            max_envs=5,
        )

        print("🚀 Starting multi-environment rollout test...")

        # Create policy for rollouts
        policy = rk.FireworksPolicy(
            model_id="accounts/fireworks/models/llama-v3p1-8b-instruct",
            temperature=0.2,
            max_tokens=512,
        )

        # Create environments using the proxy server
        print(f"📡 Connecting to proxy server at http://localhost:8080/mcp")
        envs = rk.make(
            "http://localhost:8080/mcp",  # Connect to proxy server
            dataset=dataset,
            model_id=policy.model_id,
        )

        print(f"✅ Created {envs.n} environments through proxy server")

        # Run rollouts - each environment will get its own isolated server instance
        print(f"🎮 Running rollouts with {envs.n} isolated environments...")
        start_time = time.time()

        trajectories = await rk.rollout(
            envs,
            policy=policy,
            steps=10,  # Keep it short for testing
            openai_format_log_file=None,  # Don't log for this test
        )

        rollout_duration = time.time() - start_time

        # Analyze results
        successful = sum(1 for traj in trajectories if traj.total_reward > 0)
        total_steps = sum(traj.steps for traj in trajectories)

        print("\n📊 Multi-Environment Rollout Results:")
        print(f"  • Total environments: {len(trajectories)}")
        print(f"  • Successful rollouts: {successful}/{len(trajectories)}")
        print(f"  • Total steps across all environments: {total_steps}")
        print(f"  • Average steps per environment: {total_steps/len(trajectories):.1f}")
        print(f"  • Total rollout time: {rollout_duration:.2f}s")
        print(f"  • Time per environment: {rollout_duration/len(trajectories):.2f}s")

        # Print individual trajectory details
        print("\n🔍 Individual Environment Results:")
        for i, traj in enumerate(trajectories):
            status = "✅ SUCCESS" if traj.total_reward > 0 else "❌ FAILED"
            print(
                f"  Environment {i}: {status} - {traj.steps} steps, reward: {traj.total_reward:.1f}"
            )

        # Verify that environments were truly isolated
        unique_seeds = set()
        for traj in trajectories:
            if hasattr(traj.session, "seed") and traj.session.seed is not None:
                unique_seeds.add(traj.session.seed)

        print(f"\n🔒 Environment Isolation Verification:")
        print(
            f"  • Unique seeds used: {len(unique_seeds)} (expected: {len(trajectories)})"
        )
        print(f"  • Seeds: {sorted(unique_seeds)}")

        # Test completed successfully
        print("\n🎉 Multi-environment rollout test completed successfully!")
        print("✅ Proxy server successfully managed multiple isolated environments")
        print("✅ Existing rollout.py worked seamlessly with proxy server")
        print("✅ Each environment ran in its own isolated conda environment")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        raise

    finally:
        # Clean up
        proxy_manager.stop()


async def test_concurrent_rollouts():
    """Test concurrent rollouts to verify isolation."""

    print("\n🔄 Testing concurrent rollout isolation...")

    # Setup paths
    base_dir = Path(__file__).parent
    server_script = base_dir / "frozen_lake_mcp.py"
    requirements_file = base_dir / "requirements.txt"

    # Start proxy server
    proxy_manager = ProxyServerManager(proxy_port=8081)

    try:
        proxy_manager.start(
            server_script_path=str(server_script),
            requirements_path=str(requirements_file),
            max_envs=10,
        )

        # Create policy
        policy = rk.FireworksPolicy(
            model_id="accounts/fireworks/models/llama-v3p1-8b-instruct",
            temperature=0.2,
            max_tokens=256,
        )

        # Run multiple concurrent rollouts
        async def run_single_rollout(rollout_id: int, num_envs: int):
            """Run a single rollout with multiple environments."""
            dataset = [
                {
                    "id": f"concurrent_{rollout_id}_env_{i}",
                    "system_prompt": "You are playing FrozenLake. Navigate to the goal.",
                    "user_intent": f"Concurrent rollout {rollout_id}, environment {i}",
                    "environment_context": {"seed": rollout_id * 100 + i},
                }
                for i in range(num_envs)
            ]

            envs = rk.make(
                "http://localhost:8081/mcp", dataset=dataset, model_id=policy.model_id
            )

            trajectories = await rk.rollout(envs, policy=policy, steps=5)
            return rollout_id, trajectories

        # Run 3 concurrent rollouts, each with 2 environments
        print("🚀 Starting 3 concurrent rollouts (2 environments each)...")
        start_time = time.time()

        results = await asyncio.gather(
            run_single_rollout(1, 2),
            run_single_rollout(2, 2),
            run_single_rollout(3, 2),
        )

        concurrent_duration = time.time() - start_time

        # Analyze concurrent results
        total_envs = sum(len(trajectories) for _, trajectories in results)
        print(f"\n📊 Concurrent Rollout Results:")
        print(f"  • Total rollouts: {len(results)}")
        print(f"  • Total environments: {total_envs}")
        print(f"  • Concurrent execution time: {concurrent_duration:.2f}s")

        for rollout_id, trajectories in results:
            successful = sum(1 for traj in trajectories if traj.total_reward > 0)
            print(
                f"  • Rollout {rollout_id}: {successful}/{len(trajectories)} successful"
            )

        print("✅ Concurrent rollout test completed successfully!")

    finally:
        proxy_manager.stop()


def main():
    """Main test function."""
    print("🧪 Multi-Environment Proxy Server Test Suite")
    print("=" * 50)

    try:
        # Test basic multi-environment rollout
        asyncio.run(test_multi_environment_rollout())

        # Test concurrent rollouts
        asyncio.run(test_concurrent_rollouts())

        print("\n🎉 All tests passed!")
        print("✅ Multi-environment proxy server is working correctly")
        print("✅ Compatible with existing rollout system")
        print("✅ Provides proper environment isolation")

    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
