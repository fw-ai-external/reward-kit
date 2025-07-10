#!/usr/bin/env python3
"""
Simple Multi-Environment Test

This test runs multiple single-environment instances sequentially to verify
that the control plane fix works correctly across different seeds and environments.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import reward_kit as rk
from reward_kit.mcp.execution import SimpleDeterministicPolicy


class MCPServerManager:
    """Manages MCP server lifecycle for testing."""

    def __init__(self, server_script: str, port: int = 8000):
        self.server_script = server_script
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.base_dir = Path(__file__).parent / "examples" / "frozen_lake_mcp"

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


async def test_multi_environment_sequential():
    """Test multiple environments sequentially to verify control plane fix."""
    print("\n🌟 === SIMPLIFIED MULTI-ENVIRONMENT TEST ===")

    # Test data with different seeds
    test_environments = [
        {"seed": 42, "port": 9700, "expected_behavior": "Should terminate early"},
        {"seed": 123, "port": 9701, "expected_behavior": "Should terminate early"},
        {"seed": 456, "port": 9702, "expected_behavior": "Should terminate early"},
    ]

    results = []

    for i, env_config in enumerate(test_environments):
        print(
            f"\n📍 Testing Environment {i+1}/{len(test_environments)} (seed: {env_config['seed']})"
        )

        # Start server for this environment
        server = MCPServerManager("server.py", port=env_config["port"])

        try:
            server.start()
            print(f"✅ Server started on port {env_config['port']}")

            # Create test dataset for this seed
            test_dataset = [
                {
                    "id": f"multi_env_test_{i+1:03d}",
                    "system_prompt": "You are playing FrozenLake, a 4x4 grid game. Use lake_move tool with LEFT, DOWN, RIGHT, UP actions to reach the goal 'G' while avoiding holes 'H'.",
                    "user_intent": f"Navigate safely to reach the goal 'G' while avoiding holes 'H'. Environment {i+1}.",
                    "user_prompt_template": "Current state: {observation}. Navigate to reach the goal 'G' while avoiding holes 'H'. Choose your next move wisely.",
                    "environment_context": {
                        "seed": env_config["seed"],
                        "size": 4,
                        "p": 0.8,
                    },
                }
            ]

            # Create policy
            policy = SimpleDeterministicPolicy(
                action_sequence=["RIGHT", "DOWN", "RIGHT", "DOWN", "RIGHT", "DOWN"],
                model_id=f"multi-env-test-{i+1}",
            )

            # Create environment
            envs = rk.make(
                f"http://localhost:{env_config['port']}/mcp/",
                dataset=test_dataset,
                model_id=policy.model_id,
            )

            print(f"✅ Created environment {i+1} with seed {env_config['seed']}")

            # Run rollout
            start_time = time.time()
            trajectories = await rk.rollout(
                envs, policy=policy, steps=10, openai_format_log_file=None  # Max steps
            )
            duration = time.time() - start_time

            # Analyze results
            trajectory = trajectories[0]
            result = {
                "env_index": i,
                "seed": env_config["seed"],
                "steps": trajectory.steps,
                "total_reward": trajectory.total_reward,
                "terminated": trajectory.terminated,
                "duration": duration,
                "success": trajectory.steps < 10,  # Early termination is success
            }

            results.append(result)

            print(f"📊 Environment {i+1} Results:")
            print(f"  • Seed: {result['seed']}")
            print(f"  • Steps: {result['steps']}/10")
            print(f"  • Total reward: {result['total_reward']:.2f}")
            print(f"  • Terminated: {result['terminated']}")
            print(f"  • Early termination: {'✅' if result['success'] else '❌'}")
            print(f"  • Duration: {result['duration']:.2f}s")

        finally:
            server.stop()
            print(f"🛑 Server {i+1} stopped")

        # Small delay between environments
        time.sleep(1)

    # Summary
    print(f"\n📊 === MULTI-ENVIRONMENT TEST SUMMARY ===")
    successful_envs = sum(1 for r in results if r["success"])

    print(f"Total environments tested: {len(results)}")
    print(f"Early termination success: {successful_envs}/{len(results)}")
    print(f"Success rate: {successful_envs/len(results)*100:.1f}%")

    print(f"\nDetailed results:")
    for result in results:
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(
            f"  Environment {result['env_index']+1} (seed {result['seed']}): {result['steps']} steps - {status}"
        )

    # Verify all environments had early termination
    all_successful = all(r["success"] for r in results)

    if all_successful:
        print(f"\n🎉 ALL ENVIRONMENTS SUCCESSFUL!")
        print(f"✅ Control plane fix working across multiple environments")
        print(f"✅ Independent termination per environment verified")
    else:
        print(f"\n❌ SOME ENVIRONMENTS FAILED!")
        print(f"❌ Control plane fix needs further investigation")

    return results, all_successful


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_multi_environment_sequential())
