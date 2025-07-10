#!/usr/bin/env python3
"""
Test Environment Functionality with Tool Calls

This script tests that the multi-environment proxy works correctly with:
1. Different seeds generating different environments
2. Tool calls returning proper game state
3. Each environment maintaining its own state
"""

import asyncio
import json
import logging
import subprocess
import time
from pathlib import Path

import reward_kit as rk
from reward_kit.mcp import MCPToolCall

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Reduce noise from other modules
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class EnvironmentFunctionalityTest:
    """Test that environments work correctly with tool calls."""

    def __init__(self):
        self.base_dir = Path(__file__).parent / "examples" / "frozen_lake_mcp"
        self.proxy_process = None

    async def run_functionality_test(self):
        """Run complete environment functionality test."""
        print("🧪 === ENVIRONMENT FUNCTIONALITY TEST ===")
        print("Goal: Verify environments work correctly with tool calls")
        print()

        try:
            # Step 1: Start multi-environment proxy
            await self._start_proxy_server()

            # Step 2: Test tool calls with different seeds
            await self._test_tool_calls()

            print("\n✅ Environment functionality test completed!")

        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise
        finally:
            await self._cleanup()

    async def _start_proxy_server(self):
        """Start the multi-environment proxy server."""
        print("📡 Starting multi-environment proxy server...")

        server_script = str(self.base_dir / "server.py")
        requirements_path = str(self.base_dir / "requirements.txt")

        cmd = [
            "python",
            "-m",
            "reward_kit.mcp.multi_environment_proxy",
            "--server-script",
            server_script,
            "--requirements",
            requirements_path,
            "--port",
            "8091",
            "--max-envs",
            "3",
        ]

        self.proxy_process = subprocess.Popen(
            cmd,
            cwd=self.base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for startup
        await asyncio.sleep(8)

        if self.proxy_process.poll() is not None:
            stdout, stderr = self.proxy_process.communicate()
            raise RuntimeError(
                f"Proxy failed to start:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
            )

        print("✅ Proxy server started successfully")

    async def _test_tool_calls(self):
        """Test tool calls work correctly with different seeds."""
        print("\n🎮 Testing tool calls with different seeds...")

        # Create test dataset with different seeds
        test_seeds = [42, 123, 456]
        dataset = []

        for i, seed in enumerate(test_seeds):
            dataset.append(
                {
                    "id": f"tool_test_{seed}",
                    "system_prompt": "You are playing FrozenLake, a 4x4 grid game. Use lake_move tool with LEFT, DOWN, RIGHT, UP actions.",
                    "user_prompt_template": "Current state: {observation}. Choose your move: RIGHT",
                    "environment_context": {"seed": seed, "size": 4, "p": 0.8},
                }
            )

        print(f"📊 Created dataset with seeds: {test_seeds}")

        # Create environments through proxy
        print("🔗 Connecting to proxy server...")
        envs = rk.make(
            "http://localhost:8091/mcp", dataset=dataset, model_id="functionality-test"
        )

        print(f"✅ Created {envs.n} environments through proxy")

        # Test initial environment states
        print("🎯 Testing initial environment states...")
        observations, tool_schemas, _ = await envs.reset()

        print("📋 Initial observations and tool calls:")
        grids_captured = {}

        for i, obs in enumerate(observations):
            seed = test_seeds[i]
            print(f"\n  Environment {i} (seed {seed}):")
            print(f"    Initial observation: {obs}")

            # Make a tool call to get actual game state
            if tool_schemas[i] and len(tool_schemas[i]) > 0:
                tool_name = tool_schemas[i][0]["name"]  # Use first available tool
                print(
                    f"    Available tools: {[tool['name'] for tool in tool_schemas[i]]}"
                )

                # Test tool call
                try:
                    print(f"    Making tool call: {tool_name}(action='RIGHT')")
                    tool_calls = [
                        (
                            MCPToolCall(
                                tool_name=tool_name, arguments={"action": "RIGHT"}
                            )
                            if j == i
                            else MCPToolCall(tool_name="none", arguments={})
                        )
                        for j in range(envs.n)
                    ]

                    # Execute tool calls
                    new_observations, rewards, dones, infos = await envs.step(
                        tool_calls
                    )

                    # Log result for this environment
                    result_obs = new_observations[i]
                    reward = rewards[i]
                    done = dones[i]

                    print(f"    Tool call result:")
                    print(f"      New observation: {result_obs}")
                    print(f"      Reward: {reward}")
                    print(f"      Done: {done}")

                    # Extract grid if available
                    if isinstance(result_obs, dict) and "grid" in result_obs:
                        grid = result_obs["grid"]
                        grids_captured[seed] = grid
                        grid_lines = grid.split("\n") if grid else []
                        print(f"      Grid: {grid_lines[0] if grid_lines else 'Empty'}")

                except Exception as e:
                    print(f"    ❌ Tool call failed: {e}")
            else:
                print(f"    ❌ No tools available")

        # Analyze results
        print(f"\n📊 Tool Call Results:")
        print(f"  Grids captured: {len(grids_captured)}")
        print(f"  Expected grids: {len(test_seeds)}")

        if len(grids_captured) == len(test_seeds):
            unique_grids = len(set(grids_captured.values()))
            print(f"  Unique grids: {unique_grids}")

            if unique_grids == len(test_seeds):
                print("  ✅ SUCCESS: All environments have different grids!")

                # Show grid details
                for seed, grid in grids_captured.items():
                    lines = grid.split("\n") if grid else ["Empty"]
                    print(f"    Seed {seed}: {lines[0]}")
            else:
                print("  ❌ ISSUE: Some environments have identical grids!")
        else:
            print("  ❌ ISSUE: Not all environments returned grids!")

        await envs.close()

    async def _cleanup(self):
        """Clean up test resources."""
        print("\n🧹 Cleaning up...")

        if self.proxy_process:
            self.proxy_process.terminate()
            try:
                self.proxy_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proxy_process.kill()
                self.proxy_process.wait()
            print("✅ Proxy server stopped")


async def main():
    """Run the environment functionality test."""
    test = EnvironmentFunctionalityTest()

    try:
        await test.run_functionality_test()
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
