#!/usr/bin/env python3
"""
Test Tool Calls for Seed Diversity

This script tests seed diversity by making tool calls directly instead of
relying on resource reading which has been having JSON import issues.
"""

import asyncio
import json
import subprocess
import time
from pathlib import Path

import reward_kit as rk


class ToolCallSeedTester:
    """Test seed diversity through tool calls."""

    def __init__(self):
        self.base_dir = Path(__file__).parent / "examples" / "frozen_lake_mcp"
        self.proxy_process = None

    async def test_tool_call_diversity(self):
        """Test seed diversity using tool calls."""
        print("🔧 === TOOL CALL SEED DIVERSITY TEST ===")
        print(
            "Goal: Verify different seeds create different FrozenLake grids via tool calls"
        )
        print()

        try:
            # Start proxy server
            await self._start_proxy()

            # Test tool calls for different seeds
            await self._test_seed_diversity_via_tools()

        finally:
            await self._cleanup()

    async def _start_proxy(self):
        """Start the proxy server."""
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
            "8090",
            "--max-envs",
            "5",
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

    async def _test_seed_diversity_via_tools(self):
        """Test seed diversity by making tool calls to get environment grids."""
        print("\n🎮 Testing seed diversity via tool calls...")

        test_seeds = [42, 123, 456]
        seed_results = {}

        for seed in test_seeds:
            print(f"\n  Testing seed {seed}...")

            # Create dataset for this seed
            dataset = [
                {
                    "id": f"tool_test_{seed}",
                    "system_prompt": "You are playing FrozenLake. Use lake_move tool.",
                    "user_prompt_template": "Current state: {observation}. Make a move.",
                    "environment_context": {"seed": seed, "size": 4, "p": 0.8},
                }
            ]

            # Create environment through proxy
            envs = rk.make(
                "http://localhost:8090/mcp",
                dataset=dataset,
                model_id="tool-diversity-test",
            )

            try:
                # Get initial state (fallback to default if resource fails)
                observations, _, _ = await envs.reset()
                obs = observations[0]

                print(f"    Initial observation: {obs}")

                # Make a tool call to get the actual environment grid
                from reward_kit.mcp.types import MCPToolCall

                # Create tool call
                tool_call = MCPToolCall(
                    "lake_move", {"action": "UP"}
                )  # Move that shouldn't change position 0

                # Execute tool call
                tool_responses = await envs.call_tools([tool_call])
                response = tool_responses[0]

                print(f"    Tool response: {response}")

                # Extract grid from response
                if isinstance(response, dict) and "grid" in response:
                    grid = response["grid"]
                    seed_results[seed] = {"grid": grid, "response": response}
                    print(f"    ✅ Grid extracted for seed {seed}")
                    print(f"    Grid preview: {grid.split()[0] if grid else 'N/A'}")
                else:
                    seed_results[seed] = {
                        "error": f"No grid in response: {response}",
                        "response": response,
                    }
                    print(f"    ❌ No grid found in response")

            except Exception as e:
                seed_results[seed] = {
                    "error": f"Tool call failed: {e}",
                    "response": None,
                }
                print(f"    ❌ Tool call failed: {e}")

            finally:
                await envs.close()
                await asyncio.sleep(1)  # Brief pause

        # Analyze diversity
        print("\n📊 Tool Call Diversity Analysis:")
        successful_results = {
            seed: result for seed, result in seed_results.items() if "grid" in result
        }

        if not successful_results:
            print("  ❌ No successful tool calls - all failed")
            for seed, result in seed_results.items():
                print(f"    Seed {seed}: {result.get('error', 'Unknown error')}")
            return False

        grids = [result["grid"] for result in successful_results.values()]
        unique_grids = len(set(grids))

        print(f"  Successful tool calls: {len(successful_results)}/{len(test_seeds)}")
        print(f"  Unique grids: {unique_grids}")
        print(f"  Expected unique grids: {len(successful_results)}")

        if unique_grids == len(successful_results):
            print("  ✅ SUCCESS: All seeds created unique environments!")
            for seed, result in successful_results.items():
                grid_preview = result["grid"][:20] if result["grid"] else "N/A"
                print(f"    Seed {seed}: {grid_preview}...")
            return True
        else:
            print("  ❌ ISSUE: Some seeds created identical environments!")
            for seed, result in successful_results.items():
                print(f"    Seed {seed}: {result['grid']}")
            return False

    async def _cleanup(self):
        """Clean up resources."""
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
    """Run the tool call diversity test."""
    tester = ToolCallSeedTester()

    try:
        await tester.test_tool_call_diversity()
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
