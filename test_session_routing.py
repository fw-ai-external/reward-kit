#!/usr/bin/env python3
"""
Test Session Routing in Multi-Environment Proxy

This script analyzes whether different seeds create different session IDs
and get routed to different backend servers.
"""

import asyncio
import json
import logging
import subprocess
import time
from pathlib import Path

import reward_kit as rk

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SessionRoutingTester:
    """Test session ID generation and backend server routing."""

    def __init__(self):
        self.base_dir = Path(__file__).parent / "examples" / "frozen_lake_mcp"
        self.proxy_process = None

    async def test_session_routing(self):
        """Test if different seeds create different sessions and backend servers."""
        print("🔍 === SESSION ROUTING TEST ===")
        print(
            "Goal: Verify different seeds create different session IDs and backend servers"
        )
        print()

        try:
            # Start proxy with debug logging
            await self._start_proxy_with_logging()

            # Test session creation for each seed
            await self._test_individual_sessions()

            # Test concurrent session creation
            await self._test_concurrent_sessions()

        finally:
            await self._cleanup()

    async def _start_proxy_with_logging(self):
        """Start proxy server with detailed logging."""
        print("📡 Starting proxy server with debug logging...")

        server_script = str(self.base_dir / "server.py")
        requirements_path = str(self.base_dir / "requirements.txt")

        # Set debug logging for the proxy
        env = {"PYTHONPATH": str(Path(__file__).parent)}

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

        # Capture stdout and stderr to see session creation logs
        self.proxy_process = subprocess.Popen(
            cmd,
            cwd=self.base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            env=env,
            bufsize=1,  # Line buffered
            universal_newlines=True,
        )

        # Wait for startup
        await asyncio.sleep(8)

        if self.proxy_process.poll() is not None:
            output = self.proxy_process.stdout.read()
            raise RuntimeError(f"Proxy failed to start:\n{output}")

        print("✅ Proxy server started")

    async def _test_individual_sessions(self):
        """Test creating individual sessions for each seed."""
        print("\n🧪 Testing individual session creation...")

        test_seeds = [42, 123, 456]
        session_results = {}

        for seed in test_seeds:
            print(f"\n  Testing seed {seed}...")

            # Create single environment dataset
            dataset = [
                {
                    "id": f"session_test_{seed}",
                    "system_prompt": "Test prompt for session routing",
                    "user_prompt_template": "Test: {observation}",
                    "environment_context": {"seed": seed, "size": 4, "p": 0.8},
                }
            ]

            # Create environment through proxy
            envs = rk.make(
                "http://localhost:8090/mcp",
                dataset=dataset,
                model_id="session-routing-test",
            )

            # Reset to trigger session creation
            print(f"    Resetting environment for seed {seed}...")
            observations, _, _ = await envs.reset()
            obs = observations[0]

            session_results[seed] = {
                "observation": obs,
                "grid": obs.get("grid", "N/A") if isinstance(obs, dict) else str(obs),
            }

            print(f"    Grid for seed {seed}: {session_results[seed]['grid'][:20]}...")

            await envs.close()
            await asyncio.sleep(1)  # Brief pause between tests

        # Analyze results
        print("\n📊 Individual Session Analysis:")
        grids = [result["grid"] for result in session_results.values()]
        unique_grids = len(set(grids))

        print(f"  Seeds tested: {list(test_seeds)}")
        print(f"  Unique grids: {unique_grids}/{len(test_seeds)}")

        if unique_grids == len(test_seeds):
            print("  ✅ SUCCESS: Each seed created a unique environment!")
        else:
            print("  ❌ ISSUE: Some seeds created identical environments!")
            for seed, result in session_results.items():
                print(f"    Seed {seed}: {result['grid']}")

    async def _test_concurrent_sessions(self):
        """Test creating concurrent sessions with different seeds."""
        print("\n🚀 Testing concurrent session creation...")

        # Create dataset with multiple seeds
        multi_seed_dataset = [
            {
                "id": "concurrent_42",
                "system_prompt": "Test prompt",
                "user_prompt_template": "Test: {observation}",
                "environment_context": {"seed": 42, "size": 4, "p": 0.8},
            },
            {
                "id": "concurrent_123",
                "system_prompt": "Test prompt",
                "user_prompt_template": "Test: {observation}",
                "environment_context": {"seed": 123, "size": 4, "p": 0.8},
            },
            {
                "id": "concurrent_456",
                "system_prompt": "Test prompt",
                "user_prompt_template": "Test: {observation}",
                "environment_context": {"seed": 456, "size": 4, "p": 0.8},
            },
        ]

        # Create environments concurrently through proxy
        print("  Creating 3 concurrent environments...")
        envs = rk.make(
            "http://localhost:8090/mcp",
            dataset=multi_seed_dataset,
            model_id="concurrent-routing-test",
        )

        # Reset all environments
        print("  Resetting all environments...")
        observations, _, _ = await envs.reset()

        # Analyze concurrent results
        print("\n📊 Concurrent Session Analysis:")
        grids = []
        for i, obs in enumerate(observations):
            grid = obs.get("grid", "N/A") if isinstance(obs, dict) else str(obs)
            grids.append(grid)
            seed = multi_seed_dataset[i]["environment_context"]["seed"]
            print(f"  Environment {i} (seed {seed}): {grid[:20]}...")

        unique_concurrent_grids = len(set(grids))
        print(
            f"  Unique concurrent grids: {unique_concurrent_grids}/{len(multi_seed_dataset)}"
        )

        if unique_concurrent_grids == len(multi_seed_dataset):
            print("  ✅ SUCCESS: Concurrent environments are properly isolated!")
        else:
            print("  ❌ ISSUE: Concurrent environments share state!")
            for i, grid in enumerate(grids):
                seed = multi_seed_dataset[i]["environment_context"]["seed"]
                print(f"    Env {i} (seed {seed}): {grid}")

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

            # Show proxy logs for debugging
            if self.proxy_process.stdout:
                try:
                    remaining_output = self.proxy_process.stdout.read()
                    if remaining_output.strip():
                        print("\n📋 Proxy server logs (last 20 lines):")
                        lines = remaining_output.strip().split("\n")
                        for line in lines[-20:]:
                            if "session" in line.lower() or "seed" in line.lower():
                                print(f"    {line}")
                except Exception as e:
                    print(f"Error reading proxy logs: {e}")
                    pass

            print("✅ Proxy server stopped")


async def main():
    """Run the session routing test."""
    tester = SessionRoutingTester()

    try:
        await tester.test_session_routing()
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
