"""
Simulation Wrapper for FrozenLake MCP Server

This wrapper provides testing and simulation utilities for MCP servers.
It validates the north star requirements and enables batch testing.

Usage:
    # Test basic functionality
    python simulation_wrapper.py --test basic

    # Test with multiple seeds
    python simulation_wrapper.py --test seeds --count 10

    # Performance test
    python simulation_wrapper.py --test performance --count 100
"""

import argparse
import asyncio
import os
import signal
import statistics
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class TestResult:
    """Results from a single test run."""

    success: bool
    duration: float
    steps: int
    final_reward: float
    error: Optional[str] = None


@dataclass
class BatchTestResults:
    """Results from batch testing."""

    total_tests: int
    successful_tests: int
    failed_tests: int
    avg_duration: float
    avg_steps: float
    success_rate: float
    avg_reward: float
    errors: List[str]


class MCPServerWrapper:
    """Wrapper for testing MCP servers programmatically."""

    def __init__(self, server_url: str = "http://localhost:8000"):
        """
        Initialize MCP server wrapper.

        Args:
            server_url: Base URL for the MCP server
        """
        self.server_url = server_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.server_process = None

    async def start_server(
        self,
        server_script: str = "frozen_lake_north_star.py",
        transport: str = "streamable-http",
    ):
        """Start MCP server subprocess."""
        cmd = ["python", server_script, "--transport", transport]

        print(f"🚀 Starting server: {' '.join(cmd)}")
        self.server_process = subprocess.Popen(
            cmd,
            cwd="/home/bchen/home/reward-kit/examples/frozen_lake_mcp",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to start
        await asyncio.sleep(2)

        if self.server_process.poll() is not None:
            stdout, stderr = self.server_process.communicate()
            raise RuntimeError(f"Server failed to start. STDERR: {stderr.decode()}")

        print("✅ Server started successfully")

    async def stop_server(self):
        """Stop MCP server subprocess."""
        if self.server_process:
            print("🛑 Stopping server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            self.server_process = None
            print("✅ Server stopped")

    async def test_single_episode(
        self, seed: Optional[int] = None, max_steps: int = 100
    ) -> TestResult:
        """
        Test a single episode with the MCP server.

        Args:
            seed: Random seed for reproducibility
            max_steps: Maximum steps before truncation

        Returns:
            TestResult with episode outcome
        """
        start_time = time.time()
        steps = 0
        final_reward = 0.0

        try:
            # Initialize session
            init_response = await self.client.post(
                f"{self.server_url}/mcp/tools/get_initial_observation", json={}
            )
            init_response.raise_for_status()
            init_data = init_response.json()

            initial_obs = init_data.get("initialObservation")
            print(f"🎮 Episode started, initial observation: {initial_obs}")

            # Run episode
            terminated = False
            truncated = False

            while not terminated and not truncated and steps < max_steps:
                # Choose random action for testing
                actions = ["LEFT", "DOWN", "RIGHT", "UP"]
                action = actions[steps % len(actions)]  # Simple pattern for testing

                # Execute step
                step_response = await self.client.post(
                    f"{self.server_url}/mcp/tools/lake_move", json={"action": action}
                )
                step_response.raise_for_status()
                step_data = step_response.json()

                # Extract step results
                obs = step_data.get("observation")
                reward = step_data.get("reward", 0.0)
                terminated = step_data.get("terminated", False)
                truncated = step_data.get("truncated", False)

                final_reward += reward
                steps += 1

                print(f"  Step {steps}: {action} → obs={obs}, reward={reward}")

                if terminated or truncated:
                    break

            duration = time.time() - start_time

            return TestResult(
                success=True, duration=duration, steps=steps, final_reward=final_reward
            )

        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                success=False,
                duration=duration,
                steps=steps,
                final_reward=final_reward,
                error=str(e),
            )

    async def test_batch_episodes(
        self, count: int, seeds: Optional[List[int]] = None
    ) -> BatchTestResults:
        """
        Test multiple episodes for consistency and performance.

        Args:
            count: Number of episodes to test
            seeds: Optional list of seeds (will generate if not provided)

        Returns:
            BatchTestResults with aggregated statistics
        """
        if seeds is None:
            seeds = list(range(count))
        elif len(seeds) < count:
            seeds.extend(range(len(seeds), count))

        print(f"🔄 Running {count} test episodes...")

        results = []
        errors = []

        for i, seed in enumerate(seeds[:count]):
            print(f"\n--- Episode {i+1}/{count} (seed={seed}) ---")

            result = await self.test_single_episode(seed=seed)
            results.append(result)

            if not result.success:
                errors.append(f"Episode {i+1}: {result.error}")
                print(f"❌ Episode {i+1} failed: {result.error}")
            else:
                print(
                    f"✅ Episode {i+1} completed: {result.steps} steps, reward={result.final_reward}"
                )

        # Calculate statistics
        successful_results = [r for r in results if r.success]

        if successful_results:
            avg_duration = statistics.mean(r.duration for r in successful_results)
            avg_steps = statistics.mean(r.steps for r in successful_results)
            avg_reward = statistics.mean(r.final_reward for r in successful_results)
        else:
            avg_duration = avg_steps = avg_reward = 0.0

        return BatchTestResults(
            total_tests=count,
            successful_tests=len(successful_results),
            failed_tests=len(results) - len(successful_results),
            avg_duration=avg_duration,
            avg_steps=avg_steps,
            success_rate=len(successful_results) / count * 100,
            avg_reward=avg_reward,
            errors=errors,
        )

    async def close(self):
        """Clean up resources."""
        await self.client.aclose()
        await self.stop_server()


async def run_basic_test():
    """Run basic functionality test."""
    print("🧪 Running basic functionality test...")

    wrapper = MCPServerWrapper()

    try:
        await wrapper.start_server()
        result = await wrapper.test_single_episode(seed=42)

        if result.success:
            print(f"✅ Basic test PASSED")
            print(f"   Duration: {result.duration:.2f}s")
            print(f"   Steps: {result.steps}")
            print(f"   Final reward: {result.final_reward}")
        else:
            print(f"❌ Basic test FAILED: {result.error}")

    finally:
        await wrapper.close()


async def run_seed_test(count: int):
    """Run test with multiple seeds."""
    print(f"🧪 Running seed consistency test with {count} episodes...")

    wrapper = MCPServerWrapper()

    try:
        await wrapper.start_server()
        results = await wrapper.test_batch_episodes(count)

        print(f"\n📊 BATCH TEST RESULTS:")
        print(f"   Total episodes: {results.total_tests}")
        print(f"   Successful: {results.successful_tests}")
        print(f"   Failed: {results.failed_tests}")
        print(f"   Success rate: {results.success_rate:.1f}%")
        print(f"   Average duration: {results.avg_duration:.2f}s")
        print(f"   Average steps: {results.avg_steps:.1f}")
        print(f"   Average reward: {results.avg_reward:.2f}")

        if results.errors:
            print(f"\n❌ ERRORS:")
            for error in results.errors[:5]:  # Show first 5 errors
                print(f"   {error}")
            if len(results.errors) > 5:
                print(f"   ... and {len(results.errors) - 5} more errors")

    finally:
        await wrapper.close()


async def run_performance_test(count: int):
    """Run performance test."""
    print(f"⚡ Running performance test with {count} episodes...")

    wrapper = MCPServerWrapper()

    try:
        await wrapper.start_server()

        start_time = time.time()
        results = await wrapper.test_batch_episodes(count)
        total_time = time.time() - start_time

        print(f"\n⚡ PERFORMANCE RESULTS:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Episodes per second: {count / total_time:.2f}")
        print(f"   Average episode duration: {results.avg_duration:.3f}s")
        print(f"   Success rate: {results.success_rate:.1f}%")

    finally:
        await wrapper.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="FrozenLake MCP Simulation Wrapper")
    parser.add_argument(
        "--test",
        choices=["basic", "seeds", "performance"],
        default="basic",
        help="Type of test to run",
    )
    parser.add_argument(
        "--count", type=int, default=10, help="Number of episodes for batch tests"
    )
    args = parser.parse_args()

    print("🎯 FrozenLake MCP Simulation Wrapper")
    print("=" * 50)

    if args.test == "basic":
        asyncio.run(run_basic_test())
    elif args.test == "seeds":
        asyncio.run(run_seed_test(args.count))
    elif args.test == "performance":
        asyncio.run(run_performance_test(args.count))


if __name__ == "__main__":
    main()
