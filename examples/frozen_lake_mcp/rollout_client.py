"""
Rollout Client for Simulation MCP Servers

This client demonstrates how to perform rollouts using the simulation MCP server.
It only talks to the simulation server, which handles all the complexity of
wrapping production servers.

Architecture:
  This Client -> Simulation Server -> Production Server

Usage:
    # Start production and simulation servers first:
    python frozen_lake_server.py --transport streamable-http --port 8001
    python simulation_server.py --transport streamable-http --port 8000 --prod-url http://localhost:8001

    # Then run rollouts:
    python rollout_client.py --test single --seed 42
    python rollout_client.py --test batch --count 10
"""

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class EpisodeResult:
    """Results from a single episode."""

    success: bool
    duration: float
    steps: int
    final_reward: float
    final_position: int
    seed: Optional[int] = None
    error: Optional[str] = None


@dataclass
class BatchResults:
    """Results from batch testing."""

    total_episodes: int
    successful_episodes: int
    success_rate: float
    avg_duration: float
    avg_steps: float
    avg_reward: float
    goal_rate: float
    errors: List[str]


class SimulationRolloutClient:
    """
    Client for performing rollouts using simulation MCP servers.

    This client only talks to the simulation server, which handles
    all the complexity of session management and proxying to production.
    """

    def __init__(self, simulation_url: str = "http://localhost:8000"):
        """
        Initialize rollout client.

        Args:
            simulation_url: URL of simulation MCP server
        """
        self.simulation_url = simulation_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def run_episode(
        self, seed: Optional[int] = None, max_steps: int = 50
    ) -> EpisodeResult:
        """
        Run a single episode using the simulation server.

        Args:
            seed: Random seed for reproducibility
            max_steps: Maximum steps before truncation

        Returns:
            EpisodeResult with episode outcome
        """
        start_time = time.time()
        steps = 0
        total_reward = 0.0
        final_position = 0

        try:
            print(f"🎮 Starting episode (seed={seed})")

            # Session is initialized automatically by the framework
            # No session tools exposed - simulation manages internally
            # FrozenLake always starts at position 0, so no need to query initial state
            print(f"✅ Framework-managed session (starts at position 0)")

            # Run episode
            terminated = False
            truncated = False

            while not terminated and not truncated and steps < max_steps:
                # Simple policy for testing
                if steps % 2 == 0:
                    action = "DOWN"
                else:
                    action = "RIGHT"

                # Adjust for boundaries (simple heuristic)
                if final_position // 4 == 3:  # Bottom row
                    action = "RIGHT"
                elif final_position % 4 == 3:  # Right column
                    action = "DOWN"

                # Execute move via simulation server
                move_response = await self.client.post(
                    f"{self.simulation_url}/mcp/tools/lake_move",
                    json={"action": action},
                )
                move_response.raise_for_status()
                move_data = move_response.json()

                # Extract results
                final_position = move_data.get("position", 0)
                reward = move_data.get("reward", 0.0)
                terminated = move_data.get("terminated", False)
                truncated = move_data.get("truncated", False)

                total_reward += reward
                steps += 1

                moves = move_data.get("moves", steps)
                total_sim_reward = move_data.get("total_reward", total_reward)
                print(
                    f"  Step {steps}: {action} → pos={final_position}, reward={reward}, total_moves={moves}"
                )

                if terminated or truncated:
                    break

            duration = time.time() - start_time

            return EpisodeResult(
                success=True,
                duration=duration,
                steps=steps,
                final_reward=total_reward,
                final_position=final_position,
                seed=seed,
            )

        except Exception as e:
            duration = time.time() - start_time
            return EpisodeResult(
                success=False,
                duration=duration,
                steps=steps,
                final_reward=total_reward,
                final_position=final_position,
                seed=seed,
                error=str(e),
            )

    async def run_batch(
        self, count: int, seeds: Optional[List[int]] = None
    ) -> BatchResults:
        """
        Run batch of episodes.

        Args:
            count: Number of episodes
            seeds: Optional list of seeds

        Returns:
            BatchResults with aggregated statistics
        """
        if seeds is None:
            seeds = list(range(count))
        elif len(seeds) < count:
            seeds.extend(range(len(seeds), count))

        print(f"🔄 Running {count} episodes via simulation server...")

        results = []
        errors = []
        goal_reached = 0

        for i, seed in enumerate(seeds[:count]):
            print(f"\n--- Episode {i+1}/{count} (seed={seed}) ---")

            result = await self.run_episode(seed=seed)
            results.append(result)

            if not result.success:
                errors.append(f"Episode {i+1}: {result.error}")
                print(f"❌ Episode {i+1} failed: {result.error}")
            else:
                if result.final_reward > 0:
                    goal_reached += 1
                    print(f"🏆 Episode {i+1}: GOAL! ({result.steps} steps)")
                else:
                    print(
                        f"💀 Episode {i+1}: Failed ({result.steps} steps, pos={result.final_position})"
                    )

        # Calculate statistics
        successful_results = [r for r in results if r.success]

        if successful_results:
            avg_duration = statistics.mean(r.duration for r in successful_results)
            avg_steps = statistics.mean(r.steps for r in successful_results)
            avg_reward = statistics.mean(r.final_reward for r in successful_results)
        else:
            avg_duration = avg_steps = avg_reward = 0.0

        return BatchResults(
            total_episodes=count,
            successful_episodes=len(successful_results),
            success_rate=len(successful_results) / count * 100,
            avg_duration=avg_duration,
            avg_steps=avg_steps,
            avg_reward=avg_reward,
            goal_rate=goal_reached / count * 100,
            errors=errors,
        )

    async def close(self):
        """Clean up client resources."""
        await self.client.aclose()


async def test_single_episode(seed: Optional[int] = None):
    """Test single episode via simulation server."""
    print("🧪 Testing single episode via simulation server...")

    client = SimulationRolloutClient()

    try:
        result = await client.run_episode(seed=seed)

        if result.success:
            goal_text = (
                "🏆 REACHED GOAL!"
                if result.final_reward > 0
                else f"💀 Ended at position {result.final_position}"
            )
            print(f"\n✅ Episode completed: {goal_text}")
            print(f"   Duration: {result.duration:.2f}s")
            print(f"   Steps: {result.steps}")
            print(f"   Final reward: {result.final_reward}")
            print(f"   Seed: {result.seed}")
        else:
            print(f"\n❌ Episode failed: {result.error}")

    finally:
        await client.close()


async def test_batch_episodes(count: int):
    """Test batch of episodes via simulation server."""
    print(f"🧪 Testing {count} episodes via simulation server...")

    client = SimulationRolloutClient()

    try:
        results = await client.run_batch(count)

        print(f"\n📊 BATCH RESULTS:")
        print(f"   Total episodes: {results.total_episodes}")
        print(f"   Successful: {results.successful_episodes}")
        print(f"   Success rate: {results.success_rate:.1f}%")
        print(f"   Goal reached: {results.goal_rate:.1f}%")
        print(f"   Average duration: {results.avg_duration:.2f}s")
        print(f"   Average steps: {results.avg_steps:.1f}")
        print(f"   Average reward: {results.avg_reward:.2f}")

        if results.errors:
            print(f"\n❌ ERRORS:")
            for error in results.errors[:3]:
                print(f"   {error}")
            if len(results.errors) > 3:
                print(f"   ... and {len(results.errors) - 3} more errors")

    finally:
        await client.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="FrozenLake Rollout Client")
    parser.add_argument(
        "--test",
        choices=["single", "batch"],
        default="single",
        help="Type of test to run",
    )
    parser.add_argument("--seed", type=int, help="Seed for single episode test")
    parser.add_argument(
        "--count", type=int, default=5, help="Number of episodes for batch test"
    )
    parser.add_argument(
        "--sim-url", default="http://localhost:8000", help="Simulation server URL"
    )
    args = parser.parse_args()

    print("🎯 FrozenLake Rollout Client")
    print("=" * 50)
    print("🏗️  Architecture: Client -> Simulation Server -> Production Server")
    print(f"📡 Simulation server: {args.sim_url}")
    print("🎮 Only talks to simulation server (clean separation)")
    print()

    if args.test == "single":
        asyncio.run(test_single_episode(args.seed))
    elif args.test == "batch":
        asyncio.run(test_batch_episodes(args.count))


if __name__ == "__main__":
    main()
