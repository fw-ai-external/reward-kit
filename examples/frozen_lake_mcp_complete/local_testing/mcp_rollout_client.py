#!/usr/bin/env python3
"""
MCP Rollout Client for FrozenLake

This client demonstrates the correct pattern for MCP tool calling as documented in:
https://github.com/modelcontextprotocol/python-sdk?tab=readme-ov-file#tools

Features:
- Proper async context management with ClientSession
- OpenAI-compatible tool calling interface
- Linear trajectory conversation support
- Comprehensive error handling
"""

import argparse
import asyncio
import statistics
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


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


class FixedMCPRolloutClient:
    """
    MCP rollout client using a managed async context to avoid race conditions.

    This implementation uses the async context manager pattern (`async with`) to ensure
    that all MCP sessions are created and destroyed within the same async context,
    resolving potential `anyio` task group violations during cleanup.
    """

    def __init__(self, server_url: str = "http://localhost:8000/mcp"):
        """
        Initialize rollout client.

        Args:
            server_url: URL of MCP server (no trailing slash)
        """
        self.server_url = server_url
        self._exit_stack = AsyncExitStack()
        self._mcp_sessions: Dict[Any, ClientSession] = {}
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        """Enter the async context, preparing the exit stack."""
        await self._exit_stack.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context, closing all managed sessions."""
        print(f"🔄 Closing {len(self._mcp_sessions)} MCP sessions...")
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
        print("✅ All MCP sessions closed.")

    async def _get_or_create_session(self, seed: Optional[int]) -> ClientSession:
        """Get or create a managed MCP session for the given seed."""
        session_key = seed if seed is not None else "default"

        async with self._lock:
            if session_key in self._mcp_sessions:
                return self._mcp_sessions[session_key]

            # Create new managed session within the client's exit stack.
            # The exit stack will ensure proper cleanup even if errors occur.
            read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
                streamablehttp_client(self.server_url)
            )

            mcp_session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            await mcp_session.initialize()

            self._mcp_sessions[session_key] = mcp_session

            print(f"✅ Created managed MCP session for seed={seed}")
            return mcp_session

    async def run_episode(
        self, seed: Optional[int] = None, max_steps: int = 50
    ) -> EpisodeResult:
        """
        Run a single episode using a managed MCP session.

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

            session = await self._get_or_create_session(seed)

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

                # Execute move via MCP protocol using managed session
                tool_result = await session.call_tool("lake_move", {"action": action})

                # Extract results from MCP response
                if tool_result.content and len(tool_result.content) > 0:
                    content = tool_result.content[0]
                    if hasattr(content, "text"):
                        import json

                        move_data = json.loads(content.text)
                    else:
                        move_data = content
                else:
                    raise RuntimeError(f"Unexpected tool result format: {tool_result}")

                # Extract results
                final_position = move_data.get("position", 0)
                reward = move_data.get("reward", 0.0)
                terminated = move_data.get("terminated", False)
                truncated = move_data.get("truncated", False)

                total_reward += reward
                steps += 1

                moves = move_data.get("moves", steps)
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
            print(f"❌ Episode failed with exception: {e}")
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
        Run batch of episodes concurrently, each with its own managed session.

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

        print(f"🔄 Running {count} episodes concurrently, each in its own context...")

        # Run each episode in its own client context to ensure task isolation
        tasks = [run_and_get_result(seed, self.server_url) for seed in seeds[:count]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        processed_results = []
        errors = []
        goal_reached = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = f"Episode {i+1}: {str(result)}"
                errors.append(error_msg)
                print(f"❌ Episode {i+1} failed: {result}")
                # Create a failed result
                processed_results.append(
                    EpisodeResult(
                        success=False,
                        duration=0.0,
                        steps=0,
                        final_reward=0.0,
                        final_position=0,
                        seed=seeds[i] if i < len(seeds) else None,
                        error=str(result),
                    )
                )
            else:
                processed_results.append(result)
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
        successful_results = [r for r in processed_results if r.success]

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


async def run_and_get_result(seed: int, server_url: str) -> EpisodeResult:
    """Helper to run a single episode within its own client context."""
    async with FixedMCPRolloutClient(server_url) as client:
        return await client.run_episode(seed=seed)


async def test_single_episode(
    seed: Optional[int] = None, server_url: str = "http://localhost:8000/mcp"
):
    """Test single episode via MCP server."""
    print("🧪 Testing single episode via MCP server...")

    async with FixedMCPRolloutClient(server_url) as client:
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
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")


async def test_batch_episodes(
    count: int, server_url: str = "http://localhost:8000/mcp"
):
    """Test batch of episodes via MCP server."""
    print(f"🧪 Testing {count} episodes via MCP server...")

    # The client is now used to orchestrate the batch, but each episode
    # will create its own client instance to ensure context isolation.
    orchestrator_client = FixedMCPRolloutClient(server_url)
    try:
        results = await orchestrator_client.run_batch(count)

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
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during batch execution: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fixed FrozenLake MCP Rollout Client")
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
        "--server-url",
        default="http://localhost:8000/mcp",
        help="MCP server URL (no trailing slash)",
    )
    args = parser.parse_args()

    print("🎯 Fixed FrozenLake MCP Rollout Client")
    print("=" * 42)
    print("🏗️  Using: Async context manager for session handling")
    print(f"📡 MCP server: {args.server_url}")
    print("🔧 Fixed: Proper async lifecycle management")
    print()

    if args.test == "single":
        asyncio.run(test_single_episode(args.seed, args.server_url))
    elif args.test == "batch":
        asyncio.run(test_batch_episodes(args.count, args.server_url))


if __name__ == "__main__":
    main()
