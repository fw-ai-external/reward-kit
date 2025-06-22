"""
MCP Environment API for reward-kit north star implementation.

This module provides the `rk.make()` and `rk.rollout()` functions that enable
the north star developer experience for MCP-based environments.

Usage:
    import reward_kit as rk

    envs = rk.make("http://localhost:8000/mcp/lake@mcp", n=100, seeds=seeds)
    trajectories = rk.rollout(envs, policy=policy, steps=512)
"""

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import httpx


@dataclass
class MCPSession:
    """Represents a single MCP session with an environment."""

    session_id: str
    base_url: str
    seed: Optional[int]
    model_id: str
    client: httpx.AsyncClient
    terminated: bool = False
    last_observation: Any = None


@dataclass
class Trajectory:
    """Represents a complete rollout trajectory."""

    session: MCPSession
    observations: List[Any]
    actions: List[str]
    rewards: List[float]
    terminated: bool
    total_reward: float
    steps: int


class MCPVectorEnv:
    """
    Vector environment interface for MCP sessions.

    Provides Gymnasium-like interface for parallel MCP sessions.
    """

    def __init__(self, sessions: List[MCPSession]):
        self.sessions = sessions
        self.n = len(sessions)

    async def reset(self) -> List[Any]:
        """Reset all environments and return initial observations."""
        observations = []

        async def reset_session(session: MCPSession) -> Any:
            # Get initial observation - session creation happens automatically
            # This follows the MCP protocol where sessions are managed transparently
            response = await session.client.post(
                f"{session.base_url}/call_tool",
                json={"name": "get_initial_observation", "arguments": {}},
            )
            response.raise_for_status()

            data = response.json()
            result = data["content"][0]["text"]

            # Session ID is managed by the MCP transport layer
            # We don't need to extract it explicitly
            session.terminated = False

            session.last_observation = result["initialObservation"]
            return session.last_observation

        tasks = [reset_session(session) for session in self.sessions]
        observations = await asyncio.gather(*tasks)

        return observations

    async def step(
        self, actions: List[str]
    ) -> tuple[List[Any], List[float], List[bool], List[Dict]]:
        """
        Take parallel steps in all environments.

        Args:
            actions: List of actions, one per environment

        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        if len(actions) != self.n:
            raise ValueError(f"Expected {self.n} actions, got {len(actions)}")

        async def step_session(session: MCPSession, action: str):
            if session.terminated:
                return session.last_observation, 0.0, True, {}

            response = await session.client.post(
                f"{session.base_url}/call_tool",
                json={
                    "name": "lake_move",
                    "arguments": {
                        "action": action
                        # No session_id needed - handled by MCP transport layer
                    },
                },
            )
            response.raise_for_status()

            data = response.json()
            result = data["content"][0]["text"]

            session.last_observation = result["observation"]
            session.terminated = result["terminated"] or result["truncated"]

            return (
                result["observation"],
                result["reward"],
                session.terminated,
                result["info"],
            )

        tasks = [
            step_session(session, action)
            for session, action in zip(self.sessions, actions)
        ]
        results = await asyncio.gather(*tasks)

        observations, rewards, dones, infos = zip(*results)
        return list(observations), list(rewards), list(dones), list(infos)

    async def close(self):
        """Close all sessions and HTTP clients."""

        async def close_session(session: MCPSession):
            try:
                # Session cleanup happens automatically when HTTP connection closes
                # No explicit delete tool needed - this follows MCP protocol
                await session.client.aclose()
            except Exception:
                pass  # Ignore cleanup errors

        tasks = [close_session(session) for session in self.sessions]
        await asyncio.gather(*tasks, return_exceptions=True)


def make(
    env_spec: str, n: int, seeds: Optional[List[int]] = None, model_id: str = "unknown"
) -> MCPVectorEnv:
    """
    Create a vector of MCP environment sessions.

    Args:
        env_spec: Environment specification like "http://localhost:8000@mcp"
        n: Number of parallel environments
        seeds: List of seeds, one per environment (optional)
        model_id: Model identifier to pass to sessions

    Returns:
        MCPVectorEnv instance ready for rollouts

    Example:
        envs = rk.make("http://localhost:8000@mcp", n=100, seeds=seeds)
    """
    # Parse environment specification
    if "@mcp" not in env_spec:
        raise ValueError(
            "Environment spec must end with '@mcp' to indicate MCP protocol"
        )

    base_url = env_spec.replace("@mcp", "")
    if not base_url.startswith("http"):
        raise ValueError("Environment spec must be a valid HTTP URL")

    # Generate seeds if not provided
    if seeds is None:
        import random

        seeds = [random.randint(0, 2**31 - 1) for _ in range(n)]
    elif len(seeds) != n:
        raise ValueError(f"Expected {n} seeds, got {len(seeds)}")

    # Create sessions
    sessions = []
    for i in range(n):
        session = MCPSession(
            session_id="",  # Will be set during reset
            base_url=base_url,
            seed=seeds[i],
            model_id=model_id,
            client=httpx.AsyncClient(timeout=30.0),
        )
        sessions.append(session)

    return MCPVectorEnv(sessions)


class FireworksPolicy:
    """
    Simple policy wrapper for Fireworks API that matches north star example.
    """

    def __init__(self, model_id: str, temperature: float = 0.2):
        self.model_id = model_id
        self.temperature = temperature

    async def __call__(self, observations: List[Any]) -> List[str]:
        """
        Generate actions based on observations.

        For now, this is a simple heuristic policy.
        In the full implementation, this would call the Fireworks API.
        """
        actions = []
        for obs in observations:
            # Simple policy: try to go down and right towards goal
            # In a real implementation, this would use LLM reasoning
            if isinstance(obs, int):
                if obs < 4:  # Top row, go down
                    actions.append("DOWN")
                elif obs % 4 == 0:  # Left column, go right
                    actions.append("RIGHT")
                else:
                    actions.append("DOWN")  # Default
            else:
                actions.append("DOWN")  # Fallback

        return actions


async def rollout(
    envs: MCPVectorEnv, policy: Union[FireworksPolicy, Callable], steps: int = 512
) -> List[Trajectory]:
    """
    Execute parallel rollouts across multiple MCP environments.

    Args:
        envs: MCPVectorEnv instance
        policy: Policy function that takes observations and returns actions
        steps: Maximum steps per rollout

    Returns:
        List of Trajectory objects with complete rollout data

    Example:
        trajectories = rk.rollout(envs, policy=policy, steps=512)
    """
    # Initialize trajectories
    trajectories = []
    for session in envs.sessions:
        trajectories.append(
            Trajectory(
                session=session,
                observations=[],
                actions=[],
                rewards=[],
                terminated=False,
                total_reward=0.0,
                steps=0,
            )
        )

    # Reset environments
    initial_observations = await envs.reset()

    # Record initial observations
    for trajectory, obs in zip(trajectories, initial_observations):
        trajectory.observations.append(obs)

    # Run rollouts
    current_observations = initial_observations

    for step in range(steps):
        # Check if all environments are done
        active_envs = [i for i, traj in enumerate(trajectories) if not traj.terminated]
        if not active_envs:
            break

        # Get actions from policy
        if callable(policy):
            actions = await policy(current_observations)
        else:
            # Handle non-async policies
            actions = policy(current_observations)
            if asyncio.iscoroutine(actions):
                actions = await actions

        # Take steps
        observations, rewards, dones, infos = await envs.step(actions)

        # Update trajectories
        for i, (trajectory, action, obs, reward, done) in enumerate(
            zip(trajectories, actions, observations, rewards, dones)
        ):
            if not trajectory.terminated:
                trajectory.actions.append(action)
                trajectory.observations.append(obs)
                trajectory.rewards.append(reward)
                trajectory.total_reward += reward
                trajectory.steps += 1
                trajectory.terminated = done

        current_observations = observations

    # Clean up
    await envs.close()

    return trajectories


# Convenience function for testing as mentioned in north star
async def test_mcp(base_url: str, seeds: List[int]) -> Dict[str, Any]:
    """
    Test function for validating MCP server as mentioned in north star document.

    Args:
        base_url: Base URL of MCP server (e.g., "http://localhost:8000/mcp")
        seeds: List of seeds to test

    Returns:
        Test results dictionary
    """
    results = {"total_tests": len(seeds), "successful": 0, "failed": 0, "results": []}

    for seed in seeds:
        try:
            # Create single environment
            envs = make(f"{base_url}@mcp", n=1, seeds=[seed], model_id="test-model")

            # Simple policy for testing
            policy = FireworksPolicy("test-model")

            # Run short rollout
            trajectories = await rollout(envs, policy=policy, steps=10)

            if trajectories and len(trajectories[0].observations) > 1:
                results["successful"] += 1
                results["results"].append(
                    {
                        "seed": seed,
                        "status": "success",
                        "steps": trajectories[0].steps,
                        "total_reward": trajectories[0].total_reward,
                    }
                )
            else:
                results["failed"] += 1
                results["results"].append(
                    {"seed": seed, "status": "failed", "error": "empty_trajectory"}
                )

        except Exception as e:
            results["failed"] += 1
            results["results"].append(
                {"seed": seed, "status": "failed", "error": str(e)}
            )

    return results


# Add to reward_kit.__init__.py exports
__all__ = ["make", "rollout", "FireworksPolicy", "MCPVectorEnv", "test_mcp"]
