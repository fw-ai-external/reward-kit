"""
FrozenLake Simulation MCP Server

This simulation server implements FrozenLake with simulation capabilities
using the reward-kit simulation framework. It demonstrates:

1. No session management tools exposed (framework enforced)
2. Session initialization via MCP initializationOptions
3. Only domain game tools exposed to models
4. Independent from production server (no proxying)

This is a completely separate implementation from the production server,
similar to how a Google Docs simulation would be separate from Google Docs production.

Usage:
    python simulation_server.py --transport streamable-http --port 8000
"""

import argparse
import time
from typing import Any, Callable, Dict, Optional, Tuple

# Import production server to ensure signature matching
import frozen_lake_server
from gymnasium.envs.toy_text import FrozenLakeEnv

from reward_kit.mcp.simulation_server import SimulationServerBase


class FrozenLakeSimulation(SimulationServerBase):
    """
    FrozenLake simulation server using the framework.

    This implements FrozenLake with simulation capabilities but is
    completely independent from the production server.
    """

    def create_environment(self, config: Dict[str, Any]) -> FrozenLakeEnv:
        """Create FrozenLake environment for simulation."""
        return FrozenLakeEnv(
            map_name=config.get("map_name", "4x4"),
            is_slippery=config.get(
                "is_slippery", False
            ),  # Deterministic for simulation
            render_mode=None,
        )

    def reset_environment(
        self, env: FrozenLakeEnv, seed: Optional[int] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """Reset FrozenLake environment."""
        return env.reset(seed=seed)

    def step_environment(
        self, env: FrozenLakeEnv, action: int
    ) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """Execute step in FrozenLake environment."""
        return env.step(action)

    def close_environment(self, env: FrozenLakeEnv) -> None:
        """Close FrozenLake environment."""
        if hasattr(env, "close"):
            env.close()

    def parse_action(self, action_str: str) -> int:
        """Parse action string to FrozenLake action integer."""
        action_map = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}
        if action_str not in action_map:
            raise ValueError(
                f"Invalid action '{action_str}'. Use: {', '.join(action_map.keys())}"
            )
        return action_map[action_str]

    def format_observation(self, observation: int) -> int:
        """Format FrozenLake observation."""
        return int(observation)

    def get_default_config(self) -> Dict[str, Any]:
        """Get default FrozenLake simulation configuration."""
        return {
            "map_name": "4x4",
            "is_slippery": False,  # Deterministic for consistent simulation
            "render_mode": None,
        }

    def get_domain_tools(self) -> Dict[str, Callable]:
        """
        Get FrozenLake domain tools.

        IMPORTANT: Must match production server tools exactly.
        Production server tools: lake_move
        """
        # Validate we match production server tools exactly
        production_tools = set(frozen_lake_server.app._tool_manager._tools.keys())
        simulation_tools = {"lake_move"}

        assert (
            simulation_tools == production_tools
        ), f"Tool mismatch! Production: {production_tools}, Simulation: {simulation_tools}"

        return {"lake_move": self._lake_move}

    def _lake_move(self, session_data: Dict[str, Any], action: str) -> Dict[str, Any]:
        """
        Execute a move in the FrozenLake simulation.

        Args:
            session_data: Session state (injected by framework)
            action: Movement direction (LEFT, DOWN, RIGHT, UP)

        Returns:
            Game state with position, reward, and completion status
        """
        env = session_data["env"]
        session_id = session_data["session_id"]

        # Parse and validate action
        try:
            action_int = self.parse_action(action)
        except ValueError as e:
            raise ValueError(str(e))

        # Execute step
        obs, reward, terminated, truncated, info = self.step_environment(
            env, action_int
        )

        # Update session state
        session_data["steps"] += 1
        session_data["total_reward"] += reward
        session_data["last_used"] = time.time()

        # Format response
        result = {
            "position": self.format_observation(obs),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "moves": session_data["steps"],
            "total_reward": session_data["total_reward"],
            "info": info or {},
        }

        # Simulation logging
        if terminated:
            status = "🏆 WON!" if reward > 0 else "💀 LOST!"
            print(f"🎮 {session_id[:12]}... {action} → {obs} | {status} (simulation)")
        else:
            print(
                f"🎮 {session_id[:12]}... {action} → {obs} | Move #{session_data['steps']}"
            )

        return result


def main():
    """Simulation server entry point."""
    parser = argparse.ArgumentParser(description="FrozenLake Simulation MCP Server")
    parser.add_argument(
        "--transport",
        default="streamable-http",
        choices=["stdio", "sse", "streamable-http"],
        help="MCP transport protocol",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for simulation server"
    )
    args = parser.parse_args()

    print(f"🚀 Starting FrozenLake Simulation Server")
    print(f"📡 Transport: {args.transport}")
    print(f"🌐 Port: {args.port}")
    print(f"🎯 Independent simulation (not proxying production)")
    print(f"🚫 Framework enforces: No session management tools exposed")
    print()

    # Create and run simulation server
    server = FrozenLakeSimulation("FrozenLake-Simulation")

    if args.transport in ["sse", "streamable-http"]:
        server.run(transport=args.transport, port=args.port)
    else:
        server.run(transport=args.transport)


if __name__ == "__main__":
    main()
