"""
FrozenLake Simulation MCP Server - Refactored Version

This simulation server implements FrozenLake using the unified SimulationServerBase
framework, following the proper MCP patterns.

Usage:
    python simulation_server_new.py --port 8000 --host 0.0.0.0
"""

import argparse
import json
import os
from typing import Any, Dict

from frozen_lake_adapter import FrozenLakeAdapter
from frozen_lake_mcp_server import FrozenLakeProdServer

from reward_kit.mcp import SimulationServerBase
from reward_kit.mcp.grid_renderer import render_grid
from reward_kit.mcp.simulation_server import simulation_resource, simulation_tool


class FrozenLakeSimServer(FrozenLakeAdapter, SimulationServerBase):
    """FrozenLake simulation server using unified framework."""

    def __init__(self):
        # Create production server for validation
        prod_server = FrozenLakeProdServer()

        # Initialize simulation server
        SimulationServerBase.__init__(
            self, "FrozenLake-Simulation", production_server_app=prod_server.mcp
        )
        FrozenLakeAdapter.__init__(self)

    @simulation_tool
    def lake_move(self, action: str, *, ctx, session_state) -> Dict[str, Any]:
        """
        Move in the FrozenLake game - simulation version.

        This matches the production server signature exactly.
        """
        # Validate action
        if not action or not isinstance(action, str):
            raise ValueError(
                f"Invalid action parameter: '{action}'. "
                f"Must be a non-empty string. Valid actions: LEFT, DOWN, RIGHT, UP"
            )

        action = action.strip().upper()

        # Parse and execute action
        try:
            action_int = self.parse_action(action)
        except ValueError as e:
            raise ValueError(str(e))

        # Execute step
        env = session_state["env"]
        obs, reward, terminated, truncated, info = self.step_environment(
            env, action_int
        )

        # Update session state
        session_state["steps"] += 1
        session_state["total_reward"] = session_state.get("total_reward", 0.0) + reward

        # Format response to match production server
        result = {
            "position": int(self.format_observation(obs)),
            "grid": render_grid(env.desc, obs),
            "action": action,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "moves": int(session_state["steps"]),
        }

        # Add info if available
        if info:
            result["info"] = info

        return result

    @simulation_resource("game://initial_state")
    def initial_state(self, *, ctx, session_state) -> str:
        """Get initial state resource for simulation."""
        env = session_state["env"]
        initial_observation = session_state["initial_observation"]

        initial_state = {
            "position": int(initial_observation),
            "grid": render_grid(env.desc, initial_observation),
            "moves": 0,
            "terminated": False,
            "truncated": False,
            "reward": 0.0,
            "info": {
                "grid_size": "4x4",
                "initial_position": int(initial_observation),
                "seed": session_state.get("seed", 42),
            },
        }
        return json.dumps(initial_state)


def main():
    """Main entry point for FrozenLake simulation server."""
    parser = argparse.ArgumentParser(description="FrozenLake Simulation Server")
    parser.add_argument("--port", default=8001, type=int, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")

    args = parser.parse_args()

    print(f"🚀 Starting FrozenLake Simulation Server")
    print(f"🌐 Host: {args.host}")
    print(f"🌐 Port: {args.port}")
    print("🎯 Framework: Unified SimulationServerBase")
    print()

    # Set port environment variable
    os.environ["PORT"] = str(args.port)

    # Create and run server
    server = FrozenLakeSimServer()
    server.run()


if __name__ == "__main__":
    main()
