#!/usr/bin/env python3
"""
FrozenLake MCP Server - Refactored Production Implementation

This is the refactored FrozenLake MCP server using the unified GymProductionServer framework.
Uses the new unified API as documented in the proposal.

Features:
- **UNIFIED PATTERN**: Uses GymProductionServer base class
- Initial state provided via MCP resources
- Tools used only for actions (lake_move), not for getting initial state
- Dynamic grid layout showing player position
- Proper goal detection and reward handling
- OpenAI-compatible tool calling interface

MCP Integration:
- Initial state available through "game://initial_state" resource
- Tools provide actions, resources provide state/configuration data
"""

import argparse
import os
from typing import Any, Dict

from frozen_lake_adapter import FrozenLakeAdapter
from mcp.server.fastmcp import Context

from reward_kit.mcp import GymProductionServer
from reward_kit.mcp.grid_renderer import render_grid


class FrozenLakeProdServer(GymProductionServer):
    """FrozenLake production server using unified framework."""

    def __init__(self, seed: int = None):
        super().__init__("FrozenLake-v1", FrozenLakeAdapter())
        if seed is not None:
            # The key change to make the environment reproducible
            self.env.reset(seed=seed)

    def _register_tools(self):
        """Register domain-specific tools."""

        @self.mcp.tool(
            name="lake_move",
            description="Move on the frozen lake. Actions: LEFT, DOWN, RIGHT, UP",
        )
        def lake_move(action: str, ctx: Context) -> Dict[str, Any]:
            """
            Move in the FrozenLake game.

            Args:
                action: Direction to move (LEFT, DOWN, RIGHT, UP)

            Returns:
                Game state with position, reward, and completion status
            """
            # Production server: single-session, no seed handling
            # For multi-session with seeds, use simulation_server.py

            # Validate action
            if not action or not isinstance(action, str):
                raise ValueError(
                    f"Invalid action parameter: '{action}'. "
                    f"Must be a non-empty string. Valid actions: LEFT, DOWN, RIGHT, UP"
                )

            action = action.strip().upper()

            # Parse and execute action
            try:
                action_int = self.adapter.parse_action(action)
            except ValueError as e:
                raise ValueError(str(e))

            # Execute move
            obs, reward, terminated, truncated, info = self.adapter.step_environment(
                self.env, action_int
            )

            # Update global state
            self.obs = obs

            # Log the result
            if terminated or truncated:
                status = "🏆 GOAL!" if reward > 0 else "💀 HOLE!"
                print(f"🎮 Game ended: {status}")
            else:
                print(f"🎮 {action} → position {obs}")

            # Return complete state including dynamic grid layout
            result = {
                **self._render(obs),
                "action": action,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }

            # Add info if available
            if info:
                result["info"] = info

            return result

    @staticmethod
    def format_observation(obs: int, env: Any) -> Dict[str, Any]:
        """Format observation for MCP response."""
        return {
            "position": int(obs),
            "grid": render_grid(env.desc, obs),
        }


def main():
    """Run the FrozenLake MCP server."""
    parser = argparse.ArgumentParser(description="FrozenLake MCP Server")
    parser.add_argument(
        "--transport",
        choices=["streamable-http", "stdio"],
        default="streamable-http",
        help="Transport protocol to use",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for HTTP transport"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for the environment"
    )

    args = parser.parse_args()

    # Set environment variable for HTTP port
    if args.transport == "streamable-http":
        os.environ["PORT"] = str(args.port)

    # Create and run server
    server = FrozenLakeProdServer(seed=args.seed)
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
