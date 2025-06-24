"""
FrozenLake Simulation MCP Server

This simulation server implements FrozenLake with simulation capabilities
using the reward-kit simulation framework. It demonstrates:

1. No session management tools exposed (framework enforced)
2. Session initialization via MCP initializationOptions
3. Only domain game tools exposed to models
4. **PROPER MCP PATTERN**: Initial state via MCP resources
5. Independent from production server (no proxying)

This is a completely separate implementation from the production server,
similar to how a Google Docs simulation would be separate from Google Docs production.

Usage:
    python simulation_server.py --transport streamable-http --port 8000
"""

import argparse
import json
import time
from typing import Any, Dict, Optional, Tuple

# Import production server to provide to the framework for validation
import frozen_lake_mcp_server

# Fixed: Use absolute import instead of relative import
from frozen_lake_adapter import FrozenLakeAdapter
from mcp.server.fastmcp import Context

from reward_kit.mcp.simulation_server import (
    SimulationServerBase,
    simulation_resource,
    simulation_tool,
)


class FrozenLakeSimulation(FrozenLakeAdapter, SimulationServerBase):
    """
    FrozenLake simulation server using the framework.

    This inherits from the FrozenLakeAdapter to handle environment logic
    and from SimulationServerBase for the simulation framework.
    """

    @simulation_resource("game://frozen_lake/initial_state")
    def get_initial_state_resource(self) -> str:
        """
        MCP Resource: Provides initial game state for simulation.

        This follows the proper MCP pattern where initial state comes from resources
        during session establishment, not from tool calls.
        """
        # Get default initial state
        initial_observation = self.format_observation(0)  # Starting position
        initial_state = {
            "position": initial_observation,
            "grid_layout": self._get_grid_layout(0),
            "moves": 0,
            "terminated": False,
            "truncated": False,
            "reward": 0.0,
            "info": {
                "grid_size": "4x4",
                "holes": [5, 7, 11, 12],  # Positions of holes in 4x4 grid
                "goal": 15,  # Goal position
                "initial_position": 0,
            },
        }
        return json.dumps(initial_state)

    @simulation_resource("game://frozen_lake/config")
    def get_game_config_resource(self) -> str:
        """MCP Resource: Provides game configuration information for simulation."""
        config = {
            "game_type": "FrozenLake",
            "version": "simulation-v1",
            "grid_size": "4x4",
            "deterministic": True,
            "holes": [5, 7, 11, 12],
            "goal": 15,
            "actions": ["LEFT", "DOWN", "RIGHT", "UP"],
        }
        return json.dumps(config)

    def _get_grid_layout(self, position: int) -> str:
        """Get the grid layout showing current player position."""
        # Create a simple 4x4 grid representation
        grid_chars = [
            ".",
            ".",
            ".",
            ".",
            ".",
            "H",
            ".",
            "H",
            ".",
            ".",
            ".",
            "H",
            "H",
            ".",
            ".",
            "G",
        ]

        # Show player position with 'P'
        if 0 <= position < len(grid_chars):
            grid_chars[position] = "P"

        # Format as grid
        grid_lines = []
        for i in range(0, 16, 4):
            grid_lines.append("".join(grid_chars[i : i + 4]))

        return "\n".join(grid_lines)

    @simulation_tool
    def lake_move(
        self, action: str, ctx: Context, session_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a move in the FrozenLake simulation.

        The signature of this method (excluding self, ctx, and session_state) must exactly
        match the production tool. The framework validates this automatically.

        Args:
            action: Movement direction (LEFT, DOWN, RIGHT, UP)
            ctx: The MCP Context
            session_state: Session state injected by the framework

        Returns:
            Game state with position, reward, and completion status
        """
        # The framework passes the session state as a parameter
        env = session_state["env"]
        session_id = session_state["session_id"]

        # Parse and validate action
        try:
            action_int = FrozenLakeAdapter.parse_action(self, action)
        except ValueError as e:
            raise ValueError(str(e))

        # Execute step
        obs, reward, terminated, truncated, info = self.step_environment(
            env, action_int
        )

        # Update session state
        session_state["steps"] += 1
        session_state["total_reward"] += reward
        session_state["last_used"] = time.time()

        # Format response with grid layout
        result = {
            "position": self.format_observation(obs),
            "grid_layout": self._get_grid_layout(obs),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "moves": session_state["steps"],
            "total_reward": session_state["total_reward"],
            "info": info or {},
        }

        # Simulation logging
        if terminated:
            status = "🏆 WON!" if reward > 0 else "💀 LOST!"
            print(f"🎮 {session_id[:12]}... {action} → {obs} | {status} (simulation)")
        else:
            print(
                f"🎮 {session_id[:12]}... {action} → {obs} | Move #{session_state['steps']}"
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
    # Pass the production server app to the framework for automatic validation
    server = FrozenLakeSimulation(
        "FrozenLake-Simulation",
        production_server_app=frozen_lake_mcp_server.app,
    )

    # FastMCP.run() doesn't accept port parameter directly
    # Port is configured via PORT environment variable
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
