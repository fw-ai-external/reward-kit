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
    def get_initial_state_resource(
        self, ctx: Context, session_state: Dict[str, Any]
    ) -> str:
        """
        MCP Resource: Provides initial game state for simulation.

        This follows the proper MCP pattern where initial state comes from resources
        during session establishment, not from tool calls.

        Now properly uses the seeded environment from session_state.
        """
        # Use the actual initial observation from the seeded environment
        initial_observation = session_state["initial_observation"]

        # Get the actual environment to determine the starting position
        env = session_state["env"]
        if hasattr(env, "s"):  # FrozenLake stores current state in env.s
            position = int(env.s)  # Convert numpy int64 to Python int
        else:
            position = int(initial_observation)  # Ensure it's a Python int

        # Get actual grid size from environment (consistent with config method)
        grid_size = (
            len(env.desc) if hasattr(env, "desc") and env.desc is not None else 4
        )

        initial_state = {
            "position": int(initial_observation),  # Convert to Python int
            "grid_layout": self._get_grid_layout(position, env),
            "moves": int(session_state["steps"]),  # Convert to Python int
            "terminated": False,
            "truncated": False,
            "reward": 0.0,
            "info": {
                "grid_size": f"{grid_size}x{grid_size}",
                "initial_position": int(position),  # Convert to Python int
                "seed": int(session_state["seed"]),  # Convert to Python int
            },
        }
        return json.dumps(initial_state)

    @simulation_resource("game://frozen_lake/config")
    def get_game_config_resource(
        self, ctx: Context, session_state: Dict[str, Any]
    ) -> str:
        """MCP Resource: Provides game configuration information for simulation."""
        env = session_state["env"]
        grid_size = (
            len(env.desc) if hasattr(env, "desc") and env.desc is not None else 4
        )

        config = {
            "game_type": "FrozenLake",
            "version": "simulation-v1",
            "grid_size": f"{grid_size}x{grid_size}",
            "deterministic": True,
            "actions": ["LEFT", "DOWN", "RIGHT", "UP"],
            "seed": session_state["seed"],  # Include the seed in config
            "session_id": session_state["session_id"],
        }
        return json.dumps(config)

    def _get_grid_layout(self, position: int, env=None) -> str:
        """Get the grid layout showing current player position from the actual environment."""
        if env is None:
            # Fallback to a simple representation if no environment provided
            return f"Position: {position}"

        if not hasattr(env, "desc") or env.desc is None:
            return f"Position: {position} (no map available)"

        # Get the actual map from the environment
        desc = env.desc
        size = len(desc)

        # Convert position to row, col coordinates
        row = position // size
        col = position % size

        # Create grid representation from the actual environment
        grid_lines = []
        for r, desc_row in enumerate(desc):
            line = ""
            for c, cell in enumerate(desc_row):
                # Decode bytes to string if needed
                cell_char = (
                    cell.decode("utf-8") if isinstance(cell, bytes) else str(cell)
                )

                # Show current position, preserving original cell type except for start
                if r == row and c == col:
                    if cell_char == "H":
                        line += "X"  # Player fell in hole (dead)
                    elif cell_char == "G":
                        line += "W"  # Player reached goal (won!)
                    elif cell_char == "S":
                        line += "S"  # Player at start position
                    else:
                        line += "P"  # Player on frozen safe tile
                else:
                    line += cell_char

            grid_lines.append(line)

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
        # Explicit parameter validation to catch empty/missing action
        if not action or not isinstance(action, str):
            raise ValueError(
                f"Invalid action parameter: '{action}'. "
                f"Must be a non-empty string. Valid actions: LEFT, DOWN, RIGHT, UP"
            )

        action = action.strip().upper()  # Normalize the action

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
            "position": int(self.format_observation(obs)),  # Convert to Python int
            "grid_layout": self._get_grid_layout(int(obs), env),  # Convert obs to int
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "moves": int(session_state["steps"]),  # Convert to Python int
            "total_reward": float(
                session_state["total_reward"]
            ),  # Convert to Python float
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

    def _get_or_create_session(self, ctx: Context) -> Dict[str, Any]:
        """
        Get or create session and return its state.

        This handles session initialization using MCP spec:
        - Configuration from client info
        - Automatic environment creation
        - Internal session management (no tools exposed)

        Returns:
            Session state dictionary instead of injecting into context
        """
        session_id = self._get_session_id(ctx)

        with self.session_lock:
            if session_id not in self.sessions:
                # Extract seed and config from MCP client info if available
                config = self.get_default_config()
                seed = None

                # Get configuration from client info if available
                if hasattr(ctx, "session") and hasattr(ctx.session, "client_info"):
                    client_info = ctx.session.client_info
                    if client_info and hasattr(client_info, "_extra"):
                        extra_data = client_info._extra
                        if extra_data:
                            # Extract seed from client info
                            seed = extra_data.get("seed")

                            # Get environment context from client info's config
                            if "config" in extra_data:
                                environment_context = extra_data["config"]
                                # Extract seed from environment context if not found at top level
                                if seed is None and "seed" in environment_context:
                                    seed = environment_context["seed"]

                                # Update config with environment context
                                config.update(environment_context)

                                print(
                                    f"🔧 Using environment context: {environment_context}"
                                )
                                print(f"🌱 Using seed: {seed}")

                # Use create_environment_with_seed if available (for proper seeding)
                # Otherwise fall back to separate create and reset
                if hasattr(self, "create_environment_with_seed"):
                    env, obs, info = self.create_environment_with_seed(
                        config, seed=seed
                    )
                else:
                    env = self.create_environment(config)
                    obs, info = self.reset_environment(env, seed=seed)

                self.sessions[session_id] = {
                    "env": env,
                    "config": config,
                    "seed": seed,
                    "created_at": time.time(),
                    "initial_observation": self.format_observation(obs),
                    "session_id": session_id,
                    "steps": 0,
                    "total_reward": 0.0,
                    "last_used": time.time(),
                }
                print(
                    f"🆕 Simulation session created: {session_id[:16]}... (seed={seed}, config={config})"
                )

            self.sessions[session_id]["last_used"] = time.time()
            return self.sessions[session_id]


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
