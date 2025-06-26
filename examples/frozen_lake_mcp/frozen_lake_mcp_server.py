#!/usr/bin/env python3
"""
FrozenLake MCP Server - Production Implementation

This is the main FrozenLake MCP server with full linear trajectory support.
Uses FastMCP stateless configuration as documented in:
https://github.com/modelcontextprotocol/python-sdk

Features:
- **PROPER MCP PATTERN**: Initial state provided via MCP resources
- Tools used only for actions (lake_move), not for getting initial state
- Dynamic grid layout showing player position
- Proper goal detection and reward handling
- OpenAI-compatible tool calling interface

MCP Integration:
- Initial state available through "game://frozen_lake/initial_state" resource
- Tools provide actions, resources provide state/configuration data
"""

import argparse
import os
from typing import Any, Dict, Optional

from gymnasium.envs.toy_text import FrozenLakeEnv
from mcp.server.fastmcp import Context, FastMCP

# Global game state - single session for production simplicity
GAME_ENV: FrozenLakeEnv = None
CURRENT_POSITION: int = 0
TOTAL_MOVES: int = 0

# Global seed for client info
CURRENT_SEED: Optional[int] = None


def initialize_game(seed: Optional[int] = None):
    """Initialize global game state with optional seed."""
    global GAME_ENV, CURRENT_POSITION, TOTAL_MOVES

    print("🎮 Game: 4x4 FrozenLake (deterministic)")
    print("📡 Stateless production server (single global session)")
    print("🔗 MCP Resources: Initial state via game://frozen_lake/initial_state")

    GAME_ENV = FrozenLakeEnv(
        map_name="4x4",
        is_slippery=False,  # Deterministic for production
        render_mode=None,
    )
    CURRENT_POSITION, _ = GAME_ENV.reset(seed=seed)
    TOTAL_MOVES = 0
    if seed is not None:
        print(f"🎯 Game initialized at position {CURRENT_POSITION} with seed {seed}")
    else:
        print(f"🎯 Game initialized at position {CURRENT_POSITION}")


def get_current_grid_layout() -> str:
    """Get the dynamic grid layout showing current player position."""
    global GAME_ENV, CURRENT_POSITION, TOTAL_MOVES

    grid_layout = ""
    if hasattr(GAME_ENV, "desc"):
        # Create dynamic grid showing current player position
        grid_with_player = []
        for i, row in enumerate(GAME_ENV.desc):
            row_chars = []
            for j, cell in enumerate(row):
                pos = i * 4 + j  # Convert 2D to 1D position
                cell_char = (
                    cell.decode("utf-8") if isinstance(cell, bytes) else str(cell)
                )

                if pos == CURRENT_POSITION:
                    # Show player position with 'P'
                    row_chars.append("P")
                else:
                    # Show original cell
                    row_chars.append(cell_char)
            grid_with_player.append("".join(row_chars))
        grid_layout = "\n".join(grid_with_player)

    return grid_layout


def get_initial_state() -> Dict[str, Any]:
    """Get the initial game state - used for MCP resource."""
    global GAME_ENV, CURRENT_POSITION, TOTAL_MOVES

    return {
        "position": CURRENT_POSITION,
        "grid_layout": get_current_grid_layout(),
        "moves": TOTAL_MOVES,
        "terminated": False,
        "truncated": False,
        "reward": 0.0,
        "info": {
            "grid_size": "4x4",
            "holes": [5, 7, 11, 12],  # Positions of holes in 4x4 grid
            "goal": 15,  # Goal position
            "initial_position": CURRENT_POSITION,
        },
    }


# Create FastMCP server with streamable HTTP configuration
# For Cloud Run, we need to bind to 0.0.0.0 and use PORT environment variable
# Remove stateless_http=True as it might default to SSE transport
app = FastMCP(
    "FrozenLake-v1",
    host="0.0.0.0",  # Bind to all interfaces for Cloud Run
    port=int(os.environ.get("PORT", 8000)),  # Use Cloud Run PORT env var
)


def extract_seed_from_context(ctx: Context) -> Optional[int]:
    """Extract seed from MCP client info if available."""
    global CURRENT_SEED

    if hasattr(ctx, "session") and hasattr(ctx.session, "client_info"):
        client_info = ctx.session.client_info
        if client_info and hasattr(client_info, "_extra"):
            extra_data = client_info._extra
            if extra_data and "seed" in extra_data:
                seed = extra_data["seed"]
                if CURRENT_SEED != seed:
                    print(f"🌱 New seed from client: {seed}")
                    initialize_game(seed=seed)
                    CURRENT_SEED = seed
                return seed
    return CURRENT_SEED


# PROPER MCP PATTERN: Provide initial state through resources
@app.resource("game://frozen_lake/initial_state")
def get_initial_state_resource() -> str:
    """
    MCP Resource: Provides initial game state.

    This is the CORRECT way to provide initial state in MCP - through resources
    during session establishment, not from tool calls.

    Note: For seed handling in production server, we'll check client info in tools.
    """
    import json

    return json.dumps(get_initial_state())


@app.resource("game://frozen_lake/config")
def get_game_config() -> str:
    """MCP Resource: Provides game configuration information."""
    import json

    return json.dumps(
        {
            "game_type": "FrozenLake",
            "version": "v1",
            "grid_size": "4x4",
            "deterministic": True,
            "holes": [5, 7, 11, 12],
            "goal": 15,
            "actions": ["LEFT", "DOWN", "RIGHT", "UP"],
        }
    )


@app.tool(
    name="lake_move",
    description="Move in the FrozenLake game. Actions: LEFT, DOWN, RIGHT, UP",
)
def lake_move(action: str, ctx: Context) -> Dict[str, Any]:
    """
    Move in the FrozenLake game.

    This operates on the global game state. Production servers are typically
    stateless shims, so we maintain minimal global state rather than
    per-session state.

    Args:
        action: Direction to move (LEFT, DOWN, RIGHT, UP)

    Returns:
        Game state with position, reward, and completion status
    """
    global GAME_ENV, CURRENT_POSITION, TOTAL_MOVES

    # Explicit parameter validation to catch empty/missing action
    if not action or not isinstance(action, str):
        raise ValueError(
            f"Invalid action parameter: '{action}'. "
            f"Must be a non-empty string. Valid actions: LEFT, DOWN, RIGHT, UP"
        )

    action = action.strip().upper()  # Normalize the action

    # Extract seed from client info and reinitialize if needed (on first call)
    extract_seed_from_context(ctx)

    # Validate action
    valid_actions = ["LEFT", "DOWN", "RIGHT", "UP"]
    if action not in valid_actions:
        raise ValueError(f"Invalid action '{action}'. Use: {', '.join(valid_actions)}")

    # Convert action to game command
    action_map = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}
    action_int = action_map[action]

    # Execute move on global game state
    position, reward, terminated, truncated, info = GAME_ENV.step(action_int)

    # Update global state
    CURRENT_POSITION = int(position)
    TOTAL_MOVES += 1

    # Log the result but DON'T auto-reset - let the client handle resets
    if terminated or truncated:
        print(
            f"🎮 Game ended: {'🏆 GOAL!' if reward > 0 else '💀 HOLE!'} (move #{TOTAL_MOVES})"
        )
        # Don't auto-reset - return the final state first
    else:
        print(f"🎮 {action} → position {CURRENT_POSITION} (move #{TOTAL_MOVES})")

    # Return complete state including DYNAMIC grid layout showing player position
    # Get the base grid layout from the environment
    grid_layout = ""
    if hasattr(GAME_ENV, "desc"):
        # Create dynamic grid showing current player position
        grid_with_player = []
        for i, row in enumerate(GAME_ENV.desc):
            row_chars = []
            for j, cell in enumerate(row):
                pos = i * 4 + j  # Convert 2D to 1D position
                cell_char = (
                    cell.decode("utf-8") if isinstance(cell, bytes) else str(cell)
                )

                if pos == CURRENT_POSITION and not (terminated or truncated):
                    # Show player position with 'P' if game is ongoing
                    row_chars.append("P")
                elif pos == CURRENT_POSITION and terminated and reward > 0:
                    # Show goal reached with 'W' (Won)
                    row_chars.append("W")
                elif pos == CURRENT_POSITION and terminated and reward == 0:
                    # Show hole with 'X' (failed)
                    row_chars.append("X")
                else:
                    # Show original cell
                    row_chars.append(cell_char)
            grid_with_player.append("".join(row_chars))
        grid_layout = "\n".join(grid_with_player)

    return {
        "position": CURRENT_POSITION,
        "grid_layout": grid_layout,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "moves": TOTAL_MOVES,
        "info": {
            "grid_size": "4x4",
            "holes": [5, 7, 11, 12],  # Positions of holes in 4x4 grid
            "goal": 15,  # Goal position
            **(info or {}),
        },
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

    args = parser.parse_args()

    # Initialize the game state with default (no seed) - will be reinitialized when client connects
    initialize_game()

    print("🚀 FrozenLake MCP Server Starting...")
    print(f"📡 Transport: {args.transport}")
    if args.transport == "streamable-http":
        print(f"🌐 Port: {args.port}")
    print("🎯 MCP Pattern: Resources for initial state, tools for actions")
    print("🔗 Initial state resource: game://frozen_lake/initial_state")
    print("🎮 Action tool: lake_move")
    print("🌱 Supports seeds via client info")
    print()

    # Set environment variable for HTTP port
    if args.transport == "streamable-http":
        os.environ["PORT"] = str(args.port)

    # Run the server
    app.run(transport=args.transport)


if __name__ == "__main__":
    main()
