"""
FrozenLake MCP Server - Production Implementation

This is a stateless production MCP server that provides FrozenLake gameplay.
Like most production MCP servers (Google Docs, Shopify, etc.), this is a
stateless shim that maintains minimal state.

There is only one global game session - each client interaction operates
on the same game state. Session management is handled by simulation servers
that wrap this production server.

Dependencies: FastMCP + Gymnasium only
Deploy: Docker container, Cloud Run, or any MCP-compatible platform

Usage:
    python frozen_lake_server.py --transport sse
    python frozen_lake_server.py --transport streamable-http --port 8000
"""

import argparse
from contextlib import asynccontextmanager
from typing import Any, Dict

from gymnasium.envs.toy_text import FrozenLakeEnv
from mcp.server.fastmcp import Context, FastMCP

# Global game state - single session for production simplicity
GAME_ENV: FrozenLakeEnv = None
CURRENT_POSITION: int = 0
TOTAL_MOVES: int = 0


@asynccontextmanager
async def lifespan(app: FastMCP):
    """Production server lifespan management."""
    global GAME_ENV, CURRENT_POSITION, TOTAL_MOVES

    print("🚀 FrozenLake MCP Server v1.0 - Production")
    print("🎮 Game: 4x4 FrozenLake (deterministic)")
    print("🔧 Tool: lake_move")
    print("📡 Stateless production server (single global session)")
    print()

    # Initialize global game state
    GAME_ENV = FrozenLakeEnv(
        map_name="4x4",
        is_slippery=False,  # Deterministic for production
        render_mode=None,
    )
    CURRENT_POSITION, _ = GAME_ENV.reset()
    TOTAL_MOVES = 0

    print(f"🎯 Game initialized at position {CURRENT_POSITION}")

    yield

    # Production cleanup
    print("🧹 Shutting down production server...")
    if GAME_ENV and hasattr(GAME_ENV, "close"):
        GAME_ENV.close()
    print("✅ FrozenLake production server stopped")


# Create production FastMCP server
app = FastMCP("FrozenLake-v1", lifespan=lifespan)


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

    # If game ended, reset for next interaction
    if terminated or truncated:
        print(
            f"🎮 Game ended: {'🏆 GOAL!' if reward > 0 else '💀 HOLE!'} (move #{TOTAL_MOVES})"
        )
        # Reset game for next interaction
        CURRENT_POSITION, _ = GAME_ENV.reset()
        TOTAL_MOVES = 0
        print(f"🔄 Game reset to position {CURRENT_POSITION}")
    else:
        print(f"🎮 {action} → position {CURRENT_POSITION} (move #{TOTAL_MOVES})")

    # Return current state
    return {
        "position": CURRENT_POSITION,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "moves": TOTAL_MOVES,
        "info": info or {},
    }


def main():
    """Production server entry point."""
    parser = argparse.ArgumentParser(description="FrozenLake MCP Server v1.0")
    parser.add_argument(
        "--transport",
        default="sse",
        choices=["stdio", "sse", "streamable-http"],
        help="MCP transport protocol",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for HTTP transports"
    )
    args = parser.parse_args()

    print(f"🚀 Starting FrozenLake Production MCP Server")
    print(f"📡 Transport: {args.transport}")
    if args.transport in ["sse", "streamable-http"]:
        print(f"🌐 Port: {args.port}")
    print(f"🎯 Architecture: Stateless production server (like Google Docs MCP)")
    print(f"🔧 Session management: Handled by simulation wrappers")
    print()

    # Start production server
    if args.transport in ["sse", "streamable-http"]:
        app.run(transport=args.transport, port=args.port)
    else:
        app.run(transport=args.transport)


if __name__ == "__main__":
    main()
