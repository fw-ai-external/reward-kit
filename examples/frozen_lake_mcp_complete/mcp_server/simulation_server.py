"""
FrozenLake Simulation MCP Server

This simulation server implements FrozenLake using the proper MCP low-level API
with StreamableHTTP transport following the MCP specification.

This is the correct implementation pattern for MCP servers that need to be
deployed on Cloud Run or other HTTP-based deployments.

Usage:
    python simulation_server.py --port 8000 --host 0.0.0.0
"""

import argparse
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any, Dict, Optional, Tuple

import click
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

# Import production server components
from examples.frozen_lake_mcp_complete.mcp_server.frozen_lake_adapter import (
    FrozenLakeAdapter,
)

# Configure logging
logger = logging.getLogger(__name__)


class InMemoryEventStore:
    """Simple in-memory event store for demonstration."""

    def __init__(self):
        self.events = {}

    async def store_event(self, stream_id: str, message: Any) -> str:
        event_id = str(time.time())
        self.events[event_id] = {"stream_id": stream_id, "message": message}
        return event_id

    async def replay_events_after(
        self, last_event_id: str, send_callback: Any
    ) -> Optional[str]:
        # Simple implementation - just return None for now
        return None


class FrozenLakeSimulationServer(FrozenLakeAdapter):
    """FrozenLake simulation server using proper MCP low-level API."""

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.app = Server("FrozenLake-Simulation")
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up MCP tool and resource handlers."""

        @self.app.list_tools()
        async def list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name="lake_move",
                    description="Move in the FrozenLake game. Actions: LEFT, DOWN, RIGHT, UP",
                    inputSchema={
                        "type": "object",
                        "required": ["action"],
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "Direction to move",
                                "enum": ["LEFT", "DOWN", "RIGHT", "UP"],
                            }
                        },
                    },
                )
            ]

        @self.app.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            if name == "lake_move":
                result = await self._handle_lake_move(arguments)
                return [types.TextContent(type="text", text=json.dumps(result))]
            else:
                raise ValueError(f"Unknown tool: {name}")

        @self.app.list_resources()
        async def list_resources() -> list[types.Resource]:
            return [
                types.Resource(
                    uri="game://frozen_lake/initial_state",
                    name="Initial Game State",
                    description="Provides initial game state for simulation",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="game://frozen_lake/config",
                    name="Game Configuration",
                    description="Provides game configuration information",
                    mimeType="application/json",
                ),
            ]

        @self.app.read_resource()
        async def read_resource(uri: str) -> str:
            # Convert AnyUrl to string for comparison
            uri_str = str(uri)

            if uri_str == "game://frozen_lake/initial_state":
                return await self._get_initial_state_resource()
            elif uri_str == "game://frozen_lake/config":
                return await self._get_game_config_resource()
            else:
                raise ValueError(f"Unknown resource: {uri_str}")

    async def _handle_lake_move(self, arguments: dict) -> Dict[str, Any]:
        """Handle lake_move tool call."""
        action = arguments.get("action")
        if not action:
            raise ValueError("Missing required parameter: action")

        ctx = self.app.request_context
        session_id = f"sim_{id(ctx.session)}"

        # Get or create session
        session_state = await self._get_or_create_session(session_id)

        # Validate action
        action = action.strip().upper()
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
        session_state["total_reward"] += reward
        session_state["last_used"] = time.time()

        # Format response
        result = {
            "position": int(self.format_observation(obs)),
            "grid_layout": self._get_grid_layout(int(obs), env),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "moves": int(session_state["steps"]),
            "total_reward": float(session_state["total_reward"]),
            "info": info or {},
        }

        # Simulation logging
        if terminated:
            status = "🏆 WON!" if reward > 0 else "💀 LOST!"
            logger.info(
                f"🎮 {session_id[:12]}... {action} → {obs} | {status} (simulation)"
            )
        else:
            logger.info(
                f"🎮 {session_id[:12]}... {action} → {obs} | Move #{session_state['steps']}"
            )

        return result

    async def _get_initial_state_resource(self) -> str:
        """Get initial state resource."""
        # Create a temporary session for resource calls
        temp_session_id = "resource_call"
        session_state = await self._get_or_create_session(temp_session_id)

        env = session_state["env"]
        initial_observation = session_state["initial_observation"]

        initial_state = {
            "position": int(initial_observation),
            "grid_layout": self._get_grid_layout(int(initial_observation), env),
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

    async def _get_game_config_resource(self) -> str:
        """Get game configuration resource."""
        config = {
            "game_type": "FrozenLake",
            "version": "simulation-v1",
            "grid_size": "4x4",
            "deterministic": True,
            "actions": ["LEFT", "DOWN", "RIGHT", "UP"],
        }
        return json.dumps(config)

    def _get_grid_layout(self, position: int, env=None) -> str:
        """Get the grid layout showing current player position."""
        if env is None:
            return f"Position: {position}"

        if not hasattr(env, "desc") or env.desc is None:
            return f"Position: {position} (no map available)"

        # Get the actual map from the environment
        desc = env.desc
        size = len(desc)

        # Convert position to row, col coordinates
        row = position // size
        col = position % size

        # Create grid representation
        grid_lines = []
        for r, desc_row in enumerate(desc):
            line = ""
            for c, cell in enumerate(desc_row):
                cell_char = (
                    cell.decode("utf-8") if isinstance(cell, bytes) else str(cell)
                )

                if r == row and c == col:
                    if cell_char == "H":
                        line += "X"  # Player fell in hole
                    elif cell_char == "G":
                        line += "W"  # Player reached goal
                    elif cell_char == "S":
                        line += "S"  # Player at start
                    else:
                        line += "P"  # Player on frozen tile
                else:
                    line += cell_char

            grid_lines.append(line)

        return "\n".join(grid_lines)

    async def _get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """Get or create session state."""
        if session_id not in self.sessions:
            # Create environment with default config
            config = self.get_default_config()
            seed = 42  # Default seed for reproducibility

            # Create environment with seed
            if hasattr(self, "create_environment_with_seed"):
                env, obs, info = self.create_environment_with_seed(config, seed=seed)
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
            logger.info(
                f"🆕 Simulation session created: {session_id[:16]}... (seed={seed})"
            )

        self.sessions[session_id]["last_used"] = time.time()
        return self.sessions[session_id]


@click.command()
@click.option("--port", default=8000, help="Port to listen on for HTTP")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option(
    "--log-level",
    default="INFO",
    help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
def main(port: int, host: str, log_level: str) -> int:
    """Main entry point for FrozenLake simulation server."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print(f"🚀 Starting FrozenLake Simulation Server")
    print(f"📡 Transport: streamable-http")
    print(f"🌐 Host: {host}")
    print(f"🌐 Port: {port}")
    print(f"🎯 Independent simulation (proper MCP implementation)")
    print()

    # Create server instance
    server_instance = FrozenLakeSimulationServer()
    app = server_instance.app

    # Create event store for resumability
    event_store = InMemoryEventStore()

    # Create session manager
    session_manager = StreamableHTTPSessionManager(
        app=app,
        event_store=event_store,
        json_response=False,
    )

    # ASGI handler for streamable HTTP connections
    async def handle_streamable_http(
        scope: Scope, receive: Receive, send: Send
    ) -> None:
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        """Context manager for managing session manager lifecycle."""
        async with session_manager.run():
            logger.info("🚀 FrozenLake simulation server started!")
            try:
                yield
            finally:
                logger.info("🧹 Simulation server shutting down...")

    # Create ASGI application
    starlette_app = Starlette(
        debug=True,
        routes=[
            Mount("/mcp", app=handle_streamable_http),
            Mount("/mcp/", app=handle_streamable_http),
        ],
        lifespan=lifespan,
    )

    # Run with uvicorn
    import uvicorn

    uvicorn.run(starlette_app, host=host, port=port)

    return 0


if __name__ == "__main__":
    main()
