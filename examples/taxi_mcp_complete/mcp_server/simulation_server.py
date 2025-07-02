"""
Taxi Simulation MCP Server

This simulation server implements Taxi using the proper MCP low-level API
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
from examples.taxi_mcp_complete.mcp_server.taxi_adapter import (
    TaxiAdapter,
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


class TaxiSimulationServer(TaxiAdapter):
    """Taxi simulation server using proper MCP low-level API."""

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.app = Server("Taxi-Simulation")
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up MCP tool and resource handlers."""

        @self.app.list_tools()
        async def list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name="taxi_move",
                    description="Move and act in the Taxi game. Move/Action: SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF",
                    inputSchema={
                        "type": "object",
                        "required": ["action"],
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "Direction to move or action to take",
                                "enum": ["SOUTH", "NORTH", "EAST", "WEST", "PICKUP", "DROPOFF"],
                            }
                        },
                    },
                )
            ]

        @self.app.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            if name == "taxi_move":
                result = await self._handle_taxi_move(arguments)
                return [types.TextContent(type="text", text=json.dumps(result))]
            else:
                raise ValueError(f"Unknown tool: {name}")

        @self.app.list_resources()
        async def list_resources() -> list[types.Resource]:
            return [
                types.Resource(
                    uri="game://taxi/initial_state",
                    name="Initial Game State",
                    description="Provides initial game state for simulation",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="game://taxi/config",
                    name="Game Configuration",
                    description="Provides game configuration information",
                    mimeType="application/json",
                ),
            ]

        @self.app.read_resource()
        async def read_resource(uri: str) -> str:
            # Convert AnyUrl to string for comparison
            uri_str = str(uri)

            if uri_str == "game://taxi/initial_state":
                return await self._get_initial_state_resource()
            elif uri_str == "game://taxi/config":
                return await self._get_game_config_resource()
            else:
                raise ValueError(f"Unknown resource: {uri_str}")

    def _get_grid_layout(self, position: int, env=None) -> str:
        """Get the visual grid layout showing current taxi and passenger positions."""
        if env is None:
            return f"Position: {position}"

        if not hasattr(env, "desc") or env.desc is None:
            return f"Position: {position} (no map available)"

        # Get the actual map from the environment
        desc = env.desc

        decoded = self.decode_state(position)
        taxi_row = decoded["taxi_row"]
        taxi_col = decoded["taxi_col"]
        passenger_location = decoded["passenger_location"]
        destination = decoded["destination"]

        # Convert logical 5x5 coordinates to visual 7x11 coordinates
        # Mapping: visual_row = logical_row + 1, visual_col = logical_col * 2 + 1
        taxi_visual_row = taxi_row + 1
        taxi_visual_col = taxi_col * 2 + 1

        # Create grid representation
        grid_lines = []
        for r, desc_row in enumerate(desc):
            line = ""
            for c, cell in enumerate(desc_row):
                cell_char = cell.decode("utf-8") if isinstance(cell, bytes) else str(cell)

                # Show taxi position
                if r == taxi_visual_row and c == taxi_visual_col:
                    if passenger_location == 4:  # Passenger in taxi
                        line += "T"  # Taxi with passenger
                    else:
                        line += "t"  # Empty taxi
                else:
                    # Check if this position is the destination
                    if destination < 4:  # Valid destination (0-3)
                        dest_locs = [(0, 0), (0, 4), (4, 0), (4, 3)]  # R, G, Y, B locations
                        dest_logical_row, dest_logical_col = dest_locs[destination]
                        dest_visual_row = dest_logical_row + 1
                        dest_visual_col = dest_logical_col * 2 + 1
                        
                        if r == dest_visual_row and c == dest_visual_col:
                            # Highlight destination
                            if cell_char in "RGYB":
                                line += cell_char.lower()  # Lowercase for destination
                            else:
                                line += "D"  # Destination marker
                        else:
                            line += cell_char
                    else:
                        line += cell_char

            grid_lines.append(line)

        return "\n".join(grid_lines)
        

    async def _handle_taxi_move(self, arguments: dict) -> Dict[str, Any]:
        """Handle taxi_move tool call."""
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
        # Convert numpy arrays in info to lists for JSON serialization
        clean_info = {}
        if info:
            for key, value in info.items():
                if hasattr(value, 'tolist'):  # numpy array
                    clean_info[key] = value.tolist()
                else:
                    clean_info[key] = value

        # Convert action mask to human-readable legal actions
        legal_actions = []
        blocked_actions = []
        if info and "action_mask" in info:
            action_names = ["SOUTH", "NORTH", "EAST", "WEST", "PICKUP", "DROPOFF"]
            for i, allowed in enumerate(info["action_mask"]):
                if allowed:
                    legal_actions.append(action_names[i])
                else:
                    blocked_actions.append(action_names[i])

        result = {
            "state": int(self.format_observation(obs)),
            "grid_layout": self._get_grid_layout(int(obs), env),
            "state_description": self.get_state_description(int(obs)),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "moves": int(session_state["steps"]),
            "total_reward": float(session_state["total_reward"]),
            "legal_actions": legal_actions,
            "blocked_actions": blocked_actions,
            "info": clean_info,
        }

        # Simulation logging
        if terminated:
            status = "🎉 SUCCESS!" if reward > 0 else "💀 FAILED!"
            logger.info(
                f"🚕 {session_id[:12]}... {action} → {obs} | {status} (simulation)"
            )
        else:
            logger.info(
                f"🚕 {session_id[:12]}... {action} → {obs} | Move #{session_state['steps']}"
            )

        return result

    async def _get_initial_state_resource(self) -> str:
        """Get initial state resource."""
        # Use the current request's session context instead of a shared temporary session
        ctx = self.app.request_context
        session_id = f"resource_{id(ctx.session)}"
        session_state = await self._get_or_create_session(session_id)

        env = session_state["env"]
        initial_observation = session_state["initial_observation"]

        initial_state = {
            "state": int(initial_observation),
            "grid_layout": self._get_grid_layout(int(initial_observation), env),
            "state_description": self.get_state_description(int(initial_observation)),
            "moves": 0,
            "terminated": False,
            "truncated": False,
            "reward": 0.0,
            "info": {
                "grid_size": "5x5",
                "initial_position": int(initial_observation),
                "seed": session_state.get("seed", 42),
            },
        }
        return json.dumps(initial_state)

    async def _get_game_config_resource(self) -> str:
        """Get game configuration resource."""
        # Get session to extract actual config
        ctx = self.app.request_context
        session_id = f"config_{id(ctx.session)}"
        session_state = await self._get_or_create_session(session_id)

        config = {
            "game_type": "Taxi",
            "version": "simulation-v1",
            "grid_size": "5x5",
            "deterministic": True,
            "actions": ["SOUTH", "NORTH", "EAST", "WEST", "PICKUP", "DROPOFF"],
        }
        return json.dumps(config)

    async def _get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """Get or create session state."""
        if session_id not in self.sessions:
            # Create environment with default config
            config = self.get_default_config()
            seed = 42  # Default seed for reproducibility

            # Extract seed and config from client_info if available
            ctx = self.app.request_context
            if ctx.session.client_params and ctx.session.client_params.clientInfo:
                client_info = ctx.session.client_params.clientInfo
                if hasattr(client_info, "_extra") and client_info._extra:
                    if "seed" in client_info._extra:
                        seed = client_info._extra["seed"]
                        logger.info(f"🎲 Using seed from client_info: {seed}")
                    else:
                        seed = None
                        logger.info(
                            "🎲 No seed found in client_info, using random seed"
                        )

                    # Extract taxi config from client_info
                    if "is_raining" in client_info._extra:
                        config["is_raining"] = client_info._extra["is_raining"]
                        logger.info(
                            f"🌧️ Using is_raining from client_info: {config['is_raining']}"
                        )

                    if "fickle_passenger" in client_info._extra:
                        config["fickle_passenger"] = client_info._extra["fickle_passenger"]
                        logger.info(
                            f"🎭 Using fickle_passenger from client_info: {config['fickle_passenger']}"
                        )
                else:
                    seed = None
                    logger.info("🎲 No extra info in client_info, using random seed")
            else:
                seed = None
                logger.info("🎲 No client_info available, using random seed")

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
                f"🆕 Simulation session created: {session_id[:16]}... (seed={seed}, config={config})"
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
    """Main entry point for Taxi simulation server."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print(f"🚀 Starting Taxi Simulation Server")
    print(f"📡 Transport: streamable-http")
    print(f"🌐 Host: {host}")
    print(f"🌐 Port: {port}")
    print(f"🚕 Independent simulation (proper MCP implementation)")
    print()

    # Create server instance
    server_instance = TaxiSimulationServer()
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
            logger.info("🚀 Taxi simulation server started!")
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
    import sys

    if len(sys.argv) == 1:
        # No arguments provided, use defaults for direct execution
        try:
            main.callback(port=8000, host="0.0.0.0", log_level="INFO")
        except SystemExit:
            pass  # Handle Click's normal exit
    else:
        # Arguments provided, let Click handle parsing
        try:
            main()
        except SystemExit:
            pass  # Handle Click's normal exit