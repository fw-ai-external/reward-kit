#!/usr/bin/env python3
"""
Managed Simulation Server for FrozenLake MCP

This server acts as a "meta-server" that manages a pool of production server instances
in isolated Conda environments. It provides simplified session management by automatically
handling server lifecycle and proxying requests to the appropriate instances.

Features:
- Automatic server pool management using CondaServerProcessManager
- Session-based seed handling and server allocation
- Request proxying to isolated production server instances
- Automatic cleanup of server instances and environments
- Zero game logic duplication (delegates to production servers)

Architecture:
- Client requests → Managed Server → Production Server Instance (in Conda env)
- Each session gets its own isolated server instance with a specific seed
- All game logic is handled by the production servers
- This server only handles session management and request proxying
"""

import argparse
import asyncio
import json
import logging
import os
import threading
import time
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Dict, Optional

import uvicorn
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import Implementation, Resource, TextContent, Tool
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

from reward_kit.mcp.simple_process_manager import SimpleServerProcessManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Also configure a handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class ManagedSimulationServer:
    """
    Managed simulation server that proxies requests to production server instances.

    This server manages a pool of production server instances running in isolated
    Conda environments, handling session management and request proxying.
    """

    def __init__(
        self,
        server_name: str = "FrozenLake-Managed-Simulation",
        production_script_path: str = None,
        requirements_path: str = None,
        conda_base_env: str = "base",
        use_conda_isolation: bool = False,
    ):
        """
        Initialize the managed simulation server.

        Args:
            server_name: Name for the MCP server
            production_script_path: Path to the production server script
            requirements_path: Path to requirements.txt for server environments
            conda_base_env: Base conda environment to clone from
            use_conda_isolation: Whether to use conda environments for true isolation
        """
        self.server_name = server_name
        self.production_script_path = (
            production_script_path or self._get_default_script_path()
        )
        self.requirements_path = (
            requirements_path or self._get_default_requirements_path()
        )

        # Create process manager for handling server instances
        if use_conda_isolation:
            from reward_kit.mcp.process_manager import CondaServerProcessManager

            logger.info(
                "Using CondaServerProcessManager for full environment isolation"
            )
            self.process_manager = CondaServerProcessManager(
                script_path=self.production_script_path,
                requirements_path=self.requirements_path,
                conda_base_env=conda_base_env,
            )
        else:
            logger.info("Using SimpleServerProcessManager for lightweight testing")
            self.process_manager = SimpleServerProcessManager(
                script_path=self.production_script_path,
            )

        # Create low-level MCP server
        self.app = Server(server_name)

        # Session management
        self.session_servers: Dict[str, int] = {}  # session_id -> port
        self.session_lock = threading.Lock()

        self._register_tools()
        self._register_resources()
        self._register_session_handlers()

    def _get_default_script_path(self) -> str:
        """Get default path to the production server script."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "frozen_lake_mcp_server.py")

    def _get_default_requirements_path(self) -> str:
        """Get default path to requirements.txt."""
        # Use the parent directory's requirements file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        return os.path.join(parent_dir, "requirements.txt")

    def _get_session_id_from_context(self, ctx) -> str:
        """Extract session ID from MCP request context."""
        if hasattr(ctx, "session") and hasattr(ctx.session, "client_params"):
            client_params = ctx.session.client_params
            if hasattr(client_params, "clientInfo"):
                client_info = client_params.clientInfo
                if client_info and hasattr(client_info, "_extra"):
                    extra_data = client_info._extra
                    if extra_data and isinstance(extra_data, dict):
                        # Create a stable session ID based on seed and other config
                        import hashlib

                        stable_data = {
                            "seed": extra_data.get("seed"),
                            "config": extra_data.get("config", {}),
                            "name": client_info.name,
                            "version": client_info.version,
                        }
                        stable_str = json.dumps(stable_data, sort_keys=True)
                        session_id = hashlib.md5(stable_str.encode()).hexdigest()
                        logger.debug(f"Generated stable session_id: {session_id}")
                        return session_id

        # Fallback session ID
        session_id = f"managed_sim_{id(ctx)}"
        logger.debug(f"Generated fallback session_id: {session_id}")
        return session_id

    async def _get_or_create_server_instance(self, ctx) -> int:
        """Get or create a server instance for this session. Returns the port."""
        session_id = self._get_session_id_from_context(ctx)

        with self.session_lock:
            if session_id not in self.session_servers:
                # Extract seed from client info
                seed = 42  # Default seed

                if hasattr(ctx, "session") and hasattr(ctx.session, "client_params"):
                    client_params = ctx.session.client_params
                    if hasattr(client_params, "clientInfo"):
                        client_info = client_params.clientInfo
                        if client_info and hasattr(client_info, "_extra"):
                            extra_data = client_info._extra
                            if extra_data and isinstance(extra_data, dict):
                                seed = extra_data.get("seed", seed)

                logger.info(
                    f"Creating new server instance for session {session_id} with seed {seed}"
                )

                # Start a new server instance with the extracted seed
                logger.info(f"Starting server instance via process manager...")
                port = self.process_manager.start_server(seed=seed)
                logger.info(f"Process manager returned port {port}")
                self.session_servers[session_id] = port

                logger.info(
                    f"Server instance created on port {port} for session {session_id}"
                )

        return self.session_servers[session_id]

    async def _proxy_tool_call(
        self, session_id: str, tool_name: str, arguments: dict
    ) -> str:
        """Proxy a tool call to the appropriate server instance using fresh MCP client."""
        port = self.session_servers.get(session_id)
        if not port:
            raise ValueError(f"No server instance found for session {session_id}")

        server_url = f"http://localhost:{port}/mcp/"
        logger.debug(f"Creating fresh MCP client for tool call to {server_url}")

        # Create fresh MCP client for this request
        try:
            async with AsyncExitStack() as exit_stack:
                read_stream, write_stream, _ = await exit_stack.enter_async_context(
                    streamablehttp_client(server_url, terminate_on_close=True)
                )

                client_info = Implementation(
                    name="managed-simulation-server", version="1.0.0"
                )
                mcp_client = await exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream, client_info=client_info)
                )
                await mcp_client.initialize()

                # Use proper MCP client to call the tool
                result = await mcp_client.call_tool(tool_name, arguments)

                # Extract the text content from the MCP response
            if result.content and len(result.content) > 0:
                first_content = result.content[0]
                if hasattr(first_content, "text"):
                    return first_content.text

            return "{}"
        except Exception:
            return "{}"
        return "{}"  # This should never be reached, but added for mypy

    async def _proxy_resource_request(self, session_id: str, resource_uri: str) -> str:
        """Proxy a resource request to the appropriate server instance using fresh MCP client."""
        port = self.session_servers.get(session_id)
        if not port:
            raise ValueError(f"No server instance found for session {session_id}")

        server_url = f"http://localhost:{port}/mcp/"
        logger.debug(f"Creating fresh MCP client for resource request to {server_url}")

        # Create fresh MCP client for this request
        try:
            async with AsyncExitStack() as exit_stack:
                read_stream, write_stream, _ = await exit_stack.enter_async_context(
                    streamablehttp_client(server_url, terminate_on_close=True)
                )

                client_info = Implementation(
                    name="managed-simulation-server", version="1.0.0"
                )
                mcp_client = await exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream, client_info=client_info)
                )
                await mcp_client.initialize()

                # Use proper MCP client to read the resource
                result = await mcp_client.read_resource(resource_uri)

                # Return the resource content directly
                if hasattr(result, "contents") and result.contents:
                    first_content = result.contents[0]
                    if hasattr(first_content, "text"):
                        return first_content.text

                return "{}"
        except Exception:
            return "{}"
        return "{}"  # This should never be reached, but added for mypy

    def _register_tools(self) -> None:
        """Register MCP tools that will be proxied to server instances."""

        @self.app.call_tool()
        async def call_tool(name: str, arguments: dict):
            """Handle tool calls by proxying to the appropriate server instance."""
            ctx = self.app.request_context
            session_id = self._get_session_id_from_context(ctx)

            # Ensure server instance exists for this session
            await self._get_or_create_server_instance(ctx)

            # Proxy the tool call
            result_text = await self._proxy_tool_call(session_id, name, arguments)

            return [TextContent(type="text", text=result_text)]

        @self.app.list_tools()
        async def list_tools():
            """List available tools (static list based on production server)."""
            return [
                Tool(
                    name="lake_move",
                    description="Move on the frozen lake. Actions: LEFT, DOWN, RIGHT, UP",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "Direction to move (LEFT, DOWN, RIGHT, UP)",
                            }
                        },
                        "required": ["action"],
                    },
                )
            ]

    def _register_resources(self) -> None:
        """Register MCP resources that will be proxied to server instances."""

        @self.app.read_resource()
        async def read_resource(uri: str):
            """Handle resource requests by proxying to the appropriate server instance."""
            ctx = self.app.request_context
            session_id = self._get_session_id_from_context(ctx)

            # Ensure server instance exists for this session
            await self._get_or_create_server_instance(ctx)

            # Proxy the resource request
            result_text = await self._proxy_resource_request(session_id, uri)

            return result_text

        @self.app.list_resources()
        async def list_resources():
            """List available resources (static list based on production server)."""
            return [
                Resource(
                    uri="game://initial_state",
                    name="initial_state",
                    description="Initial state of the FrozenLake game",
                    mimeType="application/json",
                )
            ]

    def _register_session_handlers(self) -> None:
        """Register session management handlers."""

        @self.app.set_logging_level()
        async def set_logging_level(level):
            """Handle logging level requests."""
            logger.setLevel(getattr(logging, level.upper()))
            return {}

    async def cleanup_session(self, session_id: str) -> None:
        """Clean up a session and its associated server instance."""
        with self.session_lock:
            if session_id in self.session_servers:
                port = self.session_servers[session_id]
                logger.info(f"Cleaning up session {session_id} (port {port})")

                # Stop the server instance and clean up its environment
                self.process_manager.stop_server(port)

                # Remove from session tracking
                del self.session_servers[session_id]

                logger.info(f"Session {session_id} cleaned up successfully")

    async def cleanup_all_sessions(self) -> None:
        """Clean up all sessions and server instances."""
        logger.info("Cleaning up all sessions...")

        with self.session_lock:
            session_ids = list(self.session_servers.keys())

        for session_id in session_ids:
            await self.cleanup_session(session_id)

        logger.info("All sessions cleaned up")

    def run(self, port: int = 8002, host: str = "127.0.0.1", **kwargs) -> None:
        """Run the managed simulation server."""
        print(f"🚀 Starting Managed Simulation Server: {self.server_name}")
        print(f"📦 Production script: {self.production_script_path}")
        print(f"📋 Requirements: {self.requirements_path}")
        print(f"🌐 Host: {host}:{port}")
        print(
            "🔧 Server instances will be created on-demand with session-specific seeds"
        )
        print()

        # Create session manager
        session_manager = StreamableHTTPSessionManager(app=self.app)

        # ASGI handler for streamable HTTP connections
        async def handle_streamable_http(
            scope: Scope, receive: Receive, send: Send
        ) -> None:
            await session_manager.handle_request(scope, receive, send)

        @asynccontextmanager
        async def lifespan(app: Starlette):
            """Manage server lifecycle."""
            async with session_manager.run():
                logger.info(f"🚀 {self.server_name} started!")
                try:
                    yield
                finally:
                    logger.info("🧹 Managed simulation server shutting down...")
                    await self.cleanup_all_sessions()
                    logger.info("✅ Managed simulation server shutdown complete")

        # Create ASGI application
        starlette_app = Starlette(
            debug=kwargs.get("debug", False),
            routes=[
                Mount("/mcp", app=handle_streamable_http),
            ],
            lifespan=lifespan,
        )

        # Run the server
        uvicorn.run(
            starlette_app,
            host=host,
            port=port,
            log_level=kwargs.get("log_level", "info"),
            **{k: v for k, v in kwargs.items() if k not in ["debug", "log_level"]},
        )


def main() -> None:
    """Main entry point for the managed simulation server."""
    parser = argparse.ArgumentParser(description="Managed FrozenLake Simulation Server")
    parser.add_argument("--port", type=int, default=8002, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument(
        "--production-script", default=None, help="Path to the production server script"
    )
    parser.add_argument(
        "--requirements",
        default=None,
        help="Path to requirements.txt for server environments",
    )
    parser.add_argument(
        "--conda-base-env", default="base", help="Base conda environment to clone from"
    )
    parser.add_argument(
        "--use-conda-isolation",
        action="store_true",
        help="Use conda environments for full isolation (slower startup, better isolation)",
    )

    args = parser.parse_args()

    # Create and run server
    server = ManagedSimulationServer(
        production_script_path=args.production_script,
        requirements_path=args.requirements,
        conda_base_env=args.conda_base_env,
        use_conda_isolation=args.use_conda_isolation,
    )
    server.run(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
