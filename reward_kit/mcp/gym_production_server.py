"""
GymProductionServer Framework

This framework provides a simplified base class for creating production MCP servers
that wrap gymnasium environments using adapters. It handles:

1. Single-session production server lifecycle
2. Automatic tool and resource registration
3. Environment management via adapters
4. MCP resource patterns for initial state
5. Standardized tool signatures

Usage:
    class MyGameProdServer(GymProductionServer):
        def __init__(self):
            super().__init__("MyGame-v1", MyAdapter())

        def _register_tools(self):
            # Register domain-specific tools

        @staticmethod
        def format_observation(obs, env):
            # Format observations for MCP responses
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from mcp.server.fastmcp import Context, FastMCP

from .adapter import EnvironmentAdapter


class GymProductionServer(ABC):
    """
    Single-session, production MCP server base class.

    Subclasses supply:
    • adapter - EnvironmentAdapter instance
    • _register_tools() - add ergonomic tools
    • format_observation(obs, env) - env-specific view dict
    """

    def __init__(self, name: str, adapter: EnvironmentAdapter):
        """
        Initialize production server.

        Args:
            name: Server name for MCP
            adapter: Environment adapter instance
        """
        self.adapter = adapter
        self.env, self.obs, _info = self._new_env()

        # Create FastMCP server
        self.mcp = FastMCP(
            name,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8000)),
        )

        # Register resources and tools
        self._register_resources()
        self._register_tools()

    def _new_env(self, seed: Optional[int] = None) -> Tuple[Any, Any, Dict]:
        """Create new environment and return initial state."""
        if hasattr(self.adapter, "create_environment_with_seed"):
            env, obs, info = self.adapter.create_environment_with_seed(
                self.adapter.get_default_config(), seed=seed
            )
        else:
            env = self.adapter.create_environment(self.adapter.get_default_config())
            obs, info = self.adapter.reset_environment(env, seed=seed)
        return env, obs, info

    def _render(self, obs) -> Dict[str, Any]:
        """Format observation using subclass implementation."""
        return self.format_observation(obs, self.env)

    def _register_resources(self):
        """Register standard MCP resources."""

        @self.mcp.resource("game://initial_state")
        def initial_state() -> str:
            """Provide initial game state as MCP resource."""
            import json

            return json.dumps(self._render(self.obs))

    def extract_seed_from_context(self, ctx: Context) -> Optional[int]:
        """Extract seed from MCP client info if available."""
        if hasattr(ctx, "session") and hasattr(ctx.session, "client_info"):
            client_info = ctx.session.client_info
            if client_info and hasattr(client_info, "_extra"):
                extra_data = client_info._extra
                if extra_data and "seed" in extra_data:
                    seed = extra_data["seed"]
                    print(f"🌱 Reinitializing with seed from client: {seed}")
                    self.env, self.obs, _info = self._new_env(seed=seed)
                    return seed
        return None

    # Abstract methods that subclasses must implement

    @abstractmethod
    def _register_tools(self):
        """Register domain-specific MCP tools."""
        pass

    @staticmethod
    @abstractmethod
    def format_observation(obs: Any, env: Any) -> Dict[str, Any]:
        """Format observation for MCP response."""
        pass

    def run(self, transport: str = "streamable-http", **kwargs):
        """Run the production server."""
        print(f"🚀 {self.mcp.name} Production Server Starting...")
        print(f"📡 Transport: {transport}")
        print("🎯 MCP Pattern: Resources for initial state, tools for actions")
        print("🔗 Initial state resource: game://initial_state")

        # Run the server
        self.mcp.run(transport=transport, **kwargs)
