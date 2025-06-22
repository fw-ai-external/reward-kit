"""
MCP Simulation Server Framework

This framework enforces the correct separation between production and simulation servers.
It ensures that:
1. No session management tools are exposed to models
2. Session initialization happens via initializationOptions (MCP spec)
3. Only domain game tools are exposed
4. Simulation logic is handled internally

Usage:
    class MyGameSimulation(SimulationServerBase):
        def create_environment(self, config): ...
        def reset_environment(self, env, seed): ...
        # etc.

    server = MyGameSimulation("MyGame")
    server.run()
"""

import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Optional, Tuple

from mcp.server.fastmcp import Context, FastMCP


class SimulationServerBase(ABC):
    """
    Base class for simulation MCP servers.

    This framework enforces correct separation by:
    - Managing sessions internally (no exposed tools)
    - Using initializationOptions for configuration
    - Only exposing domain-specific game tools
    - Preventing session management tool pollution
    """

    def __init__(self, server_name: str):
        """
        Initialize simulation server framework.

        Args:
            server_name: Name for the MCP server
        """
        self.server_name = server_name
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_lock = threading.Lock()

        # Create FastMCP server with proper lifespan
        self.mcp = FastMCP(server_name, lifespan=self._lifespan)

        # Register only domain tools (no session management tools)
        self._register_domain_tools()

    @asynccontextmanager
    async def _lifespan(self, app: FastMCP):
        """Server lifespan management."""
        print(f"🚀 {self.server_name} Simulation Server")
        print("🎯 Framework: Enforces no session tool pollution")
        print(f"🔧 Domain tools: {self._get_domain_tool_names()}")
        print("📡 Session management: Internal (MCP spec compliant)")
        print()

        yield

        # Cleanup sessions
        print("🧹 Cleaning up simulation sessions...")
        with self.session_lock:
            for session_id, session_data in self.sessions.items():
                env = session_data.get("env")
                if env:
                    try:
                        self.close_environment(env)
                    except Exception as e:
                        print(
                            f"⚠️ Error closing environment in session {session_id}: {e}"
                        )
            self.sessions.clear()
        print("✅ Simulation server shutdown complete")

    def _get_session_id(self, ctx: Context) -> str:
        """Extract session ID from FastMCP Context."""
        session_obj = ctx.session
        return f"sim_{id(session_obj)}"

    def _get_or_create_session(self, ctx: Context) -> Dict[str, Any]:
        """
        Get or create session for current FastMCP context.

        This handles session initialization using MCP spec:
        - Configuration from initializationOptions
        - Automatic environment creation
        - Internal session management (no tools exposed)
        """
        session_id = self._get_session_id(ctx)

        with self.session_lock:
            if session_id not in self.sessions:
                # TODO: Extract from ctx.session.initialization_options when available
                # For now, use defaults - in real implementation this would come from MCP
                config = self.get_default_config()
                seed = None  # Would come from initializationOptions

                # Create environment via abstract methods
                env = self.create_environment(config)
                obs, info = self.reset_environment(env, seed=seed)

                self.sessions[session_id] = {
                    "env": env,
                    "config": config,
                    "seed": seed,
                    "created_at": time.time(),
                    "last_used": time.time(),
                    "initial_observation": self.format_observation(obs),
                    "session_id": session_id,
                    "steps": 0,
                    "total_reward": 0.0,
                }

                print(
                    f"🆕 Simulation session created: {session_id[:16]}... (seed={seed})"
                )

            # Update last used time
            self.sessions[session_id]["last_used"] = time.time()
            return self.sessions[session_id]

    def _register_domain_tools(self):
        """
        Register only domain-specific tools (no session management).

        This is enforced by the framework - subclasses cannot add session tools.
        """
        # Get domain tool definitions from subclass
        domain_tools = self.get_domain_tools()

        for tool_name, tool_func in domain_tools.items():
            # Wrap tool function to inject session management
            def make_wrapped_tool(original_func):
                def wrapped_tool(*args, ctx: Context, **kwargs):
                    # Inject session automatically
                    session_data = self._get_or_create_session(ctx)
                    return original_func(session_data, *args, **kwargs)

                # Preserve original function metadata
                wrapped_tool.__name__ = original_func.__name__
                wrapped_tool.__doc__ = original_func.__doc__
                return wrapped_tool

            # Register wrapped tool with FastMCP
            wrapped = make_wrapped_tool(tool_func)
            self.mcp.tool(name=tool_name)(wrapped)

    # Abstract methods that subclasses MUST implement

    @abstractmethod
    def create_environment(self, config: Dict[str, Any]) -> Any:
        """Create environment instance."""
        pass

    @abstractmethod
    def reset_environment(
        self, env: Any, seed: Optional[int] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset environment to initial state."""
        pass

    @abstractmethod
    def step_environment(
        self, env: Any, action: Any
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute step in environment."""
        pass

    @abstractmethod
    def close_environment(self, env: Any) -> None:
        """Clean up environment resources."""
        pass

    @abstractmethod
    def parse_action(self, action_str: str) -> Any:
        """Parse action string to environment action."""
        pass

    @abstractmethod
    def format_observation(self, observation: Any) -> Any:
        """Format observation for JSON serialization."""
        pass

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get default environment configuration."""
        pass

    @abstractmethod
    def get_domain_tools(self) -> Dict[str, Callable]:
        """
        Get domain-specific tool functions.

        Returns:
            Dict mapping tool names to functions.
            Functions receive (session_data, *args, **kwargs)
        """
        pass

    def _get_domain_tool_names(self) -> list:
        """Get list of domain tool names for logging."""
        return list(self.get_domain_tools().keys())

    def run(self, transport: str = "streamable-http", **kwargs):
        """
        Run the simulation server.

        Args:
            transport: Transport protocol
            **kwargs: Additional arguments for FastMCP.run()
        """
        print(f"📡 Starting simulation server with {transport} transport")
        print(f"🎮 Domain tools: {self._get_domain_tool_names()}")
        print("🚫 No session management tools exposed (framework enforced)")
        print()

        self.mcp.run(transport, **kwargs)
