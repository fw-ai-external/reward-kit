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

import functools
import inspect
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

from mcp.server.fastmcp import Context, FastMCP


class ToolMismatchError(Exception):
    """Raised when simulation and production tools do not match."""

    pass


class SignatureMismatchError(Exception):
    """Raised when a tool's signature does not match the production version."""

    pass


def simulation_tool(func: Callable) -> Callable:
    """
    Decorator to mark methods as simulation tools.
    These tools will be exposed to the MCP client and validated against production.
    """
    func._is_simulation_tool = True
    return func


def simulation_resource(uri_pattern: str) -> Callable:
    """
    Decorator to mark methods as simulation resources.
    These resources will be exposed to the MCP client for initial state.

    Args:
        uri_pattern: URI pattern for the resource (e.g., "game://frozen_lake/initial_state")
    """

    def decorator(func: Callable) -> Callable:
        func._is_resource = True
        func._resource_uri = uri_pattern
        return func

    return decorator


class SimulationServerBase(ABC):
    """
    Base class for simulation MCP servers.

    This framework enforces correct separation by:
    - Managing sessions internally (no exposed tools)
    - Using initializationOptions for configuration
    - Only exposing domain-specific game tools
    - Preventing session management tool pollution
    - Automatically validating simulation tools against a production server
    - Supporting MCP resources for initial state following proper MCP patterns
    """

    def __init__(
        self,
        server_name: str,
        production_server_app: Optional[FastMCP] = None,
    ):
        """
        Initialize simulation server framework.

        Args:
            server_name: Name for the MCP server.
            production_server_app: The production FastMCP app instance for validation.
        """
        self.server_name = server_name
        self.production_server_app = production_server_app
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_lock = threading.Lock()
        self._domain_tools: Dict[str, Callable] = {}
        self._domain_resources: Dict[str, Callable] = {}

        # Create FastMCP server with proper lifespan
        self.mcp = FastMCP(server_name, lifespan=self._lifespan)

        # Discover, validate, and register domain tools and resources
        self._validate_and_register_tools()
        self._discover_and_register_resources()

    @asynccontextmanager
    async def _lifespan(self, app: FastMCP):
        """Server lifespan management."""
        print(f"🚀 {self.server_name} Simulation Server")
        print("🎯 Framework: Enforces no session tool pollution")
        print(f"🔧 Domain tools: {list(self._domain_tools.keys())}")
        print(f"📦 Domain resources: {list(self._domain_resources.keys())}")
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
        Get or create session and return its state.

        This handles session initialization using MCP spec:
        - Configuration from initializationOptions
        - Automatic environment creation
        - Internal session management (no tools exposed)

        Returns:
            Session state dictionary instead of injecting into context
        """
        session_id = self._get_session_id(ctx)

        with self.session_lock:
            if session_id not in self.sessions:
                # TODO: Extract from ctx.session.initialization_options when available
                config = self.get_default_config()
                seed = None  # Would come from initializationOptions

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
                }
                print(
                    f"🆕 Simulation session created: {session_id[:16]}... (seed={seed})"
                )

            self.sessions[session_id]["last_used"] = time.time()
            # Return session state instead of trying to inject into context
            return self.sessions[session_id]

    def _validate_and_register_tools(self):
        """
        Discover, validate, and register tools marked with @simulation_tool.
        """
        # 1. Discover tools on the subclass instance
        discovered_tools = {}
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, "_is_simulation_tool"):
                discovered_tools[method.__name__] = method
        self._domain_tools = discovered_tools

        # 2. Validate against production server if provided
        if self.production_server_app:
            prod_tools = self.production_server_app._tool_manager._tools
            prod_tool_names = set(prod_tools.keys())
            sim_tool_names = set(self._domain_tools.keys())

            if prod_tool_names != sim_tool_names:
                raise ToolMismatchError(
                    f"Tool mismatch!\n"
                    f"  - In Production but not Simulation: {prod_tool_names - sim_tool_names}\n"
                    f"  - In Simulation but not Production: {sim_tool_names - prod_tool_names}"
                )

            for name, sim_tool in self._domain_tools.items():
                prod_tool = prod_tools[name]
                prod_sig = inspect.signature(prod_tool.fn)
                sim_sig = inspect.signature(sim_tool)

                # Exclude 'self', 'ctx', and 'session_state' from simulation signature for comparison
                sim_params = [
                    p
                    for p in sim_sig.parameters.values()
                    if p.name not in ("self", "ctx", "session_state")
                ]
                prod_params = [
                    p
                    for p in prod_sig.parameters.values()
                    if p.name not in ("self", "ctx")
                ]

                if len(sim_params) != len(prod_params) or any(
                    s.name != p.name or s.annotation != p.annotation
                    for s, p in zip(sim_params, prod_params)
                ):
                    raise SignatureMismatchError(
                        f"Signature mismatch for tool '{name}':\n"
                        f"  - Production: {prod_sig}\n"
                        f"  - Simulation: {sim_sig}"
                    )

        # 3. Register the validated tools
        for tool_name, tool_func in self._domain_tools.items():
            # Get the production tool signature to create a properly wrapped function
            if self.production_server_app:
                prod_tool = self.production_server_app._tool_manager._tools[tool_name]
                prod_sig = inspect.signature(prod_tool.fn)

                # Create a wrapper function with the exact production signature
                # Use default parameters to capture variables from loop
                def create_wrapper(
                    original_func=tool_func, prod_signature=prod_sig, self_ref=self
                ):
                    def wrapper(*args, **kwargs):
                        # Get context from kwargs (FastMCP injects this)
                        ctx = kwargs.pop("ctx", None)
                        if ctx is None:
                            raise ValueError("Context not available in tool call")

                        # Get session state
                        session_state = self_ref._get_or_create_session(ctx)

                        # Call original function with session_state
                        return original_func(
                            *args, ctx=ctx, session_state=session_state, **kwargs
                        )

                    # Copy the production signature to the wrapper
                    wrapper.__signature__ = prod_signature
                    wrapper.__name__ = original_func.__name__
                    wrapper.__doc__ = original_func.__doc__
                    return wrapper

                wrapped_tool = create_wrapper()
            else:
                # Fallback for when no production server is provided
                # Use default parameter to capture tool_func from loop
                def create_fallback_wrapper(original_func=tool_func, self_ref=self):
                    @functools.wraps(original_func)
                    def wrapped_tool(*args, ctx: Context, **kwargs):
                        session_state = self_ref._get_or_create_session(ctx)
                        # Pass session state as a special argument instead of injecting into context
                        return original_func(
                            *args, ctx=ctx, session_state=session_state, **kwargs
                        )

                    return wrapped_tool

                wrapped_tool = create_fallback_wrapper()

            self.mcp.tool(name=tool_name)(wrapped_tool)

    def _discover_and_register_resources(self):
        """
        Discover and register resources on the subclass instance.
        """
        # 1. Discover resources on the subclass instance
        discovered_resources = {}
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, "_is_resource"):
                discovered_resources[method.__name__] = method
        self._domain_resources = discovered_resources

        # 2. Register the discovered resources with their URI patterns
        for resource_name, resource_func in self._domain_resources.items():
            uri_pattern = getattr(resource_func, "_resource_uri", resource_name)
            self.mcp.resource(uri_pattern)(resource_func)

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

    def run(self, transport: str = "streamable-http", **kwargs):
        """
        Run the simulation server.

        Args:
            transport: Transport protocol
            **kwargs: Additional arguments for FastMCP.run()
        """
        print(f"📡 Starting simulation server with {transport} transport")
        print(f"🎮 Domain tools: {list(self._domain_tools.keys())}")
        print(f"📦 Domain resources: {list(self._domain_resources.keys())}")
        if self.production_server_app:
            print("✅ Tool signatures validated against production server.")
        print("🚫 No session management tools exposed (framework enforced)")
        print()

        # Pass all arguments directly to FastMCP.run()
        self.mcp.run(transport=transport, **kwargs)
