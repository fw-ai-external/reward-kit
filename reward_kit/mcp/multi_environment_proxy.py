"""
Multi-Environment Proxy Server for MCP-Gym

This module provides a proxy MCP server that manages multiple isolated environment instances
using CondaServerProcessManager. It handles session management, routing, and provides a
standard MCP interface compatible with the existing GeneralMCPVectorEnv and rollout system.

Key Features:
- Manages multiple server instances with full conda isolation
- Session-aware routing (each MCP session gets its own server instance)
- Standard MCP interface compatible with existing rollout.py
- Automatic server lifecycle management
- Support for different seeds per environment instance
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Dict, List, Optional, Set

import aiohttp
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.fastmcp import Context, FastMCP

from .fastmcp_hacks import FunctionResourceWithContext
from .process_manager import CondaServerProcessManager
from .simple_process_manager import SimpleServerProcessManager


class MultiEnvironmentProxy:
    """
    Proxy server that manages multiple isolated MCP environment instances.

    This server acts as a proxy between rollout clients and individual environment servers,
    providing session management and routing capabilities while maintaining full MCP compatibility.
    """

    def __init__(
        self,
        server_script_path: str,
        requirements_path: str,
        conda_base_env: str = "base",
        port_range: tuple[int, int] = (10000, 11000),
        max_concurrent_envs: int = 10,
    ):
        """
        Initialize the multi-environment proxy server.
        """
        self.server_script_path = server_script_path
        self.requirements_path = requirements_path
        self.conda_base_env = conda_base_env
        self.port_range = port_range
        self.max_concurrent_envs = max_concurrent_envs

        self.logger = logging.getLogger(__name__)

        force_simple_manager = (
            os.environ.get("FORCE_SIMPLE_PROCESS_MANAGER", "false").lower() == "true"
        )
        if force_simple_manager:
            self.logger.info(
                "Forcing SimpleServerProcessManager due to FORCE_SIMPLE_PROCESS_MANAGER env var"
            )
            self.process_manager = SimpleServerProcessManager(
                script_path=server_script_path,
                port_range=port_range,
            )
        else:
            # NOTE: This part is simplified for clarity, your existing logic is fine here.
            self.logger.warning("Using SimpleServerProcessManager as a fallback.")
            self.process_manager = SimpleServerProcessManager(
                script_path=server_script_path,
                port_range=port_range,
            )

        # Session management state
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_to_port: Dict[str, int] = {}
        self.port_to_sessions: Dict[int, Set[str]] = {}
        self.session_counter = 0
        self.session_creation_lock = asyncio.Lock()
        self.http_client: Optional[aiohttp.ClientSession] = None

        # Tool discovery state
        self.discovered_tool_schemas = {}
        self.tools_discovered = False

        # Create the FastMCP instance. It will set up its own default handlers.
        self.mcp = FastMCP("MultiEnvironmentProxy", lifespan=self._lifespan)

        # NOW, after self.mcp is fully constructed, we can safely access its internals
        # and replace its default handlers with our custom proxying ones.
        self._setup_proxy_handlers()

    def _setup_proxy_handlers(self):
        """
        Manually registers our proxy handlers, overwriting the defaults set by FastMCP.
        This is called *after* FastMCP's __init__ is complete to avoid startup errors.
        """

        # Define handlers as regular nested functions. They will "close over" `self`.
        async def initial_state_resource_handler(ctx: Context) -> str:
            return await self._proxy_resource_request("game://initial_state", ctx)

        async def reward_resource_handler(ctx: Context) -> str:
            return await self._proxy_resource_request("control://reward", ctx)

        async def status_resource_handler(ctx: Context) -> str:
            return await self._proxy_resource_request("control://status", ctx)

        async def info_resource_handler(ctx: Context) -> str:
            return await self._proxy_resource_request("control://info", ctx)

        # Access the resource manager via the self.mcp instance
        resource_manager = self.mcp._resource_manager

        # Manually create and add resources, passing the FastMCP instance to our custom class
        resource_manager.add_resource(
            FunctionResourceWithContext(
                fn=initial_state_resource_handler,
                fastmcp_server=self.mcp,
                uri="game://initial_state",
            )
        )
        resource_manager.add_resource(
            FunctionResourceWithContext(
                fn=reward_resource_handler,
                fastmcp_server=self.mcp,
                uri="control://reward",
            )
        )
        resource_manager.add_resource(
            FunctionResourceWithContext(
                fn=status_resource_handler,
                fastmcp_server=self.mcp,
                uri="control://status",
            )
        )
        resource_manager.add_resource(
            FunctionResourceWithContext(
                fn=info_resource_handler, fastmcp_server=self.mcp, uri="control://info"
            )
        )

        self.logger.info("✅ Custom proxy resource handlers configured successfully.")

    def _is_conda_available(self) -> bool:
        """Check if conda is available in the system PATH."""
        import subprocess

        try:
            result = subprocess.run(
                ["conda", "--version"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @asynccontextmanager
    async def _lifespan(self, app: FastMCP):
        """Lifespan context manager for the FastMCP app."""
        # Startup
        self.http_client = aiohttp.ClientSession()
        self.logger.info("🚀 Multi-Environment Proxy server starting...")

        # Perform eager tool discovery at startup
        await self._eager_tool_discovery()

        yield  # Server is running

        # Shutdown
        self.logger.info("🧹 Shutting down Multi-Environment Proxy...")

        # Clean up all MCP client connections
        for session_id in list(self.sessions.keys()):
            await self.cleanup_session(session_id)

        self.process_manager.stop_all()
        if self.http_client:
            await self.http_client.close()

    async def _eager_tool_discovery(self):
        """Discover tools at startup so they're always available."""
        self.logger.info("🔍 Starting eager tool discovery at startup...")

        try:
            # Check if we should skip eager discovery (e.g., in test environments)
            skip_eager_discovery = (
                os.environ.get("SKIP_EAGER_TOOL_DISCOVERY", "false").lower() == "true"
            )
            if skip_eager_discovery:
                self.logger.info(
                    "⏭️ Skipping eager tool discovery due to SKIP_EAGER_TOOL_DISCOVERY env var"
                )
                return

            # Log initial state
            self.logger.info(
                f"🔍 STARTUP: Initial proxy tools registered: {list(self.discovered_tool_schemas.keys())}"
            )
            self.logger.info(
                f"🔍 STARTUP: Initial tools discovered flag: {self.tools_discovered}"
            )

            # Start a temporary backend server for discovery with timeout
            default_seed = 42
            self.logger.info(
                f"🔍 STARTUP: Starting temporary backend server with seed {default_seed}"
            )

            # Add timeout to prevent hanging during conda environment creation
            try:
                backend_port = await asyncio.wait_for(
                    self._start_backend_server(default_seed),
                    timeout=60.0,  # 60 second timeout
                )
                self.logger.info(
                    f"✅ Started temporary backend server on port {backend_port} for tool discovery"
                )
            except asyncio.TimeoutError:
                self.logger.warning(
                    "⏰ Timeout during backend server startup for tool discovery - skipping eager discovery"
                )
                return

            # Wait for server to be fully ready using resource health check
            self.logger.info(
                f"🏥 STARTUP: Checking server health using game://initial_state resource"
            )
            await self._wait_for_server_ready_with_resource_check(backend_port)

            # Now discover and register tools
            self.logger.info(
                f"🔍 STARTUP: Discovering tools from ready server {backend_port}"
            )
            await self._discover_and_register_tools(backend_port)

            # Mark discovery as complete
            self.tools_discovered = True
            self.logger.info(f"🔍 STARTUP: Setting tools_discovered flag to True")

            # Log final state
            self.logger.info(
                f"✅ STARTUP: Tool discovery completed - discovered {len(self.discovered_tool_schemas)} tools"
            )
            self.logger.info(
                f"🔍 STARTUP: Final proxy tools registered: {list(self.discovered_tool_schemas.keys())}"
            )
            self.logger.info(
                f"🔍 STARTUP: Final tools discovered flag: {self.tools_discovered}"
            )

            for tool_name in self.discovered_tool_schemas.keys():
                self.logger.info(f"  🛠️  Registered tool: {tool_name}")

        except Exception as e:
            self.logger.error(f"❌ Error during eager tool discovery: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            # Continue anyway - tools can be discovered later if needed

    async def _start_backend_server(self, seed: int) -> int:
        """Start a backend server and return its port."""
        try:
            # Run the synchronous process manager start_server in an executor to avoid blocking
            self.logger.info(f"Creating backend server process with seed {seed}")
            loop = asyncio.get_event_loop()
            server_port = await loop.run_in_executor(
                None, self.process_manager.start_server, seed
            )
            self.logger.info(
                f"Started backend server on port {server_port} with seed {seed}"
            )

            # Wait for server to be ready
            await self._wait_for_server_ready(server_port)

            return server_port
        except Exception as e:
            self.logger.error(f"Failed to start backend server with seed {seed}: {e}")
            raise

    def _setup_mcp_handlers(self):
        """
        Setup MCP protocol handlers manually to work around a limitation in the
        @mcp.resource decorator, which doesn't support context injection.
        """
        from .fastmcp_hacks import FunctionResourceWithContext

        # Define handlers as regular nested functions. They will "close over" `self`.
        async def initial_state_resource_handler(ctx: Context) -> str:
            return await self._proxy_resource_request("game://initial_state", ctx)

        async def reward_resource_handler(ctx: Context) -> str:
            return await self._proxy_resource_request("control://reward", ctx)

        async def status_resource_handler(ctx: Context) -> str:
            return await self._proxy_resource_request("control://status", ctx)

        async def info_resource_handler(ctx: Context) -> str:
            return await self._proxy_resource_request("control://info", ctx)

        # CORRECT: Access the resource manager via the self.mcp instance
        resource_manager = self.mcp._resource_manager

        # Manually create and add resources, passing the FastMCP instance to our custom class
        resource_manager.add_resource(
            FunctionResourceWithContext(
                fn=initial_state_resource_handler,
                fastmcp_server=self.mcp,  # Pass the server instance to our hack
                uri="game://initial_state",
            )
        )
        resource_manager.add_resource(
            FunctionResourceWithContext(
                fn=reward_resource_handler,
                fastmcp_server=self.mcp,
                uri="control://reward",
            )
        )
        resource_manager.add_resource(
            FunctionResourceWithContext(
                fn=status_resource_handler,
                fastmcp_server=self.mcp,
                uri="control://status",
            )
        )
        resource_manager.add_resource(
            FunctionResourceWithContext(
                fn=info_resource_handler, fastmcp_server=self.mcp, uri="control://info"
            )
        )

        self.logger.info(
            "✅ MCP resource handlers manually configured to support context injection."
        )

    def _get_session_id_from_context(self, ctx: Context) -> str:
        """
        Extract the actual MCP session ID from the context.
        """
        # --- Start of New Diagnostic Logging ---
        print("--- 🔍 DIAGNOSTIC DUMP: _get_session_id_from_context ---")
        print(f"Context object ID: {id(ctx)}")

        if hasattr(ctx, "session") and ctx.session:
            self.logger.info(
                f"✅ ctx.session found. Type: {type(ctx.session)}, ID: {id(ctx.session)}"
            )

            # Check for client_params
            if hasattr(ctx.session, "client_params"):
                params = ctx.session.client_params
                self.logger.info(f"✅ ctx.session.client_params found. Value: {params}")
                if hasattr(params, "session_id") and params.session_id:
                    self.logger.info(
                        f"  --> SUCCESS: Found session_id: {params.session_id}"
                    )
                else:
                    self.logger.warning(
                        "  --> WARNING: 'session_id' attribute NOT FOUND or is None in client_params."
                    )
            else:
                self.logger.error(
                    "  --> ERROR: ctx.session.client_params attribute NOT FOUND."
                )

            # Check for client_info as a fallback source of data
            if hasattr(ctx.session, "client_info"):
                self.logger.info(
                    f"✅ ctx.session.client_info found. Value: {ctx.session.client_info}"
                )
            else:
                self.logger.warning("  --> WARNING: ctx.session.client_info NOT FOUND.")
        else:
            self.logger.error("❌ ctx.session attribute NOT FOUND or is None.")
        self.logger.info("--- END DIAGNOSTIC DUMP ---")
        # --- End of New Diagnostic Logging ---

        # The actual logic we are testing
        if (
            hasattr(ctx, "session")
            and ctx.session
            and hasattr(ctx.session, "client_params")
            and ctx.session.client_params
            and hasattr(ctx.session.client_params, "session_id")
            and ctx.session.client_params.session_id
        ):
            session_id = ctx.session.client_params.session_id
            self.logger.info(
                f"✅ Using stable MCP session ID from client_params: {session_id}"
            )
            return session_id

        # If we reach here, it's a critical failure.
        error_msg = (
            "❌ CRITICAL: Cannot extract stable session ID from context's client_params. "
            "This prevents session reuse. Ensure the MCP client is sending session information correctly. "
            f"Context type: {type(ctx)}, "
            f"Has session: {hasattr(ctx, 'session')}, "
            f"Session value: {getattr(ctx, 'session', None)}"
        )
        self.logger.error(error_msg)
        if hasattr(ctx, "session") and ctx.session:
            self.logger.error(f"  Session attributes: {dir(ctx.session)}")
            if hasattr(ctx.session, "client_params"):
                self.logger.error(f"  Client Params: {ctx.session.client_params}")

        # Fail aggressively to prevent silent errors.
        raise RuntimeError(
            "Failed to extract a stable MCP session ID from the request context."
        )

    async def _ensure_backend_for_session(self, session_id: str, ctx: Context) -> None:
        """
        Ensures a backend server is running for the given session_id.
        If it's the first time seeing this session_id, it starts a new server.
        """
        # This function is the single point of truth for creating backend environments.
        # It's called when a stable session_id is identified for the first time.
        async with self.session_creation_lock:
            # Double-check inside the lock to prevent race conditions
            if session_id in self.sessions:
                return

            self.logger.info(
                f"🚀 First-time setup for session {session_id}. Creating backend environment."
            )

            # Extract seed and config from client_info if available
            seed = None
            config = {}
            if (
                hasattr(ctx, "session")
                and ctx.session
                and hasattr(ctx.session, "client_info")
                and ctx.session.client_info
                and hasattr(ctx.session.client_info, "_extra")
                and ctx.session.client_info._extra
            ):
                extra = ctx.session.client_info._extra
                if extra and isinstance(extra, dict):
                    seed = extra.get("seed")
                    config = extra.get("config", {})
                    if seed is None and isinstance(config, dict) and "seed" in config:
                        seed = config["seed"]
                    self.logger.info(
                        f"📋 Extracted from client_info: seed={seed}, config={config}"
                    )

            if seed is None:
                self.logger.warning(
                    f"Could not find seed for session {session_id}, using deterministic fallback."
                )
                import hashlib

                seed = int(hashlib.md5(session_id.encode()).hexdigest()[:8], 16)

            # Check if we're at max capacity
            if len(self.sessions) >= self.max_concurrent_envs:
                self.logger.error(
                    f"❌ Maximum concurrent environments ({self.max_concurrent_envs}) reached. Cannot create backend for session {session_id}."
                )
                # This will cause subsequent lookups to fail, which is the desired behavior.
                raise RuntimeError(
                    f"Maximum concurrent environments ({self.max_concurrent_envs}) reached"
                )

            # Start a new server instance
            try:
                self.logger.info(
                    f"Starting server for session {session_id} with seed {seed}"
                )
                server_port = self.process_manager.start_server(seed=seed)
                self.logger.info(
                    f"✅ Started server on port {server_port} for session {session_id}"
                )
            except Exception as e:
                self.logger.error(
                    f"❌ Failed to start server for session {session_id}: {e}",
                    exc_info=True,
                )
                raise RuntimeError(f"Could not create environment instance: {e}")

            # Store session info
            self.sessions[session_id] = {
                "server_port": server_port,
                "seed": seed,
                "created_at": time.time(),
                "env_index": self.session_counter,
            }
            self.session_to_port[session_id] = server_port
            self.port_to_sessions.setdefault(server_port, set()).add(session_id)
            self.session_counter += 1

            self.logger.info(
                f"✅ Backend for session {session_id} is ready on port {server_port}."
            )

            # Wait for server to be fully ready before allowing requests to proceed
            await self._wait_for_server_ready_with_resource_check(server_port)

            # Eagerly discover and register tools if this is the first environment
            if not self.tools_discovered:
                self.logger.info(
                    f"🛠️ Performing tool discovery from first backend server (port {server_port})..."
                )
                await self._discover_and_register_tools(server_port)
                self.tools_discovered = True

    async def _create_mcp_connection(self, server_port: int, operation_func):
        """Create an MCP client connection and execute an operation using the official pattern."""
        url = f"http://localhost:{server_port}/mcp"
        self.logger.debug(f"Creating MCP connection to {url}")

        try:
            # Use the official async context manager pattern
            async with streamablehttp_client(url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    self.logger.debug(
                        f"MCP session initialized for server {server_port}"
                    )

                    # Execute the operation with the initialized session
                    return await operation_func(session)

        except Exception as e:
            self.logger.error(
                f"Failed to execute MCP operation on server {server_port}: {e}"
            )
            raise

    async def _discover_and_register_tools(self, server_port: int):
        """Discover tools from target server and register them on the proxy."""
        try:
            self.logger.info(
                f"🔍 Starting automatic tool discovery for server {server_port}"
            )

            # Use MCP client to discover tools from the backend server
            async def discover_tools_operation(session):
                tools_response = await session.list_tools()
                return tools_response

            # Get tools from the backend server
            tools_response = await self._create_mcp_connection(
                server_port, discover_tools_operation
            )

            if (
                not tools_response
                or not hasattr(tools_response, "tools")
                or not tools_response.tools
            ):
                self.logger.warning(f"No tools found on backend server {server_port}")
                return

            self.logger.info(
                f"✅ Discovered {len(tools_response.tools)} tools from backend server {server_port}"
            )

            # Convert MCP tools to proxy-compatible format and register them
            for mcp_tool in tools_response.tools:
                tool_info = self._convert_mcp_tool_to_proxy_schema(mcp_tool)
                if tool_info:
                    tool_name = tool_info["name"]
                    if (
                        tool_name not in self.discovered_tool_schemas
                    ):  # Only register if not already discovered
                        self._register_proxy_tool(tool_info)
                        self.discovered_tool_schemas[tool_name] = (
                            tool_info  # Store detailed schema
                        )
                        self.logger.info(f"🛠️ Registered discovered tool: {tool_name}")

            self.logger.info(f"🎉 Tool discovery completed for server {server_port}")

        except Exception as e:
            self.logger.error(
                "Failed to discover tools from server on port %s: %s", server_port, e
            )
            # Log the full traceback for debugging
            import traceback

            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise  # Re-raise to let caller handle the error

    def _convert_mcp_tool_to_proxy_schema(self, mcp_tool) -> Dict[str, Any]:
        """Convert MCP tool definition to proxy-compatible schema."""
        try:
            # Extract tool information from MCP tool object
            tool_name = mcp_tool.name
            tool_description = mcp_tool.description or f"Execute {tool_name} action"

            self.logger.debug(f"Converting MCP tool: {tool_name}")
            self.logger.debug(f"  Description: {tool_description}")
            self.logger.debug(f"  Raw tool object: {mcp_tool}")

            # Extract input schema
            input_schema = {}
            if hasattr(mcp_tool, "inputSchema") and mcp_tool.inputSchema:
                input_schema = mcp_tool.inputSchema
                self.logger.debug(f"  Found inputSchema: {input_schema}")
            elif hasattr(mcp_tool, "input_schema") and mcp_tool.input_schema:
                input_schema = mcp_tool.input_schema
                self.logger.debug(f"  Found input_schema: {input_schema}")
            else:
                self.logger.warning(f"  No input schema found for tool {tool_name}")

            # Ensure input_schema is a dictionary
            if not isinstance(input_schema, dict):
                self.logger.warning(
                    f"  Input schema is not a dict for {tool_name}, got: {type(input_schema)} - {input_schema}"
                )
                input_schema = {}

            result = {
                "name": tool_name,
                "description": tool_description,
                "input_schema": input_schema,
            }

            self.logger.debug(f"  Converted schema: {result}")
            return result

        except Exception as e:
            self.logger.error(
                f"Failed to convert MCP tool {mcp_tool} to proxy schema: {e}"
            )
            return None

    def _register_proxy_tool(self, tool_info: dict):
        """Register a single proxy tool with dynamic function signature generation."""
        tool_name = tool_info["name"]
        tool_description = tool_info["description"]
        input_schema = tool_info.get("input_schema", {})

        self.logger.info(f"🛠️ REGISTRATION: Starting registration for tool: {tool_name}")
        self.logger.info(f"🛠️ REGISTRATION: FastMCP instance: {type(self.mcp)}")
        self.logger.info(f"🛠️ REGISTRATION: Tool description: {tool_description}")
        self.logger.info(f"🛠️ REGISTRATION: Input schema: {input_schema}")

        # Generate proxy function with correct signature based on input schema
        proxy_func = self._generate_proxy_function(
            tool_name, tool_description, input_schema
        )

        self.logger.info(f"🛠️ REGISTRATION: Generated proxy function: {proxy_func}")
        self.logger.info(
            f"🛠️ REGISTRATION: Proxy function name: {proxy_func.__name__ if proxy_func else 'None'}"
        )

        # Register the tool using FastMCP's add_tool method
        try:
            self.logger.info(
                f"🛠️ REGISTRATION: Calling self.mcp.add_tool for {tool_name}"
            )
            self.mcp.add_tool(proxy_func, name=tool_name, description=tool_description)
            self.logger.info(
                f"✅ REGISTRATION: Successfully registered proxy tool: {tool_name}"
            )

            # Verify registration by checking FastMCP's internal state
            if hasattr(self.mcp, "_tools"):
                self.logger.info(
                    f"🔍 REGISTRATION: FastMCP tools after registration: {list(self.mcp._tools.keys()) if self.mcp._tools else 'No _tools attribute'}"
                )
            else:
                self.logger.warning(f"⚠️ REGISTRATION: FastMCP has no _tools attribute")

        except Exception as e:
            self.logger.error(
                f"❌ REGISTRATION: Failed to register tool {tool_name}: {e}"
            )
            import traceback

            self.logger.error(
                f"❌ REGISTRATION: Full traceback: {traceback.format_exc()}"
            )

    def _generate_proxy_function(
        self, tool_name: str, tool_description: str, input_schema: Dict[str, Any]
    ):
        """Generate a proxy function with dynamic signature based on input schema."""

        # Parse the input schema to determine required parameters
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        # Log the schema we're working with for debugging
        self.logger.debug(
            f"Generating proxy function for {tool_name} with schema: {input_schema}"
        )

        # Create the proper function signature dynamically using exec
        # This ensures FastMCP introspects the correct parameters

        if not properties:
            # No parameters - simple case
            function_code = f"""
async def {tool_name}_proxy(ctx: Context) -> dict:
    \"\"\"Dynamically created proxy tool function with no parameters.\"\"\"
    return await proxy_instance._proxy_tool_call("{tool_name}", {{}}, ctx)
"""
        else:
            # Build parameter list based on schema
            param_parts = []
            param_mapping = {}

            for param_name, param_schema in properties.items():
                param_type = param_schema.get("type", "string")
                is_required = param_name in required

                # Build parameter signature
                if param_type == "string":
                    type_hint = "str"
                elif param_type == "integer":
                    type_hint = "int"
                elif param_type == "number":
                    type_hint = "float"
                elif param_type == "boolean":
                    type_hint = "bool"
                elif param_type == "array":
                    type_hint = "list"
                elif param_type == "object":
                    type_hint = "dict"
                else:
                    type_hint = "str"  # Default fallback

                if is_required:
                    param_parts.append(f"{param_name}: {type_hint}")
                else:
                    param_parts.append(f"{param_name}: {type_hint} = None")

                param_mapping[param_name] = param_name

            # Add ctx parameter at the end
            param_parts.append("ctx: Context")
            params_str = ", ".join(param_parts)

            # Create the argument mapping for the tool call
            arg_dict_parts = []
            for param_name in properties.keys():
                if param_name in required:
                    arg_dict_parts.append(f'"{param_name}": {param_name}')
                else:
                    arg_dict_parts.append(f'"{param_name}": {param_name}')

            arg_dict_str = "{" + ", ".join(arg_dict_parts) + "}"

            # Build the argument assignment lines with proper indentation
            arg_assignments = []
            for param_name in properties.keys():
                if param_name in required:
                    arg_assignments.append(f'    args["{param_name}"] = {param_name}')
                else:
                    arg_assignments.append(
                        f'    if {param_name} is not None: args["{param_name}"] = {param_name}'
                    )

            function_code = f"""
async def {tool_name}_proxy({params_str}) -> dict:
    \"\"\"Dynamically created proxy tool function for {tool_name}.\"\"\"
    # Build arguments dict, excluding None values for optional parameters
    args = {{}}
{chr(10).join(arg_assignments)}
    return await proxy_instance._proxy_tool_call("{tool_name}", args, ctx)
"""

        # Execute the function code to create the actual function
        namespace = {"Context": Context, "proxy_instance": self, "dict": dict}

        try:
            exec(function_code, namespace)
            proxy_func = namespace[f"{tool_name}_proxy"]

            # Set function metadata
            proxy_func.__name__ = tool_name
            proxy_func.__doc__ = tool_description

            self.logger.debug(
                f"Successfully created proxy function for {tool_name} with signature: {proxy_func.__annotations__}"
            )

            return proxy_func

        except Exception as e:
            self.logger.error(f"Failed to generate proxy function for {tool_name}: {e}")
            self.logger.error(f"Function code: {function_code}")

            # Fallback to generic kwargs function if generation fails
            async def fallback_proxy_func(ctx: Context, **kwargs) -> dict:
                """Fallback proxy function with generic parameters."""
                return await self._proxy_tool_call(tool_name, kwargs, ctx)

            fallback_proxy_func.__name__ = tool_name
            fallback_proxy_func.__doc__ = tool_description
            return fallback_proxy_func

    def _json_schema_type_to_python_type(self, json_type: str) -> type:
        """Convert JSON schema type to Python type."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        return type_mapping.get(json_type, str)  # Default to str

    async def _wait_for_server_ready(self, server_port: int, max_attempts: int = 10):
        """Wait for the target server to be ready to accept connections."""
        self.logger.info(f"🏥 Starting server readiness check for port {server_port}")
        for attempt in range(max_attempts):
            try:
                self.logger.debug(
                    f"🔍 Readiness check attempt {attempt + 1}/{max_attempts} for port {server_port}"
                )

                # Try the MCP endpoint first (this is more reliable for our servers)
                try:
                    async with self.http_client.get(
                        f"http://localhost:{server_port}/mcp",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as response:
                        # Any response from MCP endpoint means server is up
                        # Even 404 or redirect is fine as it means the server is responding
                        if response.status in [
                            200,
                            307,
                            404,
                            405,
                        ]:  # Common responses from MCP servers
                            self.logger.info(
                                f"✅ Server on port {server_port} is responding to MCP endpoint (attempt {attempt + 1})"
                            )
                            return
                        else:
                            self.logger.debug(
                                f"🔍 MCP endpoint returned {response.status}"
                            )
                except Exception as mcp_e:
                    self.logger.debug(f"🔍 MCP endpoint check failed: {mcp_e}")

                # Fallback to health endpoint if available
                try:
                    async with self.http_client.get(
                        f"http://localhost:{server_port}/health",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as response:
                        if response.status == 200:
                            self.logger.info(
                                f"✅ Server on port {server_port} health check passed (attempt {attempt + 1})"
                            )
                            return
                        else:
                            self.logger.debug(
                                f"🔍 Health endpoint returned {response.status}"
                            )
                except Exception as health_e:
                    self.logger.debug(f"🔍 Health endpoint check failed: {health_e}")

                # Fallback to simple connection test
                try:
                    async with self.http_client.get(
                        f"http://localhost:{server_port}/",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as response:
                        # Any response means server is up
                        self.logger.info(
                            f"✅ Server on port {server_port} is responding (attempt {attempt + 1}, status: {response.status})"
                        )
                        return
                except Exception as root_e:
                    self.logger.debug(f"🔍 Root endpoint check failed: {root_e}")

            except Exception as e:
                self.logger.debug(
                    f"🔍 Server readiness check failed (attempt {attempt + 1}): {e}"
                )

            await asyncio.sleep(2)  # Wait a bit longer between attempts

        self.logger.warning(
            f"⚠️ Server on port {server_port} may not be ready after {max_attempts} attempts, but proceeding anyway"
        )

    async def _wait_for_server_ready_with_resource_check(
        self, server_port: int, max_attempts: int = 10
    ):
        """Wait for the server to be ready using game://initial_state resource as health check."""
        self.logger.info(
            f"🏥 Starting resource-based health check for server on port {server_port}"
        )

        for attempt in range(max_attempts):
            try:
                self.logger.debug(
                    f"🔍 Resource health check attempt {attempt + 1}/{max_attempts} for port {server_port}"
                )

                # Define operation to read the initial state resource
                async def health_check_operation(session):
                    # Try to read the game://initial_state resource
                    from pydantic import AnyUrl

                    resource_url = AnyUrl("game://initial_state")
                    return await session.read_resource(resource_url)

                # Try to connect and read the resource
                resource_result = await self._create_mcp_connection(
                    server_port, health_check_operation
                )

                # If we get here, the server is responding properly
                self.logger.info(
                    f"✅ Server on port {server_port} passed resource health check (attempt {attempt + 1})"
                )
                self.logger.debug(
                    f"🎮 Initial state resource content: {resource_result}"
                )
                return

            except Exception as e:
                self.logger.debug(
                    f"🔍 Resource health check failed for port {server_port} (attempt {attempt + 1}): {e}"
                )
                await asyncio.sleep(2)  # Wait before retrying

        self.logger.warning(
            f"⚠️ Server on port {server_port} failed resource health check after {max_attempts} attempts, but proceeding anyway"
        )

    async def _proxy_tool_call(
        self, tool_name: str, arguments: dict, ctx: Context
    ) -> dict:
        """Proxy a tool call to the appropriate server."""
        self.logger.info(f"🔧 PROXY_TOOL_CALL: {tool_name}({arguments})")
        session_id = self._get_session_id_from_context(ctx)

        # Ensure a backend exists for this session, creating it if it's the first time.
        await self._ensure_backend_for_session(session_id, ctx)

        # At this point, the session MUST exist.
        if session_id not in self.sessions:
            self.logger.error(
                f"❌ PROXY: Backend creation failed for session {session_id}. Cannot proxy tool call."
            )
            return {
                "error": f"Failed to create backend environment for session {session_id}",
                "tool": tool_name,
            }

        session_info = self.sessions[session_id]
        server_port = session_info["server_port"]
        self.logger.info(
            f"📡 PROXY: Forwarding '{tool_name}' to session {session_id} on port {server_port}"
        )

        # Remainder of the function is the same, starting from the `try...except` block
        # that calls `_create_mcp_connection`. This is a simplified version of that block.
        try:

            async def call_tool_operation(session):
                return await session.call_tool(tool_name, arguments)

            tool_result = await self._create_mcp_connection(
                server_port, call_tool_operation
            )

            # Extract content from the result
            if hasattr(tool_result, "content") and tool_result.content:
                content_text = tool_result.content[0].text
                try:
                    return json.loads(content_text)
                except json.JSONDecodeError:
                    # If the backend returns a non-JSON error string
                    if tool_result.isError:
                        return {"error": content_text, "tool": tool_name}
                    return {"result": content_text}
            return {"error": "Tool result had no content", "tool": tool_name}

        except Exception as e:
            self.logger.error(
                f"❌ PROXY: Call to backend for tool '{tool_name}' failed: {e}",
                exc_info=True,
            )
            return {"error": str(e), "tool": tool_name, "session": session_id}

    async def _proxy_resource_request(self, resource_uri: str, ctx: Context) -> str:
        """Proxy a resource request to the appropriate backend server using proper MCP client."""
        self.logger.info(f"🌐 PROXY_RESOURCE: {resource_uri}")
        session_id = self._get_session_id_from_context(ctx)

        # Ensure a backend exists for this session, creating it if it's the first time.
        try:
            await self._ensure_backend_for_session(session_id, ctx)
        except Exception as e:
            self.logger.error(
                f"❌ Failed to ensure backend for session {session_id} during resource request: {e}",
                exc_info=True,
            )
            return json.dumps(
                {"error": "Failed to create backend for session", "session": session_id}
            )

        # At this point, the session MUST exist.
        if session_id not in self.sessions:
            self.logger.error(
                f"❌ PROXY: Backend creation failed for session {session_id}. Cannot proxy resource request."
            )
            return json.dumps(
                {"error": f"No backend environment for session {session_id}"}
            )

        session_info = self.sessions[session_id]
        server_port = session_info["server_port"]
        self.logger.info(
            f"📡 PROXY: Forwarding resource request for '{resource_uri}' to session {session_id} on port {server_port}"
        )

        # Remainder of the function is the same, starting from the `try...except` block
        # that calls `_create_mcp_connection`.
        try:
            from pydantic import AnyUrl

            resource_url = AnyUrl(resource_uri)

            async def read_resource_operation(session):
                return await session.read_resource(resource_url)

            resource_result = await self._create_mcp_connection(
                server_port, read_resource_operation
            )

            if hasattr(resource_result, "contents") and resource_result.contents:
                content = resource_result.contents[0]
                return content.text
            return json.dumps(
                {"error": "Resource result has no contents", "session": session_id}
            )

        except Exception as e:
            self.logger.error(
                f"❌ Failed to read resource {resource_uri} via MCP client: {e}",
                exc_info=True,
            )
            return json.dumps(
                {
                    "error": f"MCP resource request failed: {str(e)}",
                    "session": session_id,
                }
            )

    async def cleanup_session(self, session_id: str):
        """Clean up a session and its associated server if no longer needed."""
        if session_id not in self.sessions:
            return

        session_info = self.sessions[session_id]
        server_port = session_info["server_port"]

        # No need to clean up MCP sessions since we use per-operation connections
        self.logger.debug(f"Session {session_id} cleanup completed")

        # Remove session from tracking
        del self.sessions[session_id]
        del self.session_to_port[session_id]

        if server_port in self.port_to_sessions:
            self.port_to_sessions[server_port].discard(session_id)

            # If no more sessions using this server, stop it
            if not self.port_to_sessions[server_port]:
                self.process_manager.stop_server(server_port)
                del self.port_to_sessions[server_port]
                self.logger.info("Stopped server on port %s", server_port)

        self.logger.info("Cleaned up session %s", session_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get proxy server statistics."""
        return {
            "active_sessions": len(self.sessions),
            "active_servers": len(self.port_to_sessions),
            "max_concurrent_envs": self.max_concurrent_envs,
            "sessions": list(self.sessions.values()),
        }

    async def run_async(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the proxy server asynchronously."""
        import uvicorn

        # Configure uvicorn to serve the FastMCP streamable HTTP app
        config = uvicorn.Config(
            app=self.mcp.streamable_http_app, host=host, port=port, log_level="info"
        )

        server = uvicorn.Server(config)
        self.logger.info(f"🚀 Starting Multi-Environment Proxy on {host}:{port}")
        await server.serve()

    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the proxy server."""
        import asyncio

        try:
            asyncio.run(self.run_async(host, port))
        except KeyboardInterrupt:
            self.logger.info("Server stopped by user")


# Utility functions for easy integration


def create_multi_environment_proxy(
    server_script_path: str,
    requirements_path: str,
    conda_base_env: str = "base",
    port_range: tuple[int, int] = (10000, 11000),
    max_concurrent_envs: int = 10,
) -> MultiEnvironmentProxy:
    """
    Create a multi-environment proxy server.

    Args:
        server_script_path: Path to the environment server script
        requirements_path: Path to requirements.txt
        conda_base_env: Base conda environment to clone from
        port_range: Range of ports for environment instances
        max_concurrent_envs: Maximum number of concurrent environments

    Returns:
        MultiEnvironmentProxy instance
    """
    return MultiEnvironmentProxy(
        server_script_path=server_script_path,
        requirements_path=requirements_path,
        conda_base_env=conda_base_env,
        port_range=port_range,
        max_concurrent_envs=max_concurrent_envs,
    )


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Environment Proxy Server")
    parser.add_argument(
        "--server-script", required=True, help="Path to environment server script"
    )
    parser.add_argument(
        "--requirements", required=True, help="Path to requirements.txt"
    )
    parser.add_argument(
        "--conda-base-env", default="base", help="Base conda environment"
    )
    parser.add_argument("--port", type=int, default=8080, help="Proxy server port")
    parser.add_argument(
        "--max-envs", type=int, default=10, help="Maximum concurrent environments"
    )

    args = parser.parse_args()

    # Create and run proxy server
    proxy = create_multi_environment_proxy(
        server_script_path=args.server_script,
        requirements_path=args.requirements,
        conda_base_env=args.conda_base_env,
        max_concurrent_envs=args.max_envs,
    )

    print(f"Starting Multi-Environment Proxy Server on port {args.port}")
    print(f"Max concurrent environments: {args.max_envs}")
    print(f"Server script: {args.server_script}")

    proxy.run(port=args.port)
