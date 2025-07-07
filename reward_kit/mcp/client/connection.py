"""
MCP Connection Management

Handles MCP client connections, session initialization, and resource/tool discovery.
Extracted from mcp_env.py to improve modularity.
"""

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Tuple

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from ..types import MCPSession

logger = logging.getLogger(__name__)


class MCPConnectionManager:
    """Manages MCP client connections and session lifecycle."""

    async def initialize_session(self, session: MCPSession) -> None:
        """
        Initialize a persistent MCP session.

        Args:
            session: The MCPSession to initialize
        """
        if session._mcp_session:
            # If a session exists, close it before creating a new one.
            if session._exit_stack:
                try:
                    await session._exit_stack.aclose()
                except asyncio.CancelledError:
                    # Handle cancellation gracefully (especially important for Python 3.12)
                    logger.debug(
                        f"Session {session.session_id} reinit close was cancelled"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error closing existing session {session.session_id} during reinit: {e}"
                    )
                finally:
                    session._exit_stack = None
            session._mcp_session = None

        exit_stack = AsyncExitStack()

        client_info = None
        if session.seed is not None or (
            session.dataset_row and session.dataset_row.environment_context
        ):
            from mcp.types import Implementation

            client_info = Implementation(name="reward-kit", version="1.0.0", _extra={})
            if session.seed is not None:
                client_info._extra["seed"] = session.seed
            if session.dataset_row and session.dataset_row.environment_context:
                client_info._extra["config"] = session.dataset_row.environment_context

        read_stream, write_stream, _ = await exit_stack.enter_async_context(
            streamablehttp_client(session.base_url, terminate_on_close=True)
        )

        mcp_session = await exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream, client_info=client_info)
        )

        await mcp_session.initialize()

        session._mcp_session = mcp_session
        session._exit_stack = exit_stack

    async def discover_tools(self, session: MCPSession) -> List[Dict]:
        """
        Discover available tools from an MCP session.

        Args:
            session: The MCPSession to discover tools from

        Returns:
            List of tool schemas
        """
        if not session._mcp_session:
            raise RuntimeError("Session not initialized")

        mcp_session = session._mcp_session

        # Get available tools from MCP server
        tools_response = await mcp_session.list_tools()
        tools = tools_response.tools if hasattr(tools_response, "tools") else []

        # Convert tools to schema format - filter out internal tools
        tool_schemas = []
        for tool in tools:
            # Only expose action tools to the model, not internal state tools
            tool_schema = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": (
                    tool.inputSchema if hasattr(tool, "inputSchema") else {}
                ),
            }
            tool_schemas.append(tool_schema)

        return tool_schemas

    async def get_initial_state(self, session: MCPSession) -> Any:
        """
        Get initial state from MCP resources during session establishment.
        Uses proper MCP pattern: initial state comes from resources, not tools.

        Args:
            session: The MCPSession to get initial state from

        Returns:
            Initial observation/state
        """
        if not session._mcp_session:
            raise RuntimeError("Session not initialized")

        mcp_session = session._mcp_session
        initial_observation = None

        try:
            # List available resources - this is where initial state should come from
            logger.debug(
                f"Session {session.session_id}: Discovering MCP resources for initial state..."
            )
            resources_response = await mcp_session.list_resources()
            resources = (
                resources_response.resources
                if hasattr(resources_response, "resources")
                else []
            )
            logger.debug(
                f"Session {session.session_id}: Found {len(resources)} MCP resources"
            )
            for resource in resources:
                logger.debug(
                    f"Session {session.session_id}: Resource: {resource.name} | URI: {resource.uri}"
                )

            # Try to identify initial state resource based on common patterns
            initial_state_resource = None
            for resource in resources:
                resource_name_lower = resource.name.lower()
                resource_uri_lower = str(
                    resource.uri
                ).lower()  # Convert AnyUrl to string first
                if any(
                    keyword in resource_name_lower or keyword in resource_uri_lower
                    for keyword in ["initial", "state", "observation", "start"]
                ):
                    initial_state_resource = resource
                    logger.debug(
                        f"Session {session.session_id}: ✅ Found initial state resource: {resource.name} | URI: {resource.uri}"
                    )
                    break

            if initial_state_resource:
                # Read the initial state resource
                logger.debug(
                    f"Session {session.session_id}: Reading initial state from resource: {initial_state_resource.uri}"
                )

                resource_content = await mcp_session.read_resource(
                    initial_state_resource.uri
                )

                # Handle the new ResourceContents format
                if hasattr(resource_content, "text"):
                    try:
                        initial_observation = json.loads(resource_content.text)
                        logger.info(
                            f"Session {session.session_id}: ✅ Successfully parsed JSON initial state with grid_layout: {initial_observation.get('grid_layout', 'N/A')[:20]}..."
                        )
                    except json.JSONDecodeError:
                        initial_observation = {"observation": resource_content.text}
                elif (
                    hasattr(resource_content, "contents")
                    and resource_content.contents
                    and len(resource_content.contents) > 0
                ):
                    # Fallback to old format for backward compatibility
                    content = resource_content.contents[0]
                    if hasattr(content, "text"):
                        try:
                            initial_observation = json.loads(content.text)
                        except json.JSONDecodeError:
                            initial_observation = {"observation": content.text}
                    else:
                        initial_observation = {"observation": str(resource_content)}
                else:
                    logger.warning(
                        f"Session {session.session_id}: Resource content is empty or unrecognized format"
                    )
                    logger.warning(
                        f"Session {session.session_id}: Unexpected resource format"
                    )
                    initial_state_resource = None  # Fall back to other options
            else:
                logger.warning(
                    f"Session {session.session_id}: ❌ No initial state resource found among {len(resources)} resources"
                )
                # Fallback: if no initial state resource, try first available resource
                if resources:
                    first_resource = resources[0]
                    logger.debug(
                        f"Session {session.session_id}: No initial state resource found, using first resource: {first_resource.name}"
                    )
                    logger.debug(
                        f"Session {session.session_id}: About to call mcp_session.read_resource with fallback URI: {first_resource.uri}"
                    )

                    resource_content = await mcp_session.read_resource(
                        first_resource.uri
                    )

                    logger.debug(
                        f"Session {session.session_id}: fallback read_resource returned type: {type(resource_content)}"
                    )
                    logger.debug(
                        f"Session {session.session_id}: fallback read_resource returned value: {resource_content}"
                    )
                    logger.debug(
                        f"Session {session.session_id}: fallback read_resource dir(): {dir(resource_content)}"
                    )

                    # Handle the new ResourceContents format
                    if hasattr(resource_content, "text"):
                        try:
                            initial_observation = json.loads(resource_content.text)
                        except json.JSONDecodeError:
                            initial_observation = {"observation": resource_content.text}
                    elif (
                        hasattr(resource_content, "contents")
                        and resource_content.contents
                        and len(resource_content.contents) > 0
                    ):
                        # Fallback to old format for backward compatibility
                        content = resource_content.contents[0]
                        if hasattr(content, "text"):
                            try:
                                initial_observation = json.loads(content.text)
                            except json.JSONDecodeError:
                                initial_observation = {"observation": content.text}
                        else:
                            initial_observation = {"observation": str(content)}
                    else:
                        logger.warning(
                            f"Session {session.session_id}: Fallback resource has unexpected format"
                        )
                        initial_observation = {"observation": str(resource_content)}
                else:
                    logger.debug(
                        f"Session {session.session_id}: No resources available from MCP server"
                    )

        except Exception as e:
            # If resources are not available, fall back to a default observation
            # This maintains backward compatibility with servers that don't expose resources
            logger.warning(
                f"Session {session.session_id}: Could not get initial state from MCP resources: {e}"
            )
            logger.warning(f"Session {session.session_id}: Exception type: {type(e)}")
            logger.warning(f"Session {session.session_id}: Exception args: {e.args}")
            import traceback

            logger.warning(
                f"Session {session.session_id}: Full traceback: {traceback.format_exc()}"
            )
            initial_observation = {
                "observation": "initial_state",
                "message": "Session established",
            }

        # Ensure we have some observation
        if initial_observation is None:
            logger.debug(f"Session {session.session_id}: Using default initial state")
            initial_observation = {
                "observation": "default_initial_state",
                "session_id": session.session_id,
            }

        return initial_observation

    async def call_tool(
        self, session: MCPSession, tool_name: str, arguments: Dict
    ) -> Tuple[Any, float, bool, Dict]:
        """
        Execute a tool call via MCP protocol.

        Args:
            session: The MCPSession to execute the tool call on
            tool_name: Name of the tool to call
            arguments: Arguments for the tool call

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if not session._mcp_session:
            raise RuntimeError("Session not initialized")

        mcp_session = session._mcp_session

        # Execute the tool call via MCP protocol
        tool_result = await mcp_session.call_tool(tool_name, arguments)

        # Extract results using the working pattern
        if tool_result.content and len(tool_result.content) > 0:
            content = tool_result.content[0]
            if hasattr(content, "text"):
                # Fix: Handle empty or invalid JSON responses gracefully
                if not content.text or content.text.strip() == "":
                    logger.warning(
                        f"Session {session.session_id}: Empty tool response from {tool_name}"
                    )
                    result_data = {
                        "observation": "empty_response",
                        "reward": 0.0,
                        "terminated": False,
                    }
                else:
                    try:
                        result_data = json.loads(content.text)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Session {session.session_id}: Invalid JSON from {tool_name}: {content.text}. Error: {e}"
                        )
                        # Create a structured response from the raw text
                        result_data = {
                            "observation": content.text,
                            "reward": 0.0,
                            "terminated": False,
                            "error": "invalid_json_response",
                        }
            else:
                # Handle non-text content
                result_data = {
                    "observation": str(content),
                    "reward": 0.0,
                    "terminated": False,
                }
        else:
            # Handle completely empty tool result
            logger.warning(
                f"Session {session.session_id}: Tool {tool_name} returned empty result"
            )
            result_data = {
                "observation": "no_response",
                "reward": 0.0,
                "terminated": False,
            }

        # Parse result into observation, reward, done, info
        # Keep full result_data as observation for rich prompt templates
        observation = result_data
        reward = result_data.get("reward", 0.0)
        terminated = result_data.get("terminated", False)
        truncated = result_data.get("truncated", False)
        done = terminated or truncated

        info = {
            "steps": result_data.get("moves", result_data.get("steps", 0)),
            "tool_call": tool_name,
            "arguments": arguments,
        }

        return observation, reward, done, info

    async def close_session(self, session: MCPSession) -> None:
        """
        Close an MCP session and clean up resources.

        Args:
            session: The MCPSession to close
        """
        if session._exit_stack:
            try:
                await session._exit_stack.aclose()
            except asyncio.CancelledError:
                # Handle cancellation gracefully (especially important for Python 3.12)
                logger.debug(f"Session {session.session_id} close was cancelled")
            except Exception as e:
                logger.warning(f"Error closing session {session.session_id}: {e}")
            finally:
                session._exit_stack = None
                session._mcp_session = None
