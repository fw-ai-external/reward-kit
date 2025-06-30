"""
MCP Environment API for reward-kit north star implementation.

This module provides the general tool-calling interface that works with ANY MCP environment
via dataset-driven configuration and automatic tool discovery.

Usage:
    import reward_kit as rk

    # Load dataset with environment configuration and prompts
    dataset = load_jsonl("dataset.jsonl")

    # Create general policy (environment-agnostic)
    policy = rk.FireworksPolicy(model_id="accounts/fireworks/models/qwen3-235b-a22b")

    # Create environments with dataset-driven configuration
    envs = rk.make("http://localhost:8000/mcp", dataset=dataset)

    # Execute tool-calling rollouts
    trajectories = await rk.rollout(envs, policy=policy, steps=512)

Key Features:
- General tool-calling interface that works with any MCP environment
- Dataset-driven configuration with system prompts and user prompt templates
- Automatic MCP tool discovery from servers
- **PROPER MCP PATTERN**: Initial state obtained from MCP resources during session establishment
- Tools used only for actions/interactions, not for getting initial state
- Dynamic user prompt formatting based on current observations
- Environment-agnostic policy that receives tool schemas and makes structured calls
- Backward compatibility with servers that don't expose resources

MCP Integration:
- Session establishment creates MCP connection and discovers resources and tools
- Initial state comes from MCP resources (list_resources + read_resource calls)
- Tools are used for subsequent actions during rollout steps
- Resources provide static/configuration data, tools provide dynamic actions
"""

import asyncio
import json
import logging
import os
import time
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Import Fireworks Build SDK - optional at module level
try:
    from fireworks import LLM

    FIREWORKS_AVAILABLE = True
except ImportError:
    LLM = None
    FIREWORKS_AVAILABLE = False

from .auth import get_fireworks_api_key
from .mcp.types import DatasetRow, MCPSession, MCPToolCall
from .playback_policy import PlaybackPolicyBase

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """Represents a complete rollout trajectory."""

    session: MCPSession
    observations: List[Any]
    actions: List[str]
    rewards: List[float]
    terminated: bool
    total_reward: float
    steps: int
    duration: float


class GeneralMCPVectorEnv:
    """
    General MCP vector environment that works with any MCP server.

    Manages on-demand MCP sessions for rollouts.
    Driven by dataset prompts and MCP tool discovery, not hardcoded logic.
    """

    def __init__(
        self,
        sessions: List[MCPSession],
        dataset_rows: List[DatasetRow],
        user_prompt_formatter: Optional[Callable] = None,
    ):
        """
        Initialize with dataset-driven configuration.

        Args:
            sessions: MCP sessions
            dataset_rows: Full dataset rows with prompts and context
            user_prompt_formatter: Callback to format user prompts dynamically
        """
        self.sessions = sessions
        self.dataset_rows = dataset_rows
        self.user_prompt_formatter = user_prompt_formatter or self._default_formatter
        self.n = len(sessions)
        self.tool_schemas = []  # Discovered from MCP servers

        if len(sessions) != len(dataset_rows):
            raise ValueError(
                f"Sessions ({len(sessions)}) and dataset rows ({len(dataset_rows)}) must have same length"
            )

    async def _initialize_mcp_session(self, session: MCPSession):
        """Initializes a persistent MCP session."""
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

    async def reset(self) -> tuple[List[Any], List[List[Dict]], List[str]]:
        """
        Reset all environments and return observations, tools, and system prompts.

        Establishes persistent MCP sessions for each environment.
        Uses proper MCP pattern: get initial state from resources during session establishment.

        Returns:
            observations: Current state of each environment from MCP resources
            tool_schemas: Available MCP tools for each environment
            system_prompts: System prompts from dataset
        """
        print(f"🔄 Resetting {self.n} MCP environments...")

        async def reset_session(session: MCPSession) -> tuple[Any, List[Dict]]:
            # Establish a persistent session for each environment.
            await self._initialize_mcp_session(session)
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

            # PROPER MCP PATTERN: Get initial state from resources during session establishment
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
                                initial_observation = {
                                    "observation": resource_content.text
                                }
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
                logger.warning(
                    f"Session {session.session_id}: Exception type: {type(e)}"
                )
                logger.warning(
                    f"Session {session.session_id}: Exception args: {e.args}"
                )
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
                logger.debug(
                    f"Session {session.session_id}: Using default initial state"
                )
                initial_observation = {
                    "observation": "default_initial_state",
                    "session_id": session.session_id,
                }

            # Update session state
            session.terminated = False
            session.last_observation = initial_observation
            return initial_observation, tool_schemas

        # Execute resets in parallel, each with its own isolated session.
        tasks = [reset_session(session) for session in self.sessions]
        results = await asyncio.gather(*tasks)

        observations, tool_schemas_list = zip(*results)
        self.tool_schemas = list(tool_schemas_list)

        # Extract system prompts from dataset
        system_prompts = [row.system_prompt for row in self.dataset_rows]

        return list(observations), self.tool_schemas, system_prompts

    async def step(
        self, tool_calls: List[MCPToolCall]
    ) -> tuple[List[Any], List[float], List[bool], List[Dict]]:
        """
        Execute tool calls via MCP protocol using persistent sessions.

        Note: This uses MCP tools for actions/interactions during rollout.
        Initial state was obtained from MCP resources during reset() - different pattern.

        Args:
            tool_calls: Tool calls to execute in each environment

        Returns:
            observations, rewards, dones, infos
        """
        if len(tool_calls) != self.n:
            raise ValueError(f"Expected {self.n} tool calls, got {len(tool_calls)}")

        async def step_session(session: MCPSession, tool_call: MCPToolCall):
            if session.terminated:
                return session.last_observation, 0.0, True, {}

            # Handle special playback termination signal
            if tool_call.tool_name == "_playback_terminate":
                logger.info(
                    f"🎬 Session {session.session_id}: Received playback termination signal"
                )
                session.terminated = True
                return (
                    session.last_observation,
                    0.0,
                    True,
                    {"playback_terminated": True},
                )

            if not session._mcp_session:
                raise RuntimeError("Session not initialized. Call reset() first.")

            mcp_session = session._mcp_session

            # Execute the tool call via MCP protocol
            tool_result = await mcp_session.call_tool(
                tool_call.tool_name, tool_call.arguments
            )

            # Extract results using the working pattern
            if tool_result.content and len(tool_result.content) > 0:
                content = tool_result.content[0]
                if hasattr(content, "text"):
                    # Fix: Handle empty or invalid JSON responses gracefully
                    if not content.text or content.text.strip() == "":
                        logger.warning(
                            f"Session {session.session_id}: Empty tool response from {tool_call.tool_name}"
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
                                f"Session {session.session_id}: Invalid JSON from {tool_call.tool_name}: {content.text}. Error: {e}"
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
                    f"Session {session.session_id}: Tool {tool_call.tool_name} returned empty result"
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

            # Update session state
            session.last_observation = observation
            session.terminated = done

            info = {
                "steps": result_data.get("moves", result_data.get("steps", 0)),
                "tool_call": tool_call.tool_name,
                "arguments": tool_call.arguments,
            }

            return observation, reward, done, info

        # Execute steps in parallel using persistent sessions
        tasks = [
            step_session(session, tool_call)
            for session, tool_call in zip(self.sessions, tool_calls)
        ]
        results = await asyncio.gather(*tasks)

        observations, rewards, dones, infos = zip(*results)
        return list(observations), list(rewards), list(dones), list(infos)

    def format_user_prompts(self, observations: List[Any]) -> List[str]:
        """
        Format user prompts dynamically based on current observations.

        This is the callback pattern - prompts are generated based on current state.
        """
        user_prompts = []

        for obs, row in zip(observations, self.dataset_rows):
            # Use the callback to format the prompt
            prompt = self.user_prompt_formatter(
                row.user_prompt_template, obs, row.environment_context
            )
            user_prompts.append(prompt)

        return user_prompts

    def _default_formatter(self, template: str, observation: Any, context: Dict) -> str:
        """
        Default user prompt formatter.

        Extracts meaningful display data from MCP observations.
        For FrozenLake: extracts grid_layout if available, otherwise uses raw observation.
        """
        # Extract formatted display from observation if available
        display_observation = observation

        if isinstance(observation, dict):
            # For FrozenLake and similar games, prefer grid_layout for display
            if "grid_layout" in observation:
                display_observation = observation["grid_layout"]
            # For other structured observations, try to extract meaningful display
            elif (
                "observation" in observation
                and observation["observation"] != "default_initial_state"
            ):
                display_observation = observation["observation"]
            # If we still have default_initial_state, try to use position info
            elif (
                observation.get("observation") == "default_initial_state"
                and "session_id" in observation
            ):
                # This is the fallback case - we should have gotten the proper initial state from MCP resources
                display_observation = f"Initial game state (Session: {observation['session_id']})\nWaiting for grid data from server..."

        return template.format(observation=display_observation, **context)

    async def close(self):
        """Closes all MCP sessions."""
        print(f"🧹 Closing {self.n} MCP sessions...")

        async def close_session(session: MCPSession):
            """Close a single MCP session in its own task context."""
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

        # Create individual tasks for each session close to match the creation pattern
        tasks = [
            asyncio.create_task(close_session(session)) for session in self.sessions
        ]

        if tasks:
            try:
                # Wait for all close operations to complete
                await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                # Handle cancellation gracefully (especially important for Python 3.12)
                logger.debug(
                    "Close operation was cancelled, but sessions are marked as closed"
                )

        print(f"✅ All MCP sessions closed.")


# Keep the old MCPVectorEnv for backward compatibility
MCPVectorEnv = GeneralMCPVectorEnv


class FireworksPolicy(PlaybackPolicyBase):
    """
    General Fireworks AI policy that works with ANY MCP environment via tool calling.

    Maintains conversation history per environment for proper OpenAI-style trajectories.
    NO environment-specific logic - everything comes from MCP tools and dataset prompts.

    Supports both live mode (using Fireworks LLM) and playback mode (replaying recorded trajectories).
    """

    def __init__(
        self,
        model_id: str,
        temperature: float = 0.2,
        deployment_type: str = "serverless",
        max_tokens: int = 4096,
        **kwargs,
    ):
        """
        Initialize general policy with automatic record/playback detection.

        Args:
            model_id: Fireworks model identifier (e.g., "accounts/fireworks/models/qwen3-235b-a22b")
            temperature: Sampling temperature (0.0 to 2.0)
            deployment_type: "serverless", "on-demand", or "auto"
            max_tokens: Maximum tokens to generate per request

        Automatic Mode Detection:
            - If REWARD_KIT_PLAYBACK_FILE is set and file exists: Playback mode
            - Otherwise: Recording mode
        """
        # Check for automatic playback mode
        playback_file = os.environ.get("REWARD_KIT_PLAYBACK_FILE")
        _playback_actions = None

        if playback_file and os.path.exists(playback_file):
            logger.info(f"🎬 Auto-detected playback mode: {playback_file}")
            _playback_actions = self._load_trajectory_file(playback_file)
            if not _playback_actions:
                logger.warning(
                    f"⚠️  Failed to load playback file, switching to recording mode"
                )
                _playback_actions = None
        elif playback_file:
            logger.info(
                f"📝 Auto-detected recording mode: {playback_file} (file will be created)"
            )

        # Initialize playback functionality
        super().__init__(_playback_actions=_playback_actions, **kwargs)

        # Store policy configuration
        self.model_id = model_id
        self.temperature = temperature
        self.deployment_type = deployment_type
        self.max_tokens = max_tokens

        # Initialize conversation state tracking for proper OpenAI trajectories
        self.conversation_histories = {}  # {env_index: [messages]}
        self.initialized = False

        # Only initialize Fireworks LLM in live mode (not in playback mode)
        if not self._is_playback:
            if not FIREWORKS_AVAILABLE:
                raise ImportError(
                    "The 'fireworks-ai' package is required for FireworksPolicy. "
                    "Please install it with 'pip install fireworks-ai'"
                )

            # Verify authentication
            api_key = get_fireworks_api_key()
            if not api_key:
                raise ValueError(
                    "FIREWORKS_API_KEY environment variable or ~/.fireworks/auth.ini file is required "
                    "to use FireworksPolicy. See the reward-kit documentation for setup instructions."
                )

            # Set the API key for the Fireworks SDK
            os.environ["FIREWORKS_API_KEY"] = api_key

            # Initialize the LLM instance using Build SDK
            try:
                self.llm = LLM(
                    model=self.model_id,
                    deployment_type=self.deployment_type,
                    temperature=self.temperature,
                )
                logger.info(
                    f"✅ Initialized Fireworks LLM: {self.model_id} ({self.deployment_type})"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize Fireworks LLM '{self.model_id}': {e}"
                )
        else:
            # In playback mode, skip expensive LLM initialization
            self.llm = None
            logger.info(
                f"🎬 Playback mode: Skipping Fireworks LLM initialization for performance"
            )

    def initialize_conversations(
        self, n_envs: int, system_prompts: List[str], initial_user_prompts: List[str]
    ):
        """Initialize conversation histories for each environment."""
        self.conversation_histories = {}
        for i in range(n_envs):
            self.conversation_histories[i] = [
                {"role": "system", "content": system_prompts[i]},
                {"role": "user", "content": initial_user_prompts[i]},
            ]
        self.initialized = True

    def add_tool_response(
        self, env_index: int, tool_call: MCPToolCall, tool_response: str
    ):
        """Add tool call and response to conversation history."""
        if env_index not in self.conversation_histories:
            return

        conversation = self.conversation_histories[env_index]

        # Find the most recent assistant message with tool calls to get the correct call_id
        call_id = None
        for i in range(len(conversation) - 1, -1, -1):
            if (
                conversation[i]["role"] == "assistant"
                and "tool_calls" in conversation[i]
            ):
                # Find the tool call that matches our tool_name
                for tc in conversation[i]["tool_calls"]:
                    if tc["function"]["name"] == tool_call.tool_name:
                        call_id = tc["id"]
                        break
                if call_id:
                    break

        # Fallback if no matching tool call found
        if not call_id:
            call_id = f"call_{env_index}_{len(conversation)}"

        # Add tool response (the assistant message was already added by _generate_tool_call_with_history)
        tool_message = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": tool_response,
        }
        conversation.append(tool_message)

    def log_conversation_state_for_playback(self, env_index: int, step: int):
        """
        Log the current conversation state in the format required for playback.

        Expected format: {"env_index": 0, "step": 0, "messages": [{..}, {..}]}

        Args:
            env_index: Environment index
            step: Current step number
        """
        # Use REWARD_KIT_PLAYBACK_FILE environment variable for recording
        playback_file = os.environ.get("REWARD_KIT_PLAYBACK_FILE")
        if not playback_file:
            return  # No recording file specified

        conversation = self.conversation_histories.get(env_index, [])
        if not conversation:
            return

        playback_entry = {
            "env_index": env_index,
            "step": step,
            "messages": conversation.copy(),
        }

        with open(playback_file, "a") as f:
            f.write(json.dumps(playback_entry) + "\n")

    async def _generate_live_tool_calls(
        self,
        tool_schemas: List[List[Dict]],
        observations: List[Any],
        system_prompts: List[str],
        user_prompts: List[str],
    ) -> List[MCPToolCall]:
        """
        Generate tool calls for all environments using Fireworks LLM in live mode.

        For first call: Initialize conversations with system + user prompts
        For subsequent calls: Use existing conversation history for continuity

        Args:
            tool_schemas: Available MCP tools for each environment [env][tool]
            observations: Current observations from each environment
            system_prompts: System prompts from dataset (environment-specific)
            user_prompts: Formatted user prompts for current state

        Returns:
            List of tool calls to execute via MCP protocol
        """
        if not observations:
            return []

        # Initialize conversations on first call
        if not self.initialized:
            self.initialize_conversations(
                len(observations), system_prompts, user_prompts
            )

        logger.debug(
            f"🤖 Generating tool calls for {len(observations)} environments using {self.model_id}"
        )

        # Make parallel API calls to Fireworks using conversation history
        tasks = []
        for i, tools in enumerate(tool_schemas):
            task = asyncio.create_task(self._generate_tool_call_with_history(tools, i))
            tasks.append(task)

        # Wait for all API calls to complete
        tool_calls = await asyncio.gather(*tasks, return_exceptions=True)

        # Process responses and handle exceptions
        result_calls = []
        for i, tool_call in enumerate(tool_calls):
            if isinstance(tool_call, Exception):
                logger.warning(
                    f"Tool call generation {i} failed: {tool_call}, using fallback"
                )
                # Use first available tool as fallback
                if tool_schemas[i]:
                    fallback_tool = tool_schemas[i][0]
                    fallback_call = MCPToolCall(
                        tool_name=fallback_tool["name"], arguments={}
                    )
                    result_calls.append(fallback_call)
                else:
                    logger.error(f"No tools available for environment {i}")
                    result_calls.append(MCPToolCall("unknown", {}))
            else:
                result_calls.append(tool_call)

        logger.debug(f"🎯 Generated {len(result_calls)} tool calls")
        return result_calls

    async def _generate_tool_call_with_history(
        self, tools: List[Dict], env_index: int
    ) -> MCPToolCall:
        """
        Generate a tool call using conversation history for proper OpenAI trajectories.

        Args:
            tools: Available MCP tools for this environment
            env_index: Environment index

        Returns:
            MCPToolCall object
        """
        try:
            # Get conversation history for this environment
            messages = self.conversation_histories.get(env_index, [])
            if not messages:
                raise RuntimeError(
                    f"No conversation history for environment {env_index}"
                )

            # Convert MCP tools to OpenAI format
            openai_tools = self._convert_mcp_tools_to_openai(tools)

            logger.debug(
                f"Environment {env_index} - Converted {len(tools)} MCP tools to {len(openai_tools)} OpenAI tools"
            )
            logger.debug(
                f"Environment {env_index} - Conversation length: {len(messages)} messages"
            )

            # Make API call with conversation history
            loop = asyncio.get_event_loop()
            current_request = {
                "messages": messages,
                "tools": openai_tools,
                # "tool_choice": "required" if openai_tools else None,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

            response = await loop.run_in_executor(
                None, lambda: self.llm.chat.completions.create(**current_request)
            )

            # ADD ASSISTANT MESSAGE TO ACTUAL CONVERSATION HISTORY
            # This is crucial for proper tool call ID management in add_tool_response
            assistant_message_for_history = {
                "role": "assistant",
                "content": response.choices[0].message.content,
            }

            # Add tool calls if present with the actual API response IDs
            if response.choices[0].message.tool_calls:
                assistant_message_for_history["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in response.choices[0].message.tool_calls
                ]

            # Add to actual conversation history
            messages.append(assistant_message_for_history)

            # Extract tool call from response
            message = response.choices[0].message
            logger.debug(f"Environment {env_index} - Response message: {message}")

            if message.tool_calls and len(message.tool_calls) > 0:
                tool_call = message.tool_calls[0]
                logger.debug(
                    f"Environment {env_index} - Using tool call: {tool_call.function.name}({tool_call.function.arguments})"
                )

                return MCPToolCall(
                    tool_name=tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )
            else:
                # Fallback if no tool calls
                logger.warning(
                    f"No tool calls in response for env {env_index}, message content: {message.content}"
                )
                return (
                    MCPToolCall(tools[0]["name"], {})
                    if tools
                    else MCPToolCall("unknown", {})
                )

        except Exception as e:
            logger.error(f"Fireworks API call failed for env {env_index}: {e}")
            raise e

    async def _generate_tool_call(
        self, system_prompt: str, user_prompt: str, tools: List[Dict], env_index: int
    ) -> MCPToolCall:
        """
        Generate a single tool call using Fireworks API with proper tool calling.

        Args:
            system_prompt: System prompt from dataset
            user_prompt: Formatted user prompt
            tools: Available MCP tools for this environment
            env_index: Environment index for logging

        Returns:
            MCPToolCall object
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Convert MCP tools to OpenAI tool format
        openai_tools = self._convert_mcp_tools_to_openai(tools)

        # Debug logging
        logger.debug(
            f"Environment {env_index} - Converted {len(tools)} MCP tools to {len(openai_tools)} OpenAI tools"
        )
        logger.debug(
            f"Environment {env_index} - OpenAI tools: {json.dumps(openai_tools, indent=2)}"
        )

        try:
            call_params = {
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }

            if openai_tools:
                call_params["tools"] = openai_tools
                # call_params["tool_choice"] = "required"

            logger.debug(
                f"Environment {env_index} - API call params: {json.dumps(call_params, indent=2)}"
            )

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self.llm.chat.completions.create(**call_params)
            )

            message = response.choices[0].message
            logger.debug(f"Environment {env_index} - Response message: {message}")
            logger.debug(f"Environment {env_index} - Tool calls: {message.tool_calls}")

            # Parse structured tool calls from response
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                logger.debug(
                    f"Environment {env_index} - Using tool call: {tool_call.function.name}({tool_call.function.arguments})"
                )
                return MCPToolCall(
                    tool_name=tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )
            else:
                # Fallback if no tool calls
                logger.warning(
                    f"No tool calls in response for env {env_index}, message content: {message.content}"
                )
                return (
                    MCPToolCall(tools[0]["name"], {})
                    if tools
                    else MCPToolCall("unknown", {})
                )

        except Exception as e:
            logger.error(f"Fireworks API call failed for env {env_index}: {e}")
            raise e

    def _convert_mcp_tools_to_openai(self, mcp_tools: List[Dict]) -> List[Dict]:
        """
        Convert MCP tool schemas to OpenAI function calling format.

        Args:
            mcp_tools: List of MCP tool definitions

        Returns:
            List of OpenAI-compatible tool definitions
        """
        openai_tools = []

        for mcp_tool in mcp_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": mcp_tool["name"],
                    "description": mcp_tool.get(
                        "description", f"Execute {mcp_tool['name']} action"
                    ),
                    "parameters": mcp_tool.get(
                        "input_schema",
                        {"type": "object", "properties": {}, "required": []},
                    ),
                },
            }
            openai_tools.append(openai_tool)

        return openai_tools


def make(
    env_spec: str,
    dataset: Optional[List[Dict]] = None,
    n: Optional[int] = None,
    seeds: Optional[List[int]] = None,
    model_id: str = "unknown",
    user_prompt_formatter: Optional[Callable] = None,
) -> "GeneralMCPVectorEnv":
    """
    Create general MCP environments driven by dataset configuration.

    Args:
        env_spec: MCP server URL
        dataset: List of dataset rows with prompts and context (preferred)
        n: Number of environments (for backward compatibility)
        seeds: List of seeds (for backward compatibility)
        model_id: Model identifier
        user_prompt_formatter: Optional callback for formatting user prompts

    Returns:
        General MCP environment that works with any MCP server

    Example:
        # New dataset-driven approach (preferred)
        dataset = load_jsonl("dataset.jsonl")
        envs = rk.make("http://localhost:8000/mcp", dataset=dataset)

        # Legacy approach (backward compatibility)
        envs = rk.make("http://localhost:8000/mcp", n=10, seeds=seeds)
    """
    # Parse environment specification - make sure URL format is correct
    base_url = env_spec
    if not base_url.startswith("http"):
        raise ValueError("Environment spec must be a valid HTTP URL")

    # Ensure we HAVE a trailing slash to avoid 307 redirects that break POST requests
    if not base_url.endswith("/"):
        base_url += "/"

    # Handle dataset-driven vs legacy approaches
    if dataset is not None:
        # New dataset-driven approach
        dataset_rows = []
        sessions = []

        for row in dataset:
            # Parse dataset row
            if isinstance(row, dict):
                # Handle seed from both old location (backward compatibility) and new location
                environment_context = row.get("environment_context", {})
                seed = row.get("seed")  # Check old location first
                if seed is None and "seed" in environment_context:
                    seed = environment_context["seed"]  # Check new location

                dataset_row = DatasetRow(
                    id=row["id"],
                    seed=seed,
                    system_prompt=row["system_prompt"],
                    user_prompt_template=row["user_prompt_template"],
                    environment_context=environment_context,
                )
            else:
                dataset_row = row  # Assume it's already a DatasetRow

            dataset_rows.append(dataset_row)

            # Create MCP session
            session = MCPSession(
                session_id=dataset_row.id,
                base_url=base_url,
                seed=dataset_row.seed,
                model_id=model_id,
                dataset_row=dataset_row,
            )
            sessions.append(session)

        return GeneralMCPVectorEnv(sessions, dataset_rows, user_prompt_formatter)

    else:
        # Legacy approach for backward compatibility
        if n is None:
            raise ValueError("Either 'dataset' or 'n' must be provided")

        # Generate seeds if not provided
        if seeds is None:
            import random

            seeds = [random.randint(0, 2**31 - 1) for _ in range(n)]
        elif len(seeds) != n:
            raise ValueError(f"Expected {n} seeds, got {len(seeds)}")

        # Create default dataset rows for legacy mode
        dataset_rows = []
        sessions = []

        for i in range(n):
            # Create a default dataset row (environment-agnostic)
            dataset_row = DatasetRow(
                id=f"session_{i}",
                seed=seeds[i],
                system_prompt="You are an AI agent interacting with an environment via available tools.",
                user_prompt_template="Current observation: {observation}. Use available tools to interact with the environment.",
                environment_context={},
            )
            dataset_rows.append(dataset_row)

            # Create MCP session
            session = MCPSession(
                session_id=f"session_{i}",
                base_url=base_url,
                seed=seeds[i],
                model_id=model_id,
                dataset_row=dataset_row,
            )
            sessions.append(session)

        return GeneralMCPVectorEnv(sessions, dataset_rows, user_prompt_formatter)


async def rollout(
    envs: Union[GeneralMCPVectorEnv, "MCPVectorEnv"],
    policy: Union[FireworksPolicy, Callable],
    steps: int = 512,
    openai_format_log_file: Optional[str] = None,
) -> List[Trajectory]:
    """
    Execute general rollouts using tool calling interface with automatic record/playback.

    This works with ANY MCP environment because:
    1. Policy receives tool schemas and makes tool calls
    2. Environment prompts come from dataset
    3. No hardcoded environment logic

    Args:
        envs: GeneralMCPVectorEnv instance
        policy: Policy that takes tool schemas, observations, prompts and returns tool calls
        steps: Maximum steps per rollout
        openai_format_log_file: Optional file to log clean OpenAI format for terminated trajectories only

    Environment Variable Control:
        REWARD_KIT_PLAYBACK_FILE: Controls record/playback mode
        - Not set: Normal live mode
        - Set but file doesn't exist: Record mode (file will be created)
        - Set and file exists: Playback mode (uses recorded data)

    Returns:
        List of Trajectory objects with complete rollout data

    Example:
        # Live mode
        trajectories = await rk.rollout(envs, policy)

        # Recording mode
        os.environ["REWARD_KIT_PLAYBACK_FILE"] = "record.jsonl"
        trajectories = await rk.rollout(envs, policy, openai_format_log_file="sft_data.jsonl")

        # Playback mode (after recording file exists)
        trajectories = await rk.rollout(envs, policy)
    """
    start_time = time.time()

    # Check for record/playback mode
    playback_file = os.environ.get("REWARD_KIT_PLAYBACK_FILE")
    recording_mode = playback_file and not os.path.exists(playback_file)
    playback_mode = playback_file and os.path.exists(playback_file)

    if recording_mode:
        logger.info(f"📝 Recording mode: Will record to {playback_file}")
    elif playback_mode:
        logger.info(f"🎬 Playback mode: Using recorded data from {playback_file}")
    else:
        logger.info(f"🚀 Live mode: No recording/playback")

    # Initialize OpenAI format logging for terminated trajectories only
    openai_logger = None
    if openai_format_log_file:
        # Clear the file at start
        with open(openai_format_log_file, "w") as f:
            pass
        openai_logger = lambda data: _log_openai_entry(openai_format_log_file, data)

    # Initialize trajectories
    trajectories = []
    for session in envs.sessions:
        trajectories.append(
            Trajectory(
                session=session,
                observations=[],
                actions=[],
                rewards=[],
                terminated=False,
                total_reward=0.0,
                steps=0,
                duration=0.0,
            )
        )

    # Reset environments and get initial state with tool discovery
    print(f"🔄 Resetting {envs.n} MCP environments...")
    current_observations, tool_schemas, system_prompts = await envs.reset()

    # Record initial observations
    for trajectory, obs in zip(trajectories, current_observations):
        trajectory.observations.append(obs)

    print(f"✅ Starting rollouts with {envs.n} environments for {steps} steps...")

    # Run rollout loop with tool calling
    for step in range(steps):
        step_start_time = time.time()

        # Format user prompts based on current observations (callback pattern)
        user_prompts = envs.format_user_prompts(current_observations)

        # Generate tool calls using general policy
        tool_calls = await policy(
            tool_schemas, current_observations, system_prompts, user_prompts
        )

        # Execute tool calls via MCP protocol
        observations, rewards, dones, infos = await envs.step(tool_calls)

        # Update conversation histories with tool responses (for proper OpenAI trajectories)
        if hasattr(policy, "add_tool_response"):
            for i, (tool_call, obs, reward, done) in enumerate(
                zip(tool_calls, observations, rewards, dones)
            ):
                # Convert observation to tool response format
                tool_response = json.dumps(obs) if isinstance(obs, dict) else str(obs)
                policy.add_tool_response(i, tool_call, tool_response)

                # Log conversation state for playback if in recording mode
                if recording_mode and hasattr(
                    policy, "log_conversation_state_for_playback"
                ):
                    policy.log_conversation_state_for_playback(i, step)

        # Update trajectories
        for i, (trajectory, obs, reward, done, info) in enumerate(
            zip(trajectories, observations, rewards, dones, infos)
        ):
            if not trajectory.terminated:
                trajectory.observations.append(obs)
                # Record the tool call as the action
                action_str = f"{tool_calls[i].tool_name}({tool_calls[i].arguments})"
                trajectory.actions.append(action_str)
                trajectory.rewards.append(reward)
                trajectory.total_reward += reward
                trajectory.steps += 1

                if done:
                    trajectory.terminated = True

                    # Log final OpenAI conversation for terminated trajectories only
                    if openai_logger and hasattr(policy, "conversation_histories"):
                        conversation = policy.conversation_histories.get(i, [])
                        if conversation:  # Only log if we have a conversation
                            openai_logger(
                                {
                                    "messages": conversation,
                                    "metadata": {
                                        "session_id": envs.sessions[i].session_id,
                                        "seed": envs.sessions[i].seed,
                                        "total_steps": trajectory.steps,
                                        "total_reward": trajectory.total_reward,
                                        "terminated": True,
                                        "success": reward > 0,
                                    },
                                }
                            )

        # Update current observations for next step
        current_observations = observations

        # Check if all environments are done
        if all(traj.terminated for traj in trajectories):
            print(f"🏁 All environments terminated at step {step + 1}")
            break

    # Calculate durations
    total_duration = time.time() - start_time
    for trajectory in trajectories:
        trajectory.duration = total_duration

    # Clean up
    await envs.close()

    successful = sum(1 for traj in trajectories if traj.total_reward > 0)
    print(f"📊 Rollout complete: {successful}/{len(trajectories)} reached goal")
    print(f"⏱️  Total duration: {total_duration:.2f}s")

    # Print log file locations if created
    if openai_format_log_file:
        print(f"💬 OpenAI format log: {openai_format_log_file}")
    if recording_mode:
        print(f"📝 Recorded trajectory: {playback_file}")

    return trajectories


def _log_trajectory_entry(log_file: str, data: Dict[str, Any]):
    """Helper function to log trajectory entries."""
    with open(log_file, "a") as f:
        f.write(json.dumps(data) + "\n")


def _log_openai_entry(log_file: str, data: Dict[str, Any]):
    """Helper function to log OpenAI format entries."""
    with open(log_file, "a") as f:
        f.write(json.dumps(data) + "\n")


async def test_mcp(base_url: str, seeds: List[int]) -> Dict[str, Any]:
    """
    Test function for validating MCP server as mentioned in north star document.

    Args:
        base_url: Base URL of MCP server (e.g., "http://localhost:8000/mcp")
        seeds: List of seeds to test

    Returns:
        Test results dictionary
    """
    print(f"🧪 Testing MCP server at {base_url} with {len(seeds)} seeds...")

    results = {"total_tests": len(seeds), "successful": 0, "failed": 0, "results": []}

    for seed in seeds:
        try:
            # Create single environment
            envs = make(base_url, n=1, seeds=[seed], model_id="test-model")

            # Simple policy for testing
            policy = FireworksPolicy("test-model")

            # Run short rollout
            trajectories = await rollout(envs, policy=policy, steps=10)

            if trajectories and len(trajectories[0].observations) > 1:
                results["successful"] += 1
                results["results"].append(
                    {
                        "seed": seed,
                        "status": "success",
                        "steps": trajectories[0].steps,
                        "total_reward": trajectories[0].total_reward,
                    }
                )
            else:
                results["failed"] += 1
                results["results"].append(
                    {"seed": seed, "status": "failed", "error": "empty_trajectory"}
                )

        except Exception as e:
            results["failed"] += 1
            results["results"].append(
                {"seed": seed, "status": "failed", "error": str(e)}
            )

    success_rate = results["successful"] / results["total_tests"] * 100
    print(
        f"✅ Test complete: {results['successful']}/{results['total_tests']} successful ({success_rate:.1f}%)"
    )

    return results


# Add to reward_kit.__init__.py exports
__all__ = [
    "make",
    "rollout",
    "FireworksPolicy",
    "MCPVectorEnv",
    "GeneralMCPVectorEnv",
    "MCPToolCall",
    "DatasetRow",
    "test_mcp",
]
