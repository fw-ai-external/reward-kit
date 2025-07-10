"""
Session Management and Vector Environment

Handles MCPSession management and vector environment operations.
Extracted from mcp_env.py to improve modularity.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..client.connection import MCPConnectionManager
from ..types import DatasetRow, MCPSession, MCPToolCall

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages MCP sessions and their lifecycle."""

    def __init__(self):
        """Initialize the session manager."""
        self.connection_manager = MCPConnectionManager()

    async def initialize_sessions(self, sessions: List[MCPSession]) -> None:
        """
        Initialize multiple MCP sessions in parallel.

        Args:
            sessions: List of MCPSessions to initialize
        """
        tasks = [
            self.connection_manager.initialize_session(session) for session in sessions
        ]
        await asyncio.gather(*tasks)

    async def close_sessions(self, sessions: List[MCPSession]) -> None:
        """
        Close multiple MCP sessions in parallel.

        Args:
            sessions: List of MCPSessions to close
        """
        tasks = [
            asyncio.create_task(self.connection_manager.close_session(session))
            for session in sessions
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
        self.session_manager = SessionManager()

        if len(sessions) != len(dataset_rows):
            raise ValueError(
                f"Sessions ({len(sessions)}) and dataset rows ({len(dataset_rows)}) must have same length"
            )

    async def reset(self) -> Tuple[List[Any], List[List[Dict]], List[str]]:
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

        async def reset_session(session: MCPSession) -> Tuple[Any, List[Dict]]:
            # Establish a persistent session for each environment.
            await self.session_manager.connection_manager.initialize_session(session)

            # Get available tools from MCP server
            tool_schemas = await self.session_manager.connection_manager.discover_tools(
                session
            )

            # PROPER MCP PATTERN: Get initial state from resources during session establishment
            initial_observation = (
                await self.session_manager.connection_manager.get_initial_state(session)
            )

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
    ) -> Tuple[List[Any], List[float], List[bool], List[Dict]]:
        """
        Execute tool calls via MCP protocol using persistent sessions.

        Note: This uses MCP tools for actions/interactions during rollout.
        Initial state was obtained from MCP resources during reset() - different pattern.

        Args:
            tool_calls: Tool calls to execute in each environment

        Returns:
            observations, rewards, dones, infos
        """
        logger.info(
            f"🌐 STEP: ===== STARTING STEP WITH {len(tool_calls)} TOOL CALLS ====="
        )
        logger.info(
            f"🌐 STEP: Tool calls: {[(tc.tool_name, tc.arguments) if tc else None for tc in tool_calls]}"
        )

        if len(tool_calls) != self.n:
            raise ValueError(f"Expected {self.n} tool calls, got {len(tool_calls)}")

        async def step_session(session: MCPSession, tool_call: MCPToolCall):
            session_id = session.session_id
            logger.info(f"🔧 STEP: Processing session {session_id}")

            if session.terminated:
                logger.info(
                    f"🔧 STEP: Session {session_id} already terminated, returning last observation"
                )
                return session.last_observation, 0.0, True, {}

            # Handle None tool calls for terminated environments
            if tool_call is None:
                logger.info(
                    f"🔧 STEP: Session {session_id}: Received None tool call, marking as terminated"
                )
                session.terminated = True
                return session.last_observation, 0.0, True, {"none_tool_call": True}

            # Handle special playback termination signal
            if tool_call.tool_name == "_playback_terminate":
                logger.info(
                    f"🎬 STEP: Session {session_id}: Received playback termination signal"
                )
                session.terminated = True
                return (
                    session.last_observation,
                    0.0,
                    True,
                    {"playback_terminated": True},
                )

            # Handle special no-tool-call signal (episode ended, no action needed)
            if tool_call.tool_name == "_no_tool_call":
                logger.info(
                    f"🏁 STEP: Session {session_id}: No tool call generated, episode likely ended"
                )
                session.terminated = True
                return (
                    session.last_observation,
                    0.0,
                    True,
                    {
                        "no_tool_call": True,
                        "reason": tool_call.arguments.get("reason", "unknown"),
                    },
                )

            # ARCHITECTURAL FIX: Removed _terminate handling
            # Policies should NOT decide termination - this violates separation of concerns
            # Server/environment should handle hole/goal detection and termination

            # Execute the tool call via MCP protocol
            logger.info(
                f"🔧 STEP: Session {session_id}: Calling tool {tool_call.tool_name}({tool_call.arguments})"
            )
            logger.info(
                f"🔧 STEP: Session {session_id}: About to call connection_manager.call_tool"
            )
            observation, reward, done, info = (
                await self.session_manager.connection_manager.call_tool(
                    session, tool_call.tool_name, tool_call.arguments
                )
            )
            logger.info(f"🔧 STEP: Session {session_id}: Tool call completed")
            logger.info(
                f"🔧 STEP: Session {session_id}: Results - obs={observation}, reward={reward}, done={done}, info={info}"
            )

            # Update session state
            session.last_observation = observation
            session.terminated = done

            logger.info(
                f"🔧 STEP: Session {session_id}: Updated session state - terminated={session.terminated}"
            )
            return observation, reward, done, info

        # Execute steps in parallel using persistent sessions
        logger.info(
            f"🌐 STEP: Creating {len(self.sessions)} tasks for parallel execution"
        )
        tasks = [
            step_session(session, tool_call)
            for session, tool_call in zip(self.sessions, tool_calls)
        ]
        logger.info(f"🌐 STEP: About to execute {len(tasks)} tasks with asyncio.gather")
        results = await asyncio.gather(*tasks)
        logger.info(f"🌐 STEP: asyncio.gather completed, got {len(results)} results")

        observations, rewards, dones, infos = zip(*results)
        logger.info(f"🌐 STEP: Final results:")
        logger.info(f"  - Observations: {observations}")
        logger.info(f"  - Rewards: {rewards}")
        logger.info(f"  - Dones: {dones}")
        logger.info(f"  - Infos: {infos}")
        logger.info(f"🌐 STEP: ===== STEP COMPLETED =====")
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
        await self.session_manager.close_sessions(self.sessions)
        print(f"✅ All MCP sessions closed.")


# Keep the old MCPVectorEnv for backward compatibility
MCPVectorEnv = GeneralMCPVectorEnv
