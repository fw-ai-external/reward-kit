"""
Simple Deterministic Policy for MCP-Gym Testing

This policy follows a predetermined action sequence to test multi-step trajectories
without LLM complexity. It's designed to help debug the session management issue
by isolating the problem from model inference.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from ...playback_policy import PlaybackPolicyBase
from ..types import MCPToolCall

logger = logging.getLogger(__name__)


class SimpleDeterministicPolicy(PlaybackPolicyBase):
    """
    Simple deterministic policy that follows a predetermined action sequence.

    This policy is designed to help debug the session management issue in MCP-Gym
    by removing LLM complexity and focusing on the core rollout mechanics.
    """

    def __init__(
        self,
        action_sequence: List[str] = None,
        model_id: str = "deterministic-policy",
        **kwargs,
    ):
        """
        Initialize the deterministic policy.

        Args:
            action_sequence: List of actions to execute in sequence (default: DOWN, DOWN, DOWN, RIGHT, RIGHT, RIGHT)
            model_id: Model identifier for compatibility
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
        # Use RIGHT 3 times, then DOWN 3 times as requested
        # This follows path: 0 -> 1 -> 2 -> 3 -> 7 -> 11 -> 15 (goal)
        self.action_sequence = action_sequence or [
            "RIGHT",
            "RIGHT",
            "RIGHT",
            "DOWN",
            "DOWN",
            "DOWN",
        ]

        # Track state per environment
        self.env_step_counters = {}  # env_index -> step_count
        # TODO: Remove local state tracking - should query control plane resources instead
        # self.env_terminated = {}  # env_index -> bool (INCORRECT - should query control://status)
        self.conversation_histories = {}  # {env_index: [messages]} for compatibility
        self.initialized = False

    def initialize_conversations(
        self, n_envs: int, system_prompts: List[str], initial_user_prompts: List[str]
    ):
        """Initialize conversation histories for each environment."""
        self.conversation_histories = {}
        self.env_step_counters = {}
        # TODO: Remove local state tracking - should query control plane resources instead
        # self.env_terminated = {}  # INCORRECT - should query control://status

        for i in range(n_envs):
            self.conversation_histories[i] = [
                {"role": "system", "content": system_prompts[i]},
                {"role": "user", "content": initial_user_prompts[i]},
            ]
            self.env_step_counters[i] = 0
            # TODO: Remove local state tracking - should query control plane resources instead
            # self.env_terminated[i] = False  # INCORRECT - should query control://status

        self.initialized = True
        logger.info(f"🎯 Initialized deterministic policy for {n_envs} environments")

    def add_tool_response(
        self, env_index: int, tool_call: MCPToolCall, tool_response: str
    ):
        """Add tool call and response to conversation history."""
        if env_index not in self.conversation_histories:
            return

        conversation = self.conversation_histories[env_index]

        # Add assistant message with tool call
        assistant_message = {
            "role": "assistant",
            "content": f"Taking action: {tool_call.tool_name}({tool_call.arguments})",
            "tool_calls": [
                {
                    "id": f"call_{env_index}_{len(conversation)}",
                    "type": "function",
                    "function": {
                        "name": tool_call.tool_name,
                        "arguments": json.dumps(tool_call.arguments),
                    },
                }
            ],
        }
        conversation.append(assistant_message)

        # ARCHITECTURAL FIX: Policy should NOT add control plane metadata
        # The server should provide control plane metadata directly in tool responses
        # Policy only records conversation history
        tool_message = {
            "role": "tool",
            "tool_call_id": f"call_{env_index}_{len(conversation) - 1}",
            "content": tool_response,
        }

        conversation.append(tool_message)

    def log_conversation_state_for_playback(self, env_index: int, step: int):
        """Log the current conversation state for playback recording."""
        playback_file = os.environ.get("REWARD_KIT_PLAYBACK_FILE")
        if not playback_file:
            return

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
        Generate deterministic tool calls based on the action sequence.

        This is the core method that implements the deterministic behavior.
        """
        if not observations:
            return []

        logger.info(
            f"🎯 Generating deterministic tool calls for {len(observations)} environments"
        )

        tool_calls = []

        for env_index, (tools, obs) in enumerate(zip(tool_schemas, observations)):
            # Initialize step counter if not exists
            if env_index not in self.env_step_counters:
                self.env_step_counters[env_index] = 0

            current_step = self.env_step_counters[env_index]

            # ARCHITECTURAL FIX: Policy should NEVER decide termination
            # The server/environment adapter should handle hole/goal detection and termination
            # Policy only generates movement actions based on action sequence

            # Extract position for logging only (not for termination decisions)
            position = None
            if isinstance(obs, dict) and "position" in obs:
                position = obs["position"]
            elif isinstance(obs, (int, float)):
                position = int(obs)
            elif (
                isinstance(obs, dict)
                and "result" in obs
                and isinstance(obs["result"], dict)
            ):
                position = obs["result"].get("position")

            if position is not None:
                logger.info(
                    f"🎯 Environment {env_index} at position {position} (step {current_step})"
                )
            else:
                logger.warning(
                    f"⚠️ Environment {env_index}: Could not extract position from observation: {obs}"
                )

            # Get the action for this step (cycle through the sequence)
            action = self.action_sequence[current_step % len(self.action_sequence)]

            # Find the appropriate tool (should be 'lake_move' for FrozenLake)
            tool_name = None
            for tool in tools:
                if tool.get("name") == "lake_move":
                    tool_name = "lake_move"
                    break

            if tool_name:
                tool_call = MCPToolCall(
                    tool_name=tool_name, arguments={"action": action}
                )
                tool_calls.append(tool_call)

                logger.info(
                    f"🎯 Environment {env_index}, Step {current_step}: {tool_name}({action})"
                )
            else:
                # Fallback if no matching tool found
                logger.warning(
                    f"⚠️ No 'lake_move' tool found for environment {env_index}"
                )
                if tools:
                    fallback_tool = tools[0]
                    tool_call = MCPToolCall(
                        tool_name=fallback_tool["name"], arguments={"action": action}
                    )
                    tool_calls.append(tool_call)
                else:
                    tool_calls.append(MCPToolCall("unknown", {"action": action}))

            # Increment step counter
            self.env_step_counters[env_index] += 1

        logger.info(f"🎯 Generated {len(tool_calls)} deterministic tool calls")
        return tool_calls

    def check_termination_from_response(self, env_index: int, tool_response: str):
        """
        DEPRECATED: Policy should NOT check termination from responses.

        This method violates the architectural principle that policies should only generate actions.
        Termination detection should be handled by the server/environment adapter.
        """
        logger.warning(
            "⚠️ DEPRECATED: Policy should not check termination. Server should handle this."
        )
        return False

    async def __call__(
        self,
        tool_schemas: List[List[Dict]],
        observations: List[Any],
        system_prompts: List[str],
        user_prompts: List[str],
    ) -> List[MCPToolCall]:
        """
        Main policy call method with conversation history management.
        """
        # Initialize conversations if not already done
        if not hasattr(self, "initialized") or not self.initialized:
            self.initialize_conversations(
                len(observations), system_prompts, user_prompts
            )
        # BUGFIX: Remove the extra user prompt addition that causes incorrect conversation flow
        # The user prompts should only be added during initialization, not on subsequent calls
        # This was causing the problematic flow: system → user → assistant → tool → user → assistant → tool → ...
        # Correct flow should be: system → user → assistant → tool → assistant → tool → ...

        # NOTE: Removed the following problematic code:
        # else:
        #     # Add new user prompts to conversation history for subsequent calls
        #     for i, user_prompt in enumerate(user_prompts):
        #         if i in self.conversation_histories:
        #             self.conversation_histories[i].append({"role": "user", "content": user_prompt})

        if self._is_playback:
            # In playback mode, use recorded messages
            tool_calls = []
            n_envs = len(tool_schemas)

            for env_index in range(n_envs):
                messages = self._get_playback_messages(env_index)

                if messages is None:
                    tool_calls.append(
                        MCPToolCall(
                            "_playback_terminate",
                            {"reason": "no_more_recorded_actions"},
                        )
                    )
                    logger.info(
                        f"🎬 Environment {env_index}: No more recorded actions, signaling termination"
                    )
                    continue

                # Store the recorded messages in conversation history
                self.conversation_histories[env_index] = messages.copy()

                # Extract tool call from recorded messages
                tool_call = self._extract_tool_call_from_messages(messages, env_index)
                tool_calls.append(tool_call)

            return tool_calls
        else:
            # Live mode - use deterministic policy
            return await self._generate_live_tool_calls(
                tool_schemas, observations, system_prompts, user_prompts
            )

    def is_playback_mode(self) -> bool:
        """Check if policy is in playback mode."""
        return self._is_playback
