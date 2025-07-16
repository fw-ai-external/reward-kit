"""
General Static Policy for MCP Environment Testing

This policy provides a deterministic, non-LLM action sequence for fast iteration
across different MCP environments. It can be configured with custom tool names
and action sequences.

This is useful for:
- Fast testing of multi-session functionality
- Debugging environment behavior
- Performance testing without LLM overhead
"""

import asyncio
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Union

# Import the base policy and types for proper recording functionality
from reward_kit.mcp.types import MCPToolCall
from reward_kit.playback_policy import PlaybackPolicyBase

logger = logging.getLogger(__name__)


class StaticPolicy(PlaybackPolicyBase):
    """
    Static policy that follows a predetermined action sequence.
    
    Can be configured for different environments with custom tool names and actions.
    """

    def __init__(
        self,
        tool_name: str,
        action_sequence: Optional[List[str]] = None,
        available_actions: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize static policy with recording/playback support.

        Args:
            tool_name: Name of the tool to call for actions (e.g., "lake_move", "lander_action")
            action_sequence: List of actions to execute. If None, uses a default sequence.
            available_actions: List of all available actions for this environment.
            **kwargs: Additional arguments passed to PlaybackPolicyBase
        """
        # Initialize parent class for recording/playback functionality
        super().__init__(**kwargs)

        self.tool_name = tool_name
        self.available_actions = available_actions or []
        
        # Set default action sequence if not provided
        if action_sequence is None:
            if self.available_actions:
                # Use first few actions as default sequence
                self.action_sequence = self.available_actions[:min(6, len(self.available_actions))]
            else:
                self.action_sequence = ["DEFAULT_ACTION"]
        else:
            self.action_sequence = action_sequence

        self.step_counts = {}  # Track step count per environment

        # Initialize conversation history management like LLMBasePolicy
        self.conversation_histories = {}  # {env_index: [messages]}
        self.initialized = False

    def initialize_conversations(
        self, n_envs: int, system_prompts: List[str]
    ):
        """Initialize conversation histories for each environment."""
        self.step_counts = {i: 0 for i in range(n_envs)}
        self.conversation_histories = {}
        for i in range(n_envs):
            self.conversation_histories[i] = [
                {"role": "system", "content": system_prompts[i]},
            ]
        self.initialized = True
        logger.info(f"🎯 Static policy initialized for {n_envs} environments")
        logger.info(f"🛠️  Tool name: {self.tool_name}")
        logger.info(f"📋 Action sequence: {self.action_sequence}")

    async def _generate_live_tool_calls(
        self,
        tool_schemas: List[List[Dict]],
        observations: List[Any],
        system_prompts: List[str],
        user_prompts: List[Union[str, Dict[str, Any]]],
    ) -> List[MCPToolCall]:
        """
        Generate tool calls in live mode using the static action sequence.

        This implements the abstract method from PlaybackPolicyBase.

        Args:
            tool_schemas: Available tools for each environment
            observations: Current observations from environments
            system_prompts: System prompts for each environment
            user_prompts: User prompts for each environment

        Returns:
            List of MCPToolCall objects for each environment
        """
        # Initialize conversations on first call
        if not self.initialized:
            self.initialize_conversations(
                len(observations), system_prompts
            )

        # Generate actions using the same logic as before, but now create conversation entries
        env_indices = list(range(len(observations)))
        results = []

        for i, env_idx in enumerate(env_indices):
            # Get current step count for this environment
            step_count = self.step_counts.get(env_idx, 0)

            # Determine action based on step count
            if step_count < len(self.action_sequence):
                action = self.action_sequence[step_count]
            else:
                # After sequence completes, repeat the last action
                action = self.action_sequence[-1]

            # Create tool call in MCPToolCall format
            tool_call = MCPToolCall(tool_name=self.tool_name, arguments={"action": action})

            results.append(tool_call)

            # Create assistant message with tool call for conversation history
            assistant_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{env_idx}_{step_count}",
                        "type": "function",
                        "function": {
                            "name": self.tool_name,
                            "arguments": json.dumps({"action": action}),
                        },
                    }
                ],
            }

            # Add to conversation history
            self.conversation_histories[env_idx].append(assistant_message)

            # Update step count
            self.step_counts[env_idx] = step_count + 1

            logger.debug(f"🎮 Env {env_idx} step {step_count}: {action}")

        return results

    def add_tool_response(
        self,
        env_index: int,
        tool_call: MCPToolCall,
        tool_response: str,
        reward: float = 0.0,
        terminated: bool = False,
        info: Dict[str, Any] = None,
    ):
        """Add tool call and response to conversation history for recording."""
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

        # Add tool response with control plane metadata
        tool_message = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": tool_response,
        }

        # Add control plane metadata if provided
        if reward != 0.0 or terminated or info:
            tool_message["metadata"] = {
                "reward": reward,
                "terminated": terminated,
                "info": info or {},
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

    @property
    def model_id(self) -> str:
        """Model identifier for static policy."""
        return f"static-policy-{self.tool_name}-v1"


class RandomPolicy(PlaybackPolicyBase):
    """
    Random policy that selects random actions.
    Useful for testing environment robustness.
    """

    def __init__(
        self,
        tool_name: str,
        available_actions: List[str],
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize random policy with recording/playback support.

        Args:
            tool_name: Name of the tool to call for actions
            available_actions: List of all available actions for this environment
            seed: Random seed for reproducibility
            **kwargs: Additional arguments passed to PlaybackPolicyBase
        """
        # Initialize parent class for recording/playback functionality
        super().__init__(**kwargs)

        self.tool_name = tool_name
        self.available_actions = available_actions
        self.random = random.Random(seed)

        # Initialize conversation history management
        self.conversation_histories = {}
        self.initialized = False

    def initialize_conversations(
        self, n_envs: int, system_prompts: List[str]
    ):
        """Initialize conversation histories for each environment."""
        self.conversation_histories = {}
        for i in range(n_envs):
            self.conversation_histories[i] = [
                {"role": "system", "content": system_prompts[i]},
            ]
        self.initialized = True
        logger.info(f"🎲 Random policy initialized for {n_envs} environments")
        logger.info(f"🛠️  Tool name: {self.tool_name}")
        logger.info(f"🎯 Available actions: {self.available_actions}")

    async def _generate_live_tool_calls(
        self,
        tool_schemas: List[List[Dict]],
        observations: List[Any],
        system_prompts: List[str],
        user_prompts: List[Union[str, Dict[str, Any]]],
    ) -> List[MCPToolCall]:
        """
        Generate random tool calls in live mode.

        Args:
            tool_schemas: Available tools for each environment
            observations: Current observations from environments
            system_prompts: System prompts for each environment
            user_prompts: User prompts for each environment

        Returns:
            List of MCPToolCall objects for each environment
        """
        # Initialize conversations on first call
        if not self.initialized:
            self.initialize_conversations(
                len(observations), system_prompts
            )

        results = []

        for i, obs in enumerate(observations):
            # Select random action
            action = self.random.choice(self.available_actions)

            # Create tool call
            tool_call = MCPToolCall(tool_name=self.tool_name, arguments={"action": action})

            results.append(tool_call)

            # Create assistant message with tool call for conversation history
            assistant_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{i}_{self.random.randint(1000, 9999)}",
                        "type": "function",
                        "function": {
                            "name": self.tool_name,
                            "arguments": json.dumps({"action": action}),
                        },
                    }
                ],
            }

            # Add to conversation history
            self.conversation_histories[i].append(assistant_message)

            logger.debug(f"🎲 Env {i}: {action}")

        return results

    def add_tool_response(
        self,
        env_index: int,
        tool_call: MCPToolCall,
        tool_response: str,
        reward: float = 0.0,
        terminated: bool = False,
        info: Dict[str, Any] = None,
    ):
        """Add tool call and response to conversation history for recording."""
        if env_index not in self.conversation_histories:
            return

        conversation = self.conversation_histories[env_index]

        # Find the most recent assistant message with tool calls
        call_id = None
        for i in range(len(conversation) - 1, -1, -1):
            if (
                conversation[i]["role"] == "assistant"
                and "tool_calls" in conversation[i]
            ):
                for tc in conversation[i]["tool_calls"]:
                    if tc["function"]["name"] == tool_call.tool_name:
                        call_id = tc["id"]
                        break
                if call_id:
                    break

        if not call_id:
            call_id = f"call_{env_index}_{len(conversation)}"

        # Add tool response with control plane metadata
        tool_message = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": tool_response,
        }

        # Add control plane metadata if provided
        if reward != 0.0 or terminated or info:
            tool_message["metadata"] = {
                "reward": reward,
                "terminated": terminated,
                "info": info or {},
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

    @property
    def model_id(self) -> str:
        """Model identifier for random policy."""
        return f"random-policy-{self.tool_name}-v1"


# Helper functions for creating environment-specific policies
def create_frozen_lake_static_policy(action_sequence: Optional[List[str]] = None, **kwargs) -> StaticPolicy:
    """Create a static policy configured for FrozenLake environment."""
    return StaticPolicy(
        tool_name="lake_move",
        action_sequence=action_sequence or ["RIGHT", "RIGHT", "RIGHT", "DOWN", "DOWN", "DOWN"],
        available_actions=["LEFT", "DOWN", "RIGHT", "UP"],
        **kwargs
    )


def create_lunar_lander_static_policy(action_sequence: Optional[List[str]] = None, **kwargs) -> StaticPolicy:
    """Create a static policy configured for LunarLander environment."""
    return StaticPolicy(
        tool_name="lander_action",
        action_sequence=action_sequence or ["NOTHING"],
        available_actions=["NOTHING", "FIRE_LEFT", "FIRE_MAIN", "FIRE_RIGHT"],
        **kwargs
    )


def create_frozen_lake_random_policy(seed: Optional[int] = None, **kwargs) -> RandomPolicy:
    """Create a random policy configured for FrozenLake environment."""
    return RandomPolicy(
        tool_name="lake_move",
        available_actions=["LEFT", "DOWN", "RIGHT", "UP"],
        seed=seed,
        **kwargs
    )


def create_lunar_lander_random_policy(seed: Optional[int] = None, **kwargs) -> RandomPolicy:
    """Create a random policy configured for LunarLander environment."""
    return RandomPolicy(
        tool_name="lander_action",
        available_actions=["NOTHING", "FIRE_LEFT", "FIRE_MAIN", "FIRE_RIGHT"],
        seed=seed,
        **kwargs
    )


# Test the policies
if __name__ == "__main__":

    async def test_frozen_lake_static_policy():
        """Test the frozen lake static policy."""
        print("🧪 Testing FrozenLake static policy...")
        policy = create_frozen_lake_static_policy()

        # Initialize
        policy.initialize_conversations(
            n_envs=2,
            system_prompts=["System prompt 1", "System prompt 2"],
        )

        # Generate actions for several steps
        for step in range(8):
            actions = await policy(
                tool_schemas=[[], []],
                observations=[None, None],
                system_prompts=["Test system prompt 1", "Test system prompt 2"],
                user_prompts=["Test user prompt 1", "Test user prompt 2"],
            )

            print(f"Step {step}:")
            for i, action in enumerate(actions):
                print(f"  Env {i}: {action.arguments['action']}")
        print()

    async def test_lunar_lander_static_policy():
        """Test the lunar lander static policy."""
        print("🧪 Testing LunarLander static policy...")
        policy = create_lunar_lander_static_policy()

        # Initialize
        policy.initialize_conversations(
            n_envs=2,
            system_prompts=["System prompt 1", "System prompt 2"],
        )

        # Generate actions for several steps
        for step in range(8):
            actions = await policy(
                tool_schemas=[[], []],
                observations=[None, None],
                system_prompts=["Test system prompt 1", "Test system prompt 2"],
                user_prompts=["Test user prompt 1", "Test user prompt 2"],
            )

            print(f"Step {step}:")
            for i, action in enumerate(actions):
                print(f"  Env {i}: {action.arguments['action']}")
        print()

    async def test_all_policies():
        """Test all policy types."""
        await test_frozen_lake_static_policy()
        await test_lunar_lander_static_policy()

    asyncio.run(test_all_policies()) 