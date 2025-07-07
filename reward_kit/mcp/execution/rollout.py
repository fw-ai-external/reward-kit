"""
Rollout Coordination

Handles the orchestration of complete rollouts using tool calling interface.
Extracted from mcp_env.py to improve modularity.
"""

import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from ..session.manager import SessionManager
from ..types import MCPToolCall, Trajectory

if TYPE_CHECKING:
    from ..session.manager import GeneralMCPVectorEnv
    from .policy import LLMBasePolicy

logger = logging.getLogger(__name__)


class RolloutManager:
    """Manages the execution of complete rollouts using tool calling interface."""

    def __init__(self, session_manager: SessionManager):
        """
        Initialize the rollout manager.

        Args:
            session_manager: The session manager to use for environment interactions
        """
        self.session_manager = session_manager

    async def execute_rollout(
        self,
        envs: "GeneralMCPVectorEnv",
        policy: Union["LLMBasePolicy", Callable],
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
            openai_logger = lambda data: self._log_openai_entry(
                openai_format_log_file, data
            )

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
                    tool_response = (
                        json.dumps(obs) if isinstance(obs, dict) else str(obs)
                    )
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

    def _log_openai_entry(self, log_file: str, data: Dict[str, Any]):
        """Helper function to log OpenAI format entries."""
        with open(log_file, "a") as f:
            f.write(json.dumps(data) + "\n")
