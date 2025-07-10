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
            logger.info(f"🔄 ROLLOUT: ===== STARTING STEP {step} =====")

            # Early termination check - prevent tool call generation for already terminated environments
            if all(traj.terminated for traj in trajectories):
                logger.info(
                    f"🏁 ROLLOUT: All environments already terminated before step {step} (control plane signals)"
                )
                print(
                    f"🏁 All environments already terminated before step {step} (control plane signals)"
                )
                break

            # Filter active (non-terminated) environments to avoid unnecessary LLM calls
            active_env_indices = [
                i for i, traj in enumerate(trajectories) if not traj.terminated
            ]

            if not active_env_indices:
                logger.info(
                    f"🏁 ROLLOUT: No active environments remaining at step {step}"
                )
                print(f"🏁 No active environments remaining at step {step}")
                break

            logger.info(
                f"📊 ROLLOUT: Step {step}: {len(active_env_indices)}/{len(trajectories)} active, avg reward: {sum(t.total_reward for t in trajectories)/len(trajectories):.2f}"
            )
            logger.info(f"📊 ROLLOUT: Active env indices: {active_env_indices}")
            logger.info(
                f"📊 ROLLOUT: Trajectory states: {[(i, traj.terminated, traj.total_reward, traj.steps) for i, traj in enumerate(trajectories)]}"
            )
            print(
                f"📊 Step {step}: {len(active_env_indices)}/{len(trajectories)} active, avg reward: {sum(t.total_reward for t in trajectories)/len(trajectories):.2f}"
            )

            # Only generate tool calls for active environments
            logger.info(
                f"🔧 ROLLOUT: Preparing tool call generation for {len(active_env_indices)} active environments"
            )
            active_tool_schemas = [tool_schemas[i] for i in active_env_indices]
            active_observations = [current_observations[i] for i in active_env_indices]
            active_system_prompts = [system_prompts[i] for i in active_env_indices]

            logger.info(f"🔧 ROLLOUT: Active observations: {active_observations}")

            # Format user prompts only for active environments
            logger.info(
                f"🔧 ROLLOUT: Formatting user prompts for {len(active_observations)} active environments"
            )
            active_user_prompts = envs.format_user_prompts(active_observations)
            logger.info(f"🔧 ROLLOUT: Active user prompts: {active_user_prompts}")

            # Generate tool calls only for active environments using general policy
            logger.info(
                f"🤖 ROLLOUT: Calling policy for {len(active_env_indices)} active environments"
            )
            logger.info(f"🤖 ROLLOUT: Policy type: {type(policy)}")
            active_tool_calls = await policy(
                active_tool_schemas,
                active_observations,
                active_system_prompts,
                active_user_prompts,
            )
            logger.info(
                f"🤖 ROLLOUT: Policy returned {len(active_tool_calls)} tool calls"
            )
            logger.info(
                f"🤖 ROLLOUT: Tool calls: {[(tc.tool_name, tc.arguments) if tc else None for tc in active_tool_calls]}"
            )

            # Create full tool_calls list with None for terminated environments
            tool_calls = [None] * len(trajectories)
            for i, active_idx in enumerate(active_env_indices):
                tool_calls[active_idx] = active_tool_calls[i]
            logger.info(
                f"🔧 ROLLOUT: Full tool_calls list: {[(tc.tool_name, tc.arguments) if tc else None for tc in tool_calls]}"
            )

            # Execute tool calls via MCP protocol (now with control plane separation)
            logger.info(f"🌐 ROLLOUT: Executing tool calls via MCP protocol")
            logger.info(
                f"🌐 ROLLOUT: envs.step() called with {len(tool_calls)} tool calls"
            )
            observations, rewards, dones, infos = await envs.step(tool_calls)
            logger.info(f"🌐 ROLLOUT: envs.step() returned:")
            logger.info(f"  - Observations: {observations}")
            logger.info(f"  - Rewards: {rewards}")
            logger.info(f"  - Dones: {dones}")
            logger.info(f"  - Infos: {infos}")

            # Update conversation histories with tool responses (for proper OpenAI trajectories)
            logger.info(f"💬 ROLLOUT: Updating conversation histories")
            if hasattr(policy, "add_tool_response"):
                logger.info(f"💬 ROLLOUT: Policy has add_tool_response method")
                for i, (tool_call, obs, reward, done) in enumerate(
                    zip(tool_calls, observations, rewards, dones)
                ):
                    logger.info(
                        f"💬 ROLLOUT: Processing env {i}: tool_call={tool_call}, reward={reward}, done={done}"
                    )
                    # Skip adding tool responses for terminated environments (tool_call is None)
                    if tool_call is None:
                        logger.info(f"💬 ROLLOUT: Skipping env {i} (tool_call is None)")
                        continue

                    # Convert observation to tool response format
                    tool_response = (
                        json.dumps(obs) if isinstance(obs, dict) else str(obs)
                    )
                    logger.info(
                        f"💬 ROLLOUT: Adding tool response for env {i}: {tool_response[:100]}..."
                    )
                    policy.add_tool_response(i, tool_call, tool_response)
                    logger.info(
                        f"💬 ROLLOUT: Successfully added tool response for env {i}"
                    )

                    # Log conversation state for playback if in recording mode
                    if recording_mode and hasattr(
                        policy, "log_conversation_state_for_playback"
                    ):
                        logger.info(
                            f"📝 ROLLOUT: Logging conversation state for playback, env {i}, step {step}"
                        )
                        policy.log_conversation_state_for_playback(i, step)
                        logger.info(
                            f"📝 ROLLOUT: Successfully logged conversation state for env {i}"
                        )
            else:
                logger.info(
                    f"💬 ROLLOUT: Policy does not have add_tool_response method"
                )

            # Update trajectories with both data and control plane information
            logger.info(f"🗂️  ROLLOUT: Updating trajectories with step {step} data")
            for i, (trajectory, obs, reward, done, info) in enumerate(
                zip(trajectories, observations, rewards, dones, infos)
            ):
                logger.info(
                    f"🗂️  ROLLOUT: Processing trajectory {i}: terminated={trajectory.terminated}, reward={reward}, done={done}"
                )
                if not trajectory.terminated:
                    logger.info(f"🗂️  ROLLOUT: Updating active trajectory {i}")

                    # Record data plane (observation)
                    trajectory.observations.append(obs)

                    # Record action (tool call)
                    if tool_calls[i] is not None:
                        action_str = (
                            f"{tool_calls[i].tool_name}({tool_calls[i].arguments})"
                        )
                        trajectory.actions.append(action_str)
                        logger.info(
                            f"🗂️  ROLLOUT: Recorded action for trajectory {i}: {action_str}"
                        )
                    else:
                        logger.info(
                            f"🗂️  ROLLOUT: No tool call for trajectory {i} (already terminated)"
                        )

                    # Record control plane (reward/termination)
                    trajectory.rewards.append(reward)
                    trajectory.total_reward += reward
                    trajectory.steps += 1
                    logger.info(
                        f"🗂️  ROLLOUT: Updated trajectory {i}: steps={trajectory.steps}, total_reward={trajectory.total_reward}"
                    )

                    # Enhanced trajectory recording with control plane info
                    if not hasattr(trajectory, "control_plane_steps"):
                        trajectory.control_plane_steps = []

                    control_plane_step = {
                        "step": step,
                        "reward": reward,
                        "terminated": done,
                        "info": info.get("control_plane", {}),
                        "tool_call": (
                            action_str if tool_calls[i] is not None else "no_tool_call"
                        ),
                    }
                    trajectory.control_plane_steps.append(control_plane_step)
                    logger.info(
                        f"🗂️  ROLLOUT: Added control plane step for trajectory {i}: {control_plane_step}"
                    )

                    # Use control plane information for termination decision
                    if done:
                        logger.info(
                            f"🏁 ROLLOUT: Trajectory {i} terminated by control plane signal"
                        )
                        trajectory.terminated = True

                        # Add final control plane summary
                        if not hasattr(trajectory, "control_plane_summary"):
                            trajectory.control_plane_summary = {}

                        trajectory.control_plane_summary.update(
                            {
                                "total_reward": trajectory.total_reward,
                                "termination_reason": "control_plane_signal",
                                "final_step": step,
                                "control_plane_source": info.get("control_plane", {}),
                            }
                        )
                        logger.info(
                            f"🏁 ROLLOUT: Added control plane summary for trajectory {i}: {trajectory.control_plane_summary}"
                        )

                        # Log final OpenAI conversation for terminated trajectories only
                        if openai_logger and hasattr(policy, "conversation_histories"):
                            conversation = policy.conversation_histories.get(i, [])
                            if conversation:  # Only log if we have a conversation
                                logger.info(
                                    f"📝 ROLLOUT: Logging final conversation for terminated trajectory {i}"
                                )
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
                                            "control_plane_summary": trajectory.control_plane_summary,
                                        },
                                    }
                                )
                else:
                    logger.info(
                        f"🗂️  ROLLOUT: Skipping already terminated trajectory {i}"
                    )

            # Update current observations for next step
            logger.info(f"🔄 ROLLOUT: Updating current observations for next step")
            current_observations = observations
            logger.info(f"🔄 ROLLOUT: Updated observations: {current_observations}")

            # Check if all environments are done (using control plane termination)
            logger.info(f"🏁 ROLLOUT: Checking if all environments are done")
            active_count = sum(1 for traj in trajectories if not traj.terminated)
            logger.info(f"🏁 ROLLOUT: Active trajectories remaining: {active_count}")
            if all(traj.terminated for traj in trajectories):
                logger.info(f"🏁 ROLLOUT: All environments terminated, breaking loop")
                print(
                    f"🏁 All environments terminated at step {step + 1} (control plane signals)"
                )
                break

            logger.info(f"🔄 ROLLOUT: ===== COMPLETED STEP {step} =====")
            step_duration = time.time() - step_start_time
            logger.info(f"🔄 ROLLOUT: Step {step} took {step_duration:.2f}s")

            # Progress logging with control plane info
            active_envs = sum(1 for traj in trajectories if not traj.terminated)
            if step % 10 == 0 and active_envs > 0:
                avg_reward = sum(traj.total_reward for traj in trajectories) / len(
                    trajectories
                )
                print(
                    f"📊 Step {step}: {active_envs}/{len(trajectories)} active, avg reward: {avg_reward:.2f}"
                )

        # Calculate durations
        total_duration = time.time() - start_time
        for trajectory in trajectories:
            trajectory.duration = total_duration

        # Clean up
        await envs.close()

        # Enhanced reporting with control plane info
        successful = sum(1 for traj in trajectories if traj.total_reward > 0)
        terminated_by_control_plane = sum(
            1
            for traj in trajectories
            if hasattr(traj, "control_plane_summary")
            and traj.control_plane_summary.get("termination_reason")
            == "control_plane_signal"
        )

        print(f"📊 Rollout complete: {successful}/{len(trajectories)} reached goal")
        print(
            f"🎛️  Control plane terminations: {terminated_by_control_plane}/{len(trajectories)}"
        )
        print(f"⏱️  Total duration: {total_duration:.2f}s")

        # Print log file locations if created
        if openai_format_log_file:
            print(f"💬 OpenAI format log: {openai_format_log_file}")
        if recording_mode:
            print(f"📝 Recorded trajectory: {playback_file}")
            # Add note about control plane separation
            print(f"🎛️  Trajectories include control plane separation")

        return trajectories

    def _log_openai_entry(self, log_file: str, data: Dict[str, Any]):
        """Helper function to log OpenAI format entries."""
        with open(log_file, "a") as f:
            f.write(json.dumps(data) + "\n")
