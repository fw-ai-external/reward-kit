"""
Reward function for the Frozen Lake evaluation task.
"""

from typing import List, Optional

from reward_kit.models import EvaluateResult, Message, MetricResult, StepOutput
from reward_kit.typed_interface import reward_function


@reward_function
def frozen_lake_reward(messages: List[Message], state=None, **kwargs) -> EvaluateResult:
    """
    Evaluate the final message list for a success string in the Frozen Lake game.

    Args:
        messages: List of conversation messages
        state: State dictionary containing trajectory data
        **kwargs: Additional keyword arguments

    Returns:
        EvaluateResult with score 1.0 for success, 0.0 for failure
    """
    # Check if the last message (from the game) contains success indicators
    if not messages:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            metrics={"success": MetricResult(score=0.0, reason="No messages provided")},
        )

    last_message = messages[-1]
    content = last_message.content.lower()

    # Check for success indicators in the message content
    success_indicators = [
        "you win",
        "you reached the goal",
        "congratulations",
        "success",
        "goal reached",
        "you made it",
    ]

    failure_indicators = ["you lose", "game over", "you fell", "hole", "frozen"]

    is_success = any(indicator in content for indicator in success_indicators)
    is_failure = any(indicator in content for indicator in failure_indicators)

    # Determine the score
    if is_success:
        score = 1.0
        reason = "Successfully reached the goal in Frozen Lake"
    elif is_failure:
        score = 0.0
        reason = "Failed to reach the goal (fell into hole or other failure)"
    else:
        # If no clear success/failure indicator, check if game is still ongoing
        score = 0.0
        reason = "Game outcome unclear or still in progress"

    metrics = {"success": MetricResult(score=score, reason=reason, is_score_valid=True)}

    # Extract trajectory data if available
    step_outputs = None
    if state and "successful_func_calls" in state:
        successful_calls = state["successful_func_calls"]
        step_outputs = []

        # Convert function calls to StepOutput format
        step_index = 0
        for turn_calls in successful_calls:
            for call in turn_calls:
                # Extract action from function call arguments
                action = call.get("args", {}).get("action", "unknown")
                step_outputs.append(
                    StepOutput(
                        step_index=step_index,
                        action=action,
                        base_reward=(
                            0.1 if action != "unknown" else 0.0
                        ),  # Small reward for valid actions
                        reason=f"Agent took action: {action}",
                    )
                )
                step_index += 1

    return EvaluateResult(
        score=score, reason=reason, metrics=metrics, step_outputs=step_outputs
    )
