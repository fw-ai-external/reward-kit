"""
Reward function for the Frozen Lake evaluation task.
"""

from typing import List
from reward_kit.typed_interface import reward_function
from reward_kit.models import EvaluateResult, MetricResult, Message


@reward_function
def frozen_lake_reward(messages: List[Message], **kwargs) -> EvaluateResult:
    """
    Evaluate the final message list for a success string in the Frozen Lake game.
    
    Args:
        messages: List of conversation messages
        **kwargs: Additional keyword arguments
        
    Returns:
        EvaluateResult with score 1.0 for success, 0.0 for failure
    """
    # Check if the last message (from the game) contains success indicators
    if not messages:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            metrics={"success": MetricResult(score=0.0, reason="No messages provided")}
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
        "you made it"
    ]
    
    failure_indicators = [
        "you lose",
        "game over",
        "you fell",
        "hole",
        "frozen"
    ]
    
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
    
    metrics = {
        "success": MetricResult(
            score=score,
            reason=reason,
            is_score_valid=True
        )
    }
    
    return EvaluateResult(score=score, reason=reason, metrics=metrics)