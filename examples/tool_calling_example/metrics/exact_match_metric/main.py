"""
Metric folder wrapper for the built-in composite_function_call_reward.
"""

from typing import Any, Dict, List, Optional, Union

from reward_kit.models import (  # Ensure Message is imported if used in signature
    EvaluateResult,
    Message,
)

# Adjust path to import from the root of reward_kit if necessary,
# or rely on PYTHONPATH if examples are run from the root.
# Assuming this script will be called in an environment where reward_kit is importable.
from reward_kit.rewards.function_calling import (  # Changed import
    exact_tool_match_reward,
)
from reward_kit.typed_interface import reward_function


@reward_function
def evaluate_metric(  # This name is expected by the metric folder mechanism
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[Dict[str, Any]] = None,
    **kwargs
) -> EvaluateResult:
    """
    This metric directly calls the built-in exact_tool_match_reward.
    """
    return exact_tool_match_reward(  # Changed to call exact_tool_match_reward directly
        messages=messages, ground_truth=ground_truth, **kwargs
    )
