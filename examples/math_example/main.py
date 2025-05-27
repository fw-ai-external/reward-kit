"""
Example of a custom math evaluation function for Reward Kit.
This script defines a reward function that can be pointed to by the
`reward-kit run` CLI command.
"""

import logging
from typing import Any, Dict, List

from reward_kit import EvaluateResult, Message, reward_function
from reward_kit.rewards.math import math_reward  # Standard math reward

logger = logging.getLogger(__name__)


@reward_function
def evaluate(
    messages: List[Message],
    ground_truth: str,
    # Example of how custom params for this specific wrapper could be defined
    # custom_processing_param: bool = True,
    **kwargs: Any,  # Captures other params like tolerance, require_units from CLI/config
) -> EvaluateResult:
    """
    A wrapper around the standard math_reward.
    This demonstrates how a user might define their own reward function
    that could include pre/post-processing or call library functions.

    The `reward-kit run` command can be configured to use this function
    by setting `reward.function_path="examples.math_example.main.evaluate"`.

    Any parameters defined in `reward.params` in the Hydra config for `reward-kit run`
    will be passed as kwargs here (e.g., tolerance, require_units).
    """
    logger.info(f"Custom 'evaluate' function in examples.math_example.main called.")

    # Call the standard math_reward, passing through relevant kwargs
    # The math_reward function itself handles tolerance, require_units, etc. from kwargs.
    result = math_reward(messages=messages, ground_truth=ground_truth, **kwargs)

    return result
