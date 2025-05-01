"""
Fireworks Reward Kit - Simplify reward modeling for LLM RL fine-tuning.

A Python library for defining, testing, deploying, and using reward functions
for LLM fine-tuning, including launching full RL jobs on the Fireworks platform.
"""

__version__ = "0.2.0"

# Import everything from models
from .models import (
    RewardOutput,
    MetricRewardOutput,
    Message,
    MetricResult,
    EvaluateResult,
)

# Import from reward_function (will be renamed in a future version)
from .reward_function import RewardFunction, reward_function as legacy_reward_function

# Import the decorator from typed_interface (this is the one we want to expose)
from .typed_interface import reward_function

__all__ = [
    # Original classes
    "RewardOutput",
    "MetricRewardOutput",
    "RewardFunction",
    "legacy_reward_function",  # Legacy decorator (will be renamed)
    # Typed interfaces
    "Message",
    "MetricResult",
    "EvaluateResult",
    "reward_function",  # New decorator from typed_interface
]
