"""
Fireworks Reward Kit - Simplify reward modeling for LLM RL fine-tuning.

A Python library for defining, testing, deploying, and using reward functions
for LLM fine-tuning, including launching full RL jobs on the Fireworks platform.
"""

__version__ = "0.2.0"

from .models import RewardOutput, MetricRewardOutput, Message, MetricResult, EvaluateResult
from .reward_function import RewardFunction, reward_function
from .typed_interface import reward_function

__all__ = [
    # Original classes
    "RewardOutput",
    "MetricRewardOutput",
    "RewardFunction",
    "reward_function",
    
    # New typed interfaces
    "Message",
    "MetricResult",
    "EvaluateResult",
    "reward_function",
]