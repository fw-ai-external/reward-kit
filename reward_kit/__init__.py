"""
Fireworks Reward Kit - Simplify reward modeling for LLM RL fine-tuning.

A Python library for defining, testing, deploying, and using reward functions
for LLM fine-tuning, including launching full RL jobs on the Fireworks platform.
"""

__version__ = "0.1.0"

from .models import RewardOutput, MetricRewardOutput
from .reward_function import RewardFunction, reward_function

__all__ = [
    "RewardOutput",
    "MetricRewardOutput",
    "RewardFunction",
    "reward_function",
]