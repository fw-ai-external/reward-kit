"""
Fireworks Reward Kit - Simplify reward modeling for LLM RL fine-tuning.

A Python library for defining, testing, deploying, and using reward functions
for LLM fine-tuning, including launching full RL jobs on the Fireworks platform.

The library also provides an agent evaluation framework for testing and evaluating
tool-augmented models using self-contained task bundles.
"""

import warnings

from .models import EvaluateResult, Message, MetricResult
from .reward_function import RewardFunction
from .typed_interface import reward_function

warnings.filterwarnings("default", category=DeprecationWarning, module="reward_kit")

__all__ = [
    # Core interfaces
    "Message",
    "MetricResult",
    "EvaluateResult",
    "reward_function",
    "RewardFunction",
]

from . import _version

__version__ = _version.get_versions()["version"]
