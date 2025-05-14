"""
Fireworks Reward Kit - Simplify reward modeling for LLM RL fine-tuning.

A Python library for defining, testing, deploying, and using reward functions
for LLM fine-tuning, including launching full RL jobs on the Fireworks platform.

The library also provides an agent evaluation framework for testing and evaluating
tool-augmented models using self-contained task bundles.
"""

__version__ = "0.2.0"

# Import everything from models
from .models import (
    Message,
    MetricResult,
    EvaluateResult,
)

# Import from reward_function
from .reward_function import (
    RewardFunction,
    reward_function as legacy_reward_function,
)  # Deprecated

# Import the decorator from typed_interface (this is the one we want to expose)
from .typed_interface import reward_function

import warnings

# Show deprecation warnings
warnings.filterwarnings(
    "default", category=DeprecationWarning, module="reward_kit"
)

__all__ = [
    # Preferred interfaces
    "Message",
    "MetricResult",
    "EvaluateResult",
    "reward_function",  # New decorator from typed_interface
    "RewardFunction",
    # Deprecated (will be removed in a future version)
    "legacy_reward_function",
]
