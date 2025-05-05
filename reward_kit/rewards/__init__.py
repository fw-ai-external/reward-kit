"""
Out-of-the-box reward functions for common use cases.
"""

# Import specific reward functions
from . import function_calling
from . import json_schema
from . import math
from . import code_execution
from . import format
from . import tag_count
from . import accuracy
from . import language_consistency
from . import reasoning_steps
from . import length
from . import repetition
from . import cpp_code
from . import accuracy_length

# Directly import specific reward functions for easy access
from .code_execution import fractional_code_reward
from .cpp_code import ioi_cpp_code_reward, binary_cpp_code_reward
from .accuracy_length import cosine_scaled_accuracy_length_reward

__all__ = [
    "function_calling", 
    "json_schema", 
    "math", 
    "code_execution", 
    "format", 
    "tag_count",
    "accuracy",
    "language_consistency",
    "reasoning_steps",
    "length",
    "repetition",
    "cpp_code",
    "accuracy_length",
    "fractional_code_reward",
    "ioi_cpp_code_reward",
    "binary_cpp_code_reward",
    "cosine_scaled_accuracy_length_reward"
]
