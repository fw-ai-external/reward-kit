"""
Out-of-the-box reward functions for common use cases.
"""

# Import specific reward functions
from . import function_calling
from . import json_schema
from . import math
from . import advanced_math  # Add advanced_math import
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
from . import lean_prover
from . import deepcoder_reward # Added import
from . import multiple_choice_math_reward
from . import list_comparison_math_reward
from . import bfcl_reward # Import bfcl_reward

# Directly import specific reward functions for easy access
from .code_execution import fractional_code_reward
from .deepcoder_reward import deepcoder_code_reward # Added import
from .cpp_code import ioi_cpp_code_reward, binary_cpp_code_reward
from .accuracy_length import cosine_scaled_accuracy_length_reward
from .lean_prover import (
    lean_prover_reward,
    deepseek_prover_v2_reward,
    deepseek_huggingface_prover_benchmark,
)
from .multiple_choice_math_reward import multiple_choice_math_reward
from .list_comparison_math_reward import list_comparison_math_reward
from .bfcl_reward import bfcl_reward # Import bfcl_reward function

__all__ = [
    "function_calling",
    "json_schema",
    "math",
    "advanced_math",  # Add advanced_math to __all__
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
    "lean_prover",
    "deepcoder_reward", # Added module to __all__
    "multiple_choice_math_reward",
    "list_comparison_math_reward",
    "fractional_code_reward",
    "deepcoder_code_reward", # Added function to __all__
    "multiple_choice_math_reward", # Added function to __all__
    "list_comparison_math_reward", # Added function to __all__
    "ioi_cpp_code_reward",
    "binary_cpp_code_reward",
    "cosine_scaled_accuracy_length_reward",
    "lean_prover_reward",
    "deepseek_prover_v2_reward",
    "deepseek_huggingface_prover_benchmark",
    "bfcl_reward", # Add bfcl_reward to __all__
]
