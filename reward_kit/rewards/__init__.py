"""
Out-of-the-box reward functions for common use cases.
"""

# Import specific reward functions
from . import bfcl_reward  # Import bfcl_reward
from . import deepcoder_reward  # Added import
from . import (
    accuracy,
    accuracy_length,
    code_execution,
    cpp_code,
    format,
    function_calling,
    json_schema,
    language_consistency,
    lean_prover,
    length,
    list_comparison_math_reward,
    math,
    multiple_choice_math_reward,
    reasoning_steps,
    repetition,
    tag_count,
)
from .accuracy_length import cosine_scaled_accuracy_length_reward

# Import function separately to avoid name conflict with the module
from .bfcl_reward import bfcl_reward as bfcl_reward_function

# Directly import specific reward functions for easy access
from .code_execution import fractional_code_reward
from .cpp_code import binary_cpp_code_reward, ioi_cpp_code_reward
from .deepcoder_reward import deepcoder_code_reward  # Added import
from .lean_prover import (
    deepseek_huggingface_prover_benchmark,
    deepseek_prover_v2_reward,
    lean_prover_reward,
)

# Import these with aliases to avoid name conflicts
from .list_comparison_math_reward import (
    list_comparison_math_reward as list_comparison_math_reward_function,
)
from .multiple_choice_math_reward import (
    multiple_choice_math_reward as multiple_choice_math_reward_function,
)

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
    "deepcoder_reward",  # Added module to __all__
    "multiple_choice_math_reward",
    "list_comparison_math_reward",
    "fractional_code_reward",
    "deepcoder_code_reward",  # Added function to __all__
    "multiple_choice_math_reward",  # Added module to __all__
    "multiple_choice_math_reward_function",  # Added function to __all__
    "list_comparison_math_reward",  # Added module to __all__
    "list_comparison_math_reward_function",  # Added function to __all__
    "ioi_cpp_code_reward",
    "binary_cpp_code_reward",
    "cosine_scaled_accuracy_length_reward",
    "lean_prover_reward",
    "deepseek_prover_v2_reward",
    "deepseek_huggingface_prover_benchmark",
    "bfcl_reward",  # Add bfcl_reward module to __all__
    "bfcl_reward_function",  # Add bfcl_reward function to __all__
]
