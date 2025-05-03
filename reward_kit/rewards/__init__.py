"""
Out-of-the-box reward functions for common use cases.
"""

# Import specific reward functions
from . import function_calling
from . import json_schema
from . import math

__all__ = ["function_calling", "json_schema", "math"]
