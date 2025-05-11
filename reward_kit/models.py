from typing import Dict, List, Optional, Any  # Union, Callable, Literal removed
import json
from pydantic import BaseModel, Field

# Import OpenAI message types
# from openai.types.chat import ChatCompletionMessageParam # Unused import
from openai.types.chat.chat_completion_message import (
    FunctionCall,
    ChatCompletionMessageToolCall,
)


# Create a Message class compatible with OpenAI's interface
class Message(BaseModel):
    """Chat message model compatible with OpenAI's interface."""

    role: str
    content: Optional[str] = (
        ""  # Content can be None for tool calls in OpenAI API
    )
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
    function_call: Optional[FunctionCall] = None

    # Model validators
    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        if isinstance(obj, dict) and "role" not in obj:
            raise ValueError("Role is required")
        return super().model_validate(obj, *args, **kwargs)


class MetricResult(BaseModel):
    """Result of a single metric evaluation."""

    success: Optional[bool] = None
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str

    def __getitem__(self, key: str) -> Any:
        if key in self.model_fields:
            value = getattr(self, key)
            return value
        raise KeyError(f"'{key}'")

    def __contains__(self, key: str) -> bool:
        return key in self.model_fields

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return self.model_fields.keys()

    def values(self):
        # For consistency with __getitem__ returning raw attribute values (including nested models)
        return [getattr(self, key) for key in self.model_fields.keys()]

    def items(self):
        return [(key, getattr(self, key)) for key in self.model_fields.keys()]

    def __iter__(self):
        return iter(self.model_fields.keys())


class EvaluateResult(BaseModel):
    """The complete result of an evaluator with multiple metrics."""

    error: Optional[str] = None
    score: float = Field(..., ge=0.0, le=1.0)
    reason: Optional[str] = None
    metrics: Dict[str, MetricResult]

    def __getitem__(self, key: str) -> Any:
        if key in self.model_fields:
            value = getattr(self, key)
            # If the value is a dict of MetricResult, and we want __getitem__ on metrics
            # to return a dict of dicts (rather than dict of MetricResult objects),
            # we'd need special handling here.
            # For now, return the raw attribute value, consistent with MetricResult.__getitem__
            return value
        raise KeyError(f"'{key}'")

    def __contains__(self, key: str) -> bool:
        return key in self.model_fields

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return self.model_fields.keys()

    def values(self):
        # For consistency with __getitem__ returning raw attribute values
        return [getattr(self, key) for key in self.model_fields.keys()]

    def items(self):
        return [(key, getattr(self, key)) for key in self.model_fields.keys()]

    def __iter__(self):
        return iter(self.model_fields.keys())


# Original dataclass-based models for backwards compatibility
# These are deprecated and will be removed in a future version
# Use EvaluateResult and MetricResult instead
# MetricRewardOutput and RewardOutput are fully removed.
