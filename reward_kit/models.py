from typing import Dict, List, Optional, Any  # Union, Callable, Literal removed
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
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


class EvaluateResult(BaseModel):
    """The complete result of an evaluator with multiple metrics."""

    error: Optional[str] = None
    score: float = Field(..., ge=0.0, le=1.0)
    reason: Optional[str] = None
    metrics: Dict[str, MetricResult]


# Original dataclass-based models for backwards compatibility
# These are deprecated and will be removed in a future version
# Use EvaluateResult and MetricResult instead
# MetricRewardOutput and RewardOutput are fully removed.
