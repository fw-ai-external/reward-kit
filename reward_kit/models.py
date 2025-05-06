from typing import Dict, List, Optional, Any, Union, Callable, Literal
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import json
from pydantic import BaseModel, Field

# Import OpenAI message types
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message import FunctionCall, ChatCompletionMessageToolCall

# Create a Message class compatible with OpenAI's interface
class Message(BaseModel):
    """Chat message model compatible with OpenAI's interface."""
    role: str
    content: Optional[str] = ""  # Content can be None for tool calls in OpenAI API
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
    
    def to_reward_output(self) -> "RewardOutput":
        """Convert EvaluateResult to RewardOutput (for backwards compatibility)."""
        metrics = {
            k: MetricRewardOutput(
                score=v.score,
                reason=v.reason
            )
            for k, v in self.metrics.items()
        }
        return RewardOutput(score=self.score, metrics=metrics)


# Original dataclass-based models for backwards compatibility
# These are deprecated and will be removed in a future version
# Use EvaluateResult and MetricResult instead
@dataclass_json
@dataclass
class MetricRewardOutput:
    """
    Individual component metric for reward output.
    
    DEPRECATED: Use MetricResult instead.

    Args:
        score: The score value for this metric component
        reason: Optional explanation for why this score was assigned
    """

    score: float
    reason: Optional[str] = None


@dataclass_json
@dataclass
class RewardOutput:
    """
    Complete output from a reward function including overall score and component metrics.
    
    DEPRECATED: Use EvaluateResult instead.

    Args:
        score: The final aggregate reward score
        metrics: Dictionary of component metrics and their individual scores/reasons
    """

    score: float
    metrics: Dict[str, MetricRewardOutput] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert RewardOutput to a dictionary representation."""
        return {
            "score": self.score,
            "metrics": {
                k: {"score": v.score, "reason": v.reason}
                for k, v in self.metrics.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RewardOutput":
        """Create a RewardOutput from a dictionary representation."""
        metrics = {
            k: MetricRewardOutput(score=v.get("score", 0.0), reason=v.get("reason"))
            for k, v in data.get("metrics", {}).items()
        }
        return cls(score=data.get("score", 0.0), metrics=metrics)

    def __str__(self) -> str:
        """String representation of the reward output."""
        return json.dumps(self.to_dict(), indent=2)
        
    def to_evaluate_result(self) -> "EvaluateResult":
        """Convert RewardOutput to EvaluateResult."""
        metrics = {
            k: MetricResult(
                score=v.score, 
                reason=v.reason or "", 
                success=None
            )
            for k, v in self.metrics.items()
        }
        return EvaluateResult(
            score=self.score,
            reason=None,
            metrics=metrics
        )
