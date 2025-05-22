import json
from typing import Any, Dict, List, Optional, KeysView, Iterator

# Import OpenAI message types
# from openai.types.chat import ChatCompletionMessageParam # Unused import
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessageToolCall,
    FunctionCall,
)
from pydantic import BaseModel, Field


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
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any) -> "Message":
        if isinstance(obj, dict) and "role" not in obj:
            raise ValueError("Role is required")
        return super().model_validate(obj, *args, **kwargs)


class MetricResult(BaseModel):
    """Result of a single metric evaluation.

    Attributes:
        is_score_valid (bool): Whether the score is valid for this metric (required).
        score (float): The score for this metric.
        reason (str): Explanation for the score.
    """

    is_score_valid: bool = True
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str

    def __getitem__(self, key: str) -> Any:
        if (
            key in self.__fields__
        ):  # Changed to __fields__ for Pydantic v1 compatibility
            value = getattr(self, key)
            return value
        raise KeyError(f"'{key}'")

    def __contains__(self, key: str) -> bool:
        return key in self.__fields__  # Changed to __fields__

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> KeysView[str]:
        return self.__fields__.keys()  # Changed to __fields__

    def values(self) -> List[Any]:
        # For consistency with __getitem__ returning raw attribute values (including nested models)
        return [
            getattr(self, key) for key in self.__fields__.keys()
        ]  # Changed to __fields__

    def items(self) -> List[tuple[str, Any]]:
        return [
            (key, getattr(self, key)) for key in self.__fields__.keys()
        ]  # Changed to __fields__

    def __iter__(self) -> Iterator[str]:
        return iter(self.__fields__.keys())  # Changed to __fields__


class EvaluateResult(BaseModel):
    """The complete result of an evaluator with multiple metrics.

    Attributes:
        is_score_valid (bool): Whether the overall score is valid (required).
        score (float): The overall evaluation score.
        reason (Optional[str]): Optional explanation for the overall score.
        error (Optional[str]): Optional error message.
        metrics (Dict[str, MetricResult]): Dictionary of component metrics.
    """

    error: Optional[str] = None
    score: float = Field(..., ge=0.0, le=1.0)
    reason: Optional[str] = None
    is_score_valid: bool = True
    metrics: Dict[str, MetricResult]

    def __getitem__(self, key: str) -> Any:
        if key in self.__fields__:  # Changed to __fields__
            value = getattr(self, key)
            # If the value is a dict of MetricResult, and we want __getitem__ on metrics
            # to return a dict of dicts (rather than dict of MetricResult objects),
            # we'd need special handling here.
            # For now, return the raw attribute value, consistent with MetricResult.__getitem__
            return value
        raise KeyError(f"'{key}'")

    def __contains__(self, key: str) -> bool:
        return key in self.__fields__  # Changed to __fields__

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> KeysView[str]:
        return self.__fields__.keys()  # Changed to __fields__

    def values(self) -> List[Any]:
        # For consistency with __getitem__ returning raw attribute values
        return [
            getattr(self, key) for key in self.__fields__.keys()
        ]  # Changed to __fields__

    def items(self) -> List[tuple[str, Any]]:
        return [
            (key, getattr(self, key)) for key in self.__fields__.keys()
        ]  # Changed to __fields__

    def __iter__(self) -> Iterator[str]:
        return iter(self.__fields__.keys())  # Changed to __fields__


# Original dataclass-based models for backwards compatibility
# These are deprecated and will be removed in a future version
# Use EvaluateResult and MetricResult instead
# MetricRewardOutput and RewardOutput are fully removed.


# --- Models for New Agent Evaluation Framework (V2) ---


class EvaluationCriteriaModel(BaseModel):
    """
    Defines criteria for evaluating task success, often by querying the final state of a resource.
    """

    final_state_query: Optional[str] = Field(
        default=None,
        description="A query (e.g., SQL) to run on the final state of the resource.",
    )
    expected_query_result_transform: Optional[str] = Field(
        default=None,
        description="A Python lambda string (e.g., 'lambda x: x > 0') to transform and evaluate the query result to a boolean.",
    )

    # Explicit fields for ground truth data for BFCL evaluation
    ground_truth_function_calls: Optional[List[List[str]]] = Field(
        default=None, description="Ground truth function calls for BFCL evaluation."
    )
    ground_truth_comparable_state: Optional[Dict[str, Any]] = Field(
        default=None, description="Ground truth comparable state for BFCL evaluation."
    )

    # Future: Could include other complex evaluation logic or references


class TaskDefinitionModel(BaseModel):
    """
    Pydantic model for validating the structure of a V2 agent evaluation task definition file (YAML/JSON).
    """

    name: str = Field(description="Unique name for the task.")
    description: Optional[str] = Field(
        default=None, description="A brief description of the task."
    )

    resource_type: str = Field(
        description="The type of ForkableResource to use (e.g., 'SQLResource', 'PythonStateResource', 'FileSystemResource', 'DockerResource')."
    )
    base_resource_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration dictionary passed to the base resource's setup() method.",
    )

    tools_module_path: Optional[str] = Field(
        default=None,
        description="Optional Python import path to a module containing custom tool functions for this task.",
    )
    reward_function_path: str = Field(
        description="Python import path to the reward function (e.g., 'my_module.my_reward_func')."
    )

    goal_description: Optional[str] = Field(
        default=None,
        description="A human-readable description of the agent's goal for this task.",
    )
    evaluation_criteria: Optional[EvaluationCriteriaModel] = Field(
        default=None,
        description="Criteria used by the Orchestrator to determine if the primary goal was achieved.",
    )

    initial_user_prompt: Optional[str] = Field(
        default=None,
        description="The initial prompt or message to start the agent interaction. Deprecated if 'messages' field is used for multi-turn.",
    )
    messages: Optional[List[Dict[str, Any]]] = (
        Field(  # Explicit field for initial/multi-turn messages
            default=None,
            description="A list of messages to start the conversation, can represent multiple user turns for sequential processing.",
        )
    )

    # PoC / Task specific parameters
    poc_max_turns: int = Field(
        default=3,
        ge=1,
        description="For PoC Orchestrator, the maximum number of interaction turns.",
    )

    # Allow other custom fields to be captured if needed by specific tasks or resources
    # These will be accessible via `model_extra` if `model_config` has `extra = 'allow'`
    # Or define a specific field:
    # custom_task_params: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"  # Allow and capture extra fields not explicitly defined
        # For Pydantic v2, it's model_config = {"extra": "allow"}
        # Assuming Pydantic v1 style for now based on existing file, can update if needed.
        # If using Pydantic v2, this should be:
        # from pydantic import ConfigDict
        # model_config = ConfigDict(extra='allow')
        # For Pydantic v1, `Config.extra = "allow"` is correct.
