from functools import wraps
from typing import Any, Dict, List, TypeVar, cast, Protocol, Union

from pydantic import TypeAdapter, ValidationError

from .models import Message, EvaluateResult

# Create a type adapter that can handle OpenAI message types
_msg_adapter = TypeAdapter(List[Message])
_res_adapter = TypeAdapter(EvaluateResult)

T = TypeVar("T")


# Define protocol for more precise typing
class EvaluateFunction(Protocol):
    """Protocol for evaluate functions that take typed messages."""

    def __call__(
        self,
        messages: Union[List[Message], List[Dict[str, Any]]],
        **kwargs: Any,
    ) -> Union[EvaluateResult, Dict[str, Any]]: ...


# Define return type protocol
class DictEvaluateFunction(Protocol):
    """Protocol for functions that take dict messages and return dict results."""

    def __call__(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> Dict[str, Any]: ...


def reward_function(func: EvaluateFunction) -> DictEvaluateFunction:
    """
    Wrap an `evaluate`-style function so callers still use raw JSON-ish types.

    This decorator allows you to write evaluator functions with typed Pydantic models
    while maintaining backward compatibility with the existing API that uses lists
    of dictionaries.

    Args:
        func: Function that takes List[Message] and returns EvaluateResult

    Returns:
        Wrapped function that takes List[dict] and returns Dict[str, Any]
    """

    @wraps(func)
    def wrapper(
        messages: Union[List[Dict[str, Any]], List[Message]], **kwargs: Any
    ) -> Dict[str, Any]:
        # 1. Validate / coerce the incoming messages to list[Message]
        try:
            # Convert messages to Message objects if they're not already
            typed_messages = []

            for msg in messages:
                if isinstance(msg, Message):
                    # Already a Message object, use it directly
                    typed_messages.append(msg)
                else:
                    # It's a dictionary, validate and convert to Message
                    if "role" not in msg:
                        raise ValueError("Role is required in message")

                    role = msg.get("role", "")
                    content = msg.get(
                        "content", ""
                    )  # Default to empty string if None

                    # Common message parameters
                    message_params = {"role": role}

                    # Add content only if it exists (can be None for tool calls)
                    if "content" in msg:
                        message_params["content"] = (
                            content if content is not None else ""
                        )

                    # Add role-specific parameters
                    if role == "tool":
                        message_params["tool_call_id"] = msg.get(
                            "tool_call_id", ""
                        )
                        message_params["name"] = msg.get("name", "")
                    elif role == "function":
                        message_params["name"] = msg.get("name", "")
                    elif role == "assistant" and "tool_calls" in msg:
                        message_params["tool_calls"] = msg.get("tool_calls")

                    # Create the message object
                    typed_messages.append(Message(**message_params))
        except Exception as err:
            raise ValueError(
                f"Input messages failed validation:\n{err}"
            ) from None

        # 2. Call the author's function
        result = func(typed_messages, **kwargs)

        # Author might return EvaluateResult *or* a bare dict → coerce either way
        try:
            # If it's already an EvaluateResult, use it directly
            if isinstance(result, EvaluateResult):
                result_model = result
            else:
                # Otherwise validate it
                result_model = _res_adapter.validate_python(result)
        except ValidationError as err:
            raise ValueError(
                f"Return value failed validation:\n{err}"
            ) from None

        # 3. Dump back to a plain dict for the outside world
        # Handle the updated EvaluateResult model structure
        if isinstance(result_model, EvaluateResult):
            # Build a response including all the metrics
            return result_model.model_dump()
        else:
            return _res_adapter.dump_python(result_model, mode="json")

    return cast(DictEvaluateFunction, wrapper)
