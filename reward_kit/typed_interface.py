from functools import wraps
from typing import Any, Callable, Dict, List, TypeVar, cast, Protocol, Optional, Union

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
        self, messages: Union[List[Message], List[Dict[str, Any]]], **kwargs: Any
    ) -> EvaluateResult: ...


# Define return type protocol
class DictEvaluateFunction(Protocol):
    """Protocol for functions that take dict messages and return dict results."""

    def __call__(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> Dict[str, Dict[str, Any]]: ...


def reward_function(func: EvaluateFunction) -> DictEvaluateFunction:
    """
    Wrap an `evaluate`-style function so callers still use raw JSON-ish types.

    This decorator allows you to write evaluator functions with typed Pydantic models
    while maintaining backward compatibility with the existing API that uses lists
    of dictionaries.

    Args:
        func: Function that takes List[Message] and returns EvaluateResult

    Returns:
        Wrapped function that takes List[dict] and returns Dict[str, Dict[str, Any]]
    """

    @wraps(func)
    def wrapper(
        messages: Union[List[Dict[str, Any]], List[Message]], **kwargs: Any
    ) -> Dict[str, Dict[str, Any]]:
        # 1. Validate / coerce the incoming messages to list[Message]
        try:
            # Convert messages to Message objects if they're not already
            typed_messages = []

            for msg in messages:
                if isinstance(msg, Message):
                    # Already a Message object, use it directly
                    typed_messages.append(msg)
                else:
                    # It's a dictionary, convert to Message
                    role = msg.get("role", "")
                    content = msg.get("content", "")

                    if role == "system":
                        typed_messages.append(Message(role=role, content=content))
                    elif role == "user":
                        typed_messages.append(Message(role=role, content=content))
                    elif role == "assistant":
                        typed_messages.append(Message(role=role, content=content))
                    elif role == "tool":
                        typed_messages.append(
                            Message(
                                role=role,
                                content=content,
                                tool_call_id=msg.get("tool_call_id", ""),
                            )
                        )
                    elif role == "function":
                        typed_messages.append(
                            Message(role=role, content=content, name=msg.get("name", ""))
                        )
                    else:
                        # Unknown role type, convert as best we can
                        typed_messages.append(Message(**msg))
        except Exception as err:
            raise ValueError(f"Input messages failed validation:\n{err}") from None

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
            raise ValueError(f"Return value failed validation:\n{err}") from None

        # 3. Dump back to a plain dict for the outside world
        # Handle the updated EvaluateResult model structure
        if isinstance(result_model, EvaluateResult):
            # Build a response including all the metrics
            result_dict = {}
            
            # Add each metric to the result dictionary
            for key, metric in result_model.metrics.items():
                result_dict[key] = {
                    "success": metric.success,
                    "score": metric.score,
                    "reason": metric.reason,
                }
            
            # If there's an error, add it to the result
            if result_model.error:
                result_dict["error"] = {"error": result_model.error}
            
            return result_dict
        else:
            return _res_adapter.dump_python(result_model, mode="json")

    return cast(DictEvaluateFunction, wrapper)
