from functools import wraps
from typing import Any, Dict, List, TypeVar, cast, Protocol, Union

from pydantic import TypeAdapter, ValidationError

from .models import Message, EvaluateResult # EvaluateResult is now the hybrid model

_res_adapter = TypeAdapter(EvaluateResult)
# _msg_adapter is not used. T is not used.


# Define protocol for more precise typing
class EvaluateFunction(Protocol):
    """Protocol for evaluate functions that take typed messages."""

    def __call__(
        self,
        messages: Union[List[Message], List[Dict[str, Any]]],
        **kwargs: Any,
    ) -> Union[EvaluateResult, Dict[str, Any]]: ...


# Define return type protocol for the wrapped function
class HybridEvaluateFunction(Protocol):
    """
    Protocol for functions that take a list of dictionaries (JSON-like messages)
    and return an EvaluateResult object (which is now a hybrid Pydantic/dict-like model).
    """
    def __call__(
        self, messages: Union[List[Dict[str, Any]], List[Message]], **kwargs: Any
    ) -> EvaluateResult: ...


def reward_function(func: EvaluateFunction) -> HybridEvaluateFunction:
    """
    Wrap an `evaluate`-style function. It coerces raw JSON-ish input messages
    to Pydantic `Message` objects for the wrapped function and ensures the output
    is an `EvaluateResult` object.

    The returned `EvaluateResult` object is a hybrid model that supports both
    Pydantic attribute access (e.g., result.score) and dictionary-style
    access (e.g., result['score']).

    Args:
        func: A function that accepts `List[Message]` (or `List[Dict]`) and
              returns an `EvaluateResult` instance or a dictionary that can be
              coerced into one.

    Returns:
        A wrapped function that takes `List[Dict[str, Any]]` (or `List[Message]`)
        and returns an `EvaluateResult` object.
    """

    @wraps(func)
    def wrapper(
        messages: Union[List[Dict[str, Any]], List[Message]], **kwargs: Any
    ) -> EvaluateResult: # Changed return type
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

        # 3. Return the EvaluateResult object directly
        # The result_model is an instance of our hybrid EvaluateResult
        return result_model

    return cast(HybridEvaluateFunction, wrapper)
