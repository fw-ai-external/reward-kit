import inspect  # Added for signature inspection
from functools import wraps
from typing import Any, Dict, List, Protocol, TypeVar, Union, cast, get_args, get_origin

from pydantic import TypeAdapter, ValidationError

from .models import EvaluateResult, Message  # EvaluateResult is now the hybrid model

_res_adapter = TypeAdapter(EvaluateResult)
# _msg_adapter is not used. T is not used.


# Define protocol for more precise typing
class EvaluateFunction(Protocol):
    """Protocol for evaluate functions that take typed messages."""

    def __call__(
        self, *args: Any, **kwargs: Any
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
    ) -> EvaluateResult:  # Changed return type

        sig = inspect.signature(func)
        params = sig.parameters

        processed_messages = messages  # Default to original messages

        # 1. Conditional Pydantic conversion for 'messages'
        if "messages" in params:
            messages_param_annotation = params["messages"].annotation
            # Check if the annotation is List[Message]
            # Handles List[Message] and typing.List[Message]
            is_list_message_hint = False
            if get_origin(messages_param_annotation) in (list, List):
                args = get_args(messages_param_annotation)
                if args and args[0] == Message:
                    is_list_message_hint = True

            if is_list_message_hint:
                try:
                    typed_messages_list = []
                    for msg_data in messages:
                        if isinstance(msg_data, Message):
                            typed_messages_list.append(msg_data)
                        elif isinstance(msg_data, dict):
                            # Simplified conversion, assuming valid dict structure for Message
                            # Full validation as before can be reinstated if needed
                            typed_messages_list.append(Message(**msg_data))
                        else:
                            # Handle unexpected item type in messages list
                            raise TypeError(
                                f"Unexpected type in messages list: {type(msg_data)}"
                            )
                    processed_messages = typed_messages_list
                except Exception as err:
                    raise ValueError(
                        f"Input 'messages' failed Pydantic validation: {err}"
                    ) from None

        # 2. Conditional Pydantic conversion for 'ground_truth'
        if "ground_truth" in params and "ground_truth" in kwargs:
            ground_truth_param_annotation = params["ground_truth"].annotation
            ground_truth_data = kwargs["ground_truth"]

            # Check if the annotation is List[Message]
            is_list_message_gt_hint = False
            if get_origin(ground_truth_param_annotation) in (list, List):
                args = get_args(ground_truth_param_annotation)
                if args and args[0] == Message:
                    is_list_message_gt_hint = True

            if is_list_message_gt_hint and ground_truth_data is not None:
                if not isinstance(ground_truth_data, list):
                    raise TypeError(
                        f"'ground_truth' expected a list for List[Message] hint, got {type(ground_truth_data)}"
                    )
                try:
                    typed_ground_truth_list = []
                    for gt_item_data in ground_truth_data:
                        if isinstance(gt_item_data, Message):
                            typed_ground_truth_list.append(gt_item_data)
                        elif isinstance(gt_item_data, dict):
                            # Simplified conversion
                            typed_ground_truth_list.append(Message(**gt_item_data))
                        else:
                            raise TypeError(
                                f"Unexpected type in ground_truth list: {type(gt_item_data)}"
                            )
                    kwargs["ground_truth"] = typed_ground_truth_list
                except Exception as err:
                    raise ValueError(
                        f"Input 'ground_truth' failed Pydantic validation for List[Message]: {err}"
                    ) from None
            # (Optional: Add handling for single Message hint if that's ever re-introduced)

        # 3. Call the author's function with processed inputs
        result = func(processed_messages, **kwargs)

        # 4. Author might return EvaluateResult *or* a bare dict → coerce either way
        try:
            # If it's already an EvaluateResult, use it directly
            if isinstance(result, EvaluateResult):
                result_model = result
            else:
                # Otherwise validate it
                result_model = _res_adapter.validate_python(result)
        except ValidationError as err:
            raise ValueError(f"Return value failed validation:\n{err}") from None

        # 3. Return the EvaluateResult object directly
        # The result_model is an instance of our hybrid EvaluateResult
        return result_model

    return cast(HybridEvaluateFunction, wrapper)
