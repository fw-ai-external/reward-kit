from functools import wraps
from typing import Any, Callable, Dict, List, TypeVar, cast, Protocol, Optional

from pydantic import TypeAdapter, ValidationError

from .models import Message, EvaluateResult

_msg_adapter = TypeAdapter(List[Message])
_res_adapter = TypeAdapter(EvaluateResult)

T = TypeVar('T')

# Define protocol for more precise typing
class EvaluateFunction(Protocol):
    """Protocol for evaluate functions that take typed messages."""
    def __call__(self, messages: List[Message], **kwargs: Any) -> EvaluateResult: ...

# Define return type protocol
class DictEvaluateFunction(Protocol):
    """Protocol for functions that take dict messages and return dict results."""
    def __call__(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Dict[str, Any]]: ...


def reward_function(
    func: EvaluateFunction
) -> DictEvaluateFunction:
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
    def wrapper(messages: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Dict[str, Any]]:
        # 1. Validate / coerce the incoming list[dict] → list[Message]
        try:
            typed_messages = _msg_adapter.validate_python(messages)
        except ValidationError as err:
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
        # For RootModel, we need to access the .root attribute before dumping
        if isinstance(result_model, EvaluateResult):
            # Convert MetricResult objects to dicts
            return {
                key: {
                    "success": metric.success,
                    "score": metric.score,
                    "reason": metric.reason
                }
                for key, metric in result_model.root.items()
            }
        else:
            return _res_adapter.dump_python(result_model, mode="json")

    return cast(DictEvaluateFunction, wrapper)