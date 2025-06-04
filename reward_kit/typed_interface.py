import inspect
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

from pydantic import TypeAdapter, ValidationError

# EvaluateResult and StepOutput are now extended/defined in models.py
from .models import (  # Removed StepOutput as it's not used here directly
    EvaluateResult,
    Message,
)

_single_res_adapter = TypeAdapter(EvaluateResult)
_list_res_adapter = TypeAdapter(List[EvaluateResult])


# Define a type for the mode parameter
EvaluationMode = Literal["pointwise", "batch"]

# TypeVar for the function being decorated, to preserve its signature as much as possible.
F = TypeVar("F", bound=Callable[..., Any])


def reward_function(
    _func: Optional[F] = None,
    *,
    mode: EvaluationMode = "pointwise",
    id: Optional[str] = None,
    requirements: Optional[List[str]] = None,
) -> Union[F, Callable[[F], F]]:
    """
    Decorator for user-defined reward and evaluation functions.

    It handles:
    - Coercing input messages (and ground truths if applicable) to Pydantic `Message` objects
      if the decorated function is type-hinted to receive them. This part currently targets
      parameters named 'messages' and 'ground_truth'.
    - Validating that the output conforms to `EvaluateResult` (for pointwise) or `List[EvaluateResult]` (for batch).

    Args:
        _func: The user's reward/evaluation function. Optional for decorator usage with args.
        mode: Specifies the operational mode. Defaults to "pointwise".
              - "pointwise": Function processes one rollout. Expected output: `EvaluateResult`.
              - "batch": Function processes a batch of rollouts. Expected output: `List[EvaluateResult]`.
        id: Optional identifier for the reward function, used for deployment
        requirements: Optional list of dependency requirements for deployment

    Returns:
        A decorator if `_func` is None, or the decorated function.
    """

    def decorator(func: F) -> F:
        sig = inspect.signature(func)
        params = sig.parameters

        @wraps(func)
        def wrapper(
            # The wrapper's signature should be generic enough to accept what the original func might take.
            # The specific coercion below targets 'messages' and 'ground_truth'.
            # For 'pointwise', 'messages' is expected. For 'batch', 'rollouts_messages' is more typical.
            # This simplified wrapper signature might need adjustment if we generalize input param coercion.
            *args: Any,
            **kwargs: Any,
        ) -> Union[EvaluateResult, List[EvaluateResult]]:  # Return type depends on mode

            # For now, we'll assume 'messages' is the primary input for pointwise,
            # and it's passed via kwargs or as the first arg if not named 'messages'.
            # This part needs to be more robust based on actual function signatures.
            # The current reverted logic specifically looks for 'messages' in params.

            # Extract 'messages' or 'rollouts_messages' from args/kwargs for processing
            # This is a simplification; a more robust solution would inspect `sig`
            # to find the correct parameter name based on `mode`.
            # For now, sticking to the structure of the reverted file which explicitly handles 'messages'.

            current_messages_arg_name = "messages"  # Default for pointwise
            if mode == "batch":
                # A common pattern for batch mode is 'rollouts_messages'
                # We'd need to find this in *args or **kwargs if not hardcoded.
                # For now, let's assume it's still passed as 'messages' for simplicity of this step,
                # or that the user function handles it. The key is output validation.
                pass

            # The reverted logic for input coercion:
            # It explicitly looks for 'messages' and 'ground_truth' in params.
            # We need to adapt how `processed_messages` and `kwargs` are prepared for `func` call.

            # Bind arguments to handle *args and **kwargs correctly for the wrapped function
            bound_args = sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()

            # Create a mutable copy of arguments to modify
            final_func_args = dict(bound_args.arguments)

            def _coerce_to_list_message(
                data_list: Any, arg_name_for_error: str
            ) -> List[Message]:
                if not isinstance(data_list, list):
                    raise TypeError(
                        f"Expected a list for '{arg_name_for_error}', got {type(data_list)}"
                    )
                typed_list = []
                for i, item_data in enumerate(data_list):
                    if isinstance(item_data, Message):
                        typed_list.append(item_data)
                    elif isinstance(item_data, dict):
                        typed_list.append(Message(**item_data))
                    else:
                        raise TypeError(
                            f"Unexpected type for item {i} in '{arg_name_for_error}': {type(item_data)}"
                        )
                return typed_list

            # 1. Conditional Pydantic conversion for 'messages' (pointwise) or 'rollouts_messages' (batch)
            if (
                mode == "pointwise"
                and "messages" in params
                and "messages" in final_func_args
            ):
                messages_param_annotation = params["messages"].annotation
                is_list_message_hint = False
                if get_origin(messages_param_annotation) in (list, List):
                    ann_args = get_args(messages_param_annotation)
                    if ann_args and ann_args[0] == Message:
                        is_list_message_hint = True

                if is_list_message_hint:
                    try:
                        final_func_args["messages"] = _coerce_to_list_message(
                            final_func_args["messages"], "messages"
                        )
                    except Exception as err:
                        raise ValueError(
                            f"Input 'messages' failed Pydantic validation: {err}"
                        ) from None

            elif (
                mode == "batch"
                and "rollouts_messages" in params
                and "rollouts_messages" in final_func_args
            ):
                param_annotation = params["rollouts_messages"].annotation
                is_list_list_msg_hint = False
                # Check if the annotation is List[List[Message]]
                # get_origin(typing.List[T]) is list.
                if get_origin(param_annotation) == list:
                    inner_list_type_args = get_args(param_annotation)
                    if inner_list_type_args and len(inner_list_type_args) == 1:
                        inner_list_type = inner_list_type_args[0]
                        if get_origin(inner_list_type) == list:
                            msg_type_args = get_args(inner_list_type)
                            if (
                                msg_type_args
                                and len(msg_type_args) == 1
                                and msg_type_args[0] == Message
                            ):
                                is_list_list_msg_hint = True

                if is_list_list_msg_hint:
                    try:
                        coerced_rollouts = []
                        for i, rollout_data in enumerate(
                            final_func_args["rollouts_messages"]
                        ):
                            coerced_rollouts.append(
                                _coerce_to_list_message(
                                    rollout_data, f"rollouts_messages[{i}]"
                                )
                            )
                        final_func_args["rollouts_messages"] = coerced_rollouts
                    except Exception as err:
                        raise ValueError(
                            f"Input 'rollouts_messages' failed Pydantic validation: {err}"
                        ) from None

            # 2. Conditional Pydantic conversion for 'ground_truth' (applies to both modes if present)
            # This logic might need to be mode-aware if ground_truth structure changes with mode (e.g. List[Any] for batch)
            if "ground_truth" in params and "ground_truth" in final_func_args:
                ground_truth_param_annotation = params["ground_truth"].annotation
                ground_truth_data = final_func_args["ground_truth"]

                is_list_message_gt_hint = False
                if get_origin(ground_truth_param_annotation) in (list, List):
                    ann_args = get_args(ground_truth_param_annotation)
                    if ann_args and ann_args[0] == Message:
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
                                typed_ground_truth_list.append(Message(**gt_item_data))
                            else:
                                raise TypeError(
                                    f"Unexpected type in ground_truth list: {type(gt_item_data)}"
                                )
                        final_func_args["ground_truth"] = typed_ground_truth_list
                    except Exception as err:
                        raise ValueError(
                            f"Input 'ground_truth' failed Pydantic validation for List[Message]: {err}"
                        ) from None

            # Call the author's function using the (potentially modified) arguments dictionary.
            # final_func_args should contain all parameters expected by func, correctly mapped.
            # Reconstruct args and kwargs for the call to func
            call_args = []
            call_kwargs = {}
            for (
                p_name,
                p_obj,
            ) in params.items():  # params from inspect.signature(func).parameters
                if p_obj.kind == inspect.Parameter.VAR_POSITIONAL:
                    # If original func had *pos_args, final_func_args might contain it as a tuple
                    call_args.extend(final_func_args.get(p_name, ()))
                elif p_obj.kind == inspect.Parameter.VAR_KEYWORD:  # **kwargs
                    # If original func had **kw_args, final_func_args contains the dict of these
                    call_kwargs.update(final_func_args.get(p_name, {}))
                elif p_name in final_func_args:  # Named parameters
                    if p_obj.kind == inspect.Parameter.POSITIONAL_ONLY:
                        call_args.append(final_func_args[p_name])
                    else:  # POSITIONAL_OR_KEYWORD, KEYWORD_ONLY
                        call_kwargs[p_name] = final_func_args[p_name]

            result = func(*call_args, **call_kwargs)

            # --- Output Validation ---
            try:
                if mode == "pointwise":
                    if isinstance(result, EvaluateResult):  # Already correct type
                        return result
                    return _single_res_adapter.validate_python(result)
                elif mode == "batch":
                    if isinstance(result, list) and all(
                        isinstance(item, EvaluateResult) for item in result
                    ):
                        return result
                    return _list_res_adapter.validate_python(result)
                else:  # Should not happen due to EvaluationMode typing
                    raise ValueError(
                        f"Internal error: Invalid mode '{mode}' in wrapper."
                    )
            except ValidationError as err:
                raise ValueError(
                    f"Return value from function '{func.__name__}' failed Pydantic validation for mode '{mode}':\n{err}"
                ) from None

        # Set id and requirements attributes
        wrapper._reward_function_id = id
        wrapper._reward_function_requirements = requirements

        return cast(F, wrapper)

    if (
        _func is None
    ):  # Decorator called with arguments, e.g., @reward_function(mode="batch")
        return decorator
    else:  # Decorator called without arguments, e.g., @reward_function (defaults to pointwise)
        return decorator(_func)
