import importlib
import importlib.util
import inspect
import logging
import os
import warnings
from functools import wraps
from typing import (  # Any, Type removed
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
    cast,
)

import requests

from .models import EvaluateResult, MetricResult
from .typed_interface import (  # Note: This is the new decorator, not the legacy one below
    reward_function,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type for reward function
T = TypeVar("T", bound=Callable[..., EvaluateResult])

# Show deprecation warning
# warnings.warn(
#     "RewardOutput and legacy_reward_function are deprecated and will be removed in a future version. "
#     "Use EvaluateResult and the reward_function decorator instead.",
#     DeprecationWarning,
#     stacklevel=2,
# )


class RewardFunction:
    """
    A wrapper for reward functions that allows them to be run locally or remotely.

    The RewardFunction class wraps a reward function (either a local function or a remote endpoint)
    and provides a unified interface for calling it. It supports:

    - Local functions (mode="local")
    - Remote endpoints (mode="remote")
    - Fireworks-hosted models (mode="fireworks_hosted")

    Args:
        func: The local function to use (for mode="local")
        func_path: A string path to a function (e.g., "module.submodule:function_name")
        mode: The mode of operation ("local", "remote", or "fireworks_hosted")
        endpoint: The URL of the remote endpoint (for mode="remote")
        model_id: The ID of the Fireworks-hosted model (for mode="fireworks_hosted")
        **kwargs: Additional keyword arguments to pass to the function
    """

    def __init__(
        self,
        func: Optional[Callable] = None,
        func_path: Optional[str] = None,
        mode: str = "local",
        endpoint: Optional[str] = None,
        name: Optional[str] = None,
        model_id: Optional[str] = None,
        **kwargs,
    ):
        self.mode = mode
        self.func = func
        self.func_path = func_path
        self.endpoint = endpoint
        self.name = name
        self.model_id = model_id
        self.kwargs = kwargs

        if mode == "local":
            if func is None and func_path is None:
                raise ValueError(
                    "Either 'func' or 'func_path' must be provided for local mode"
                )
            if func_path and func is None:
                self.func = self._load_function_from_path(func_path)
        elif mode == "remote":
            if endpoint is None and name is None:
                raise ValueError(
                    "Either 'endpoint' or 'name' must be provided for remote mode"
                )
            if name and endpoint is None:
                # Construct endpoint URL from name (in a real implementation,
                # this would fetch from the Fireworks API)
                self.endpoint = f"https://api.fireworks.ai/v1/reward/{name}"
        elif mode == "fireworks_hosted":
            if model_id is None:
                raise ValueError(
                    "'model_id' must be provided for fireworks_hosted mode"
                )
            # Construct endpoint for the Fireworks-hosted model
            self.endpoint = f"https://api.fireworks.ai/v1/models/{model_id}/reward"
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _load_function_from_path(self, func_path: str) -> Callable:
        """
        Load a function from a path string.

        Handles two formats:
        - 'module.path:function_name' - Module with colon separator
        - 'module.path.function_name' - Module with function as last component
        """
        # Check for the colon format first (preferred)
        if ":" in func_path:
            module_path, func_name = func_path.split(":", 1)

            try:
                module = importlib.import_module(module_path)
                func = getattr(module, func_name)
                return func
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"Failed to load function from path {func_path}: {str(e)}"
                )

        # Try dot notation format: module.path.function_name
        # This assumes the last component is the function name
        parts = func_path.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid func_path format: {func_path}, expected 'module.path:function_name' or 'module.path.function_name'"
            )

        module_path = ".".join(parts[:-1])
        func_name = parts[-1]

        try:
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            return func
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to load function from path {func_path}: {str(e)}"
            )

    def __call__(
        self,
        messages: List[Dict[str, str]],
        original_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> EvaluateResult:
        """
        Call the reward function with the provided messages.

        Args:
            messages: List of conversation messages, each with 'role' and 'content' keys
            original_messages: Original conversation messages (for context)
            **kwargs: Additional keyword arguments to pass to the function

        Returns:
            EvaluateResult object with score and metrics
        """
        if original_messages is None:
            original_messages = messages[:-1] if messages else []

        # Combine instance kwargs with call kwargs
        combined_kwargs = {**self.kwargs, **kwargs}

        if self.mode == "local":
            if self.func is None:
                raise ValueError("No function provided for local mode")

            # Call the local function
            try:
                result = self.func(
                    messages=messages,
                    original_messages=original_messages,
                    **combined_kwargs,
                )

                # Handle different result types
                if isinstance(result, EvaluateResult):
                    # Preferred return type
                    return result
                elif isinstance(result, tuple) and len(result) == 2:
                    # Handle legacy (score, components) tuple format
                    warnings.warn(
                        "Tuple return format is deprecated. Use EvaluateResult instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    score, components = result
                    # Convert to EvaluateResult
                    metrics = {
                        k: MetricResult(score=v, reason=f"{k} score", success=None)
                        for k, v in components.items()
                    }
                    return EvaluateResult(score=score, metrics=metrics)
                elif isinstance(result, dict) and "score" in result:
                    # Handle dictionary return format
                    warnings.warn(
                        "Dictionary return format is deprecated. Use EvaluateResult instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    # Convert to EvaluateResult
                    metrics = {}
                    if "metrics" in result:
                        for k, v in result["metrics"].items():
                            if isinstance(v, dict):
                                metrics[k] = MetricResult(
                                    score=v.get("score", 0.0),
                                    reason=v.get("reason", f"{k} score"),
                                    success=v.get("success", None),
                                )
                            else:
                                metrics[k] = MetricResult(
                                    score=float(v),
                                    reason=f"{k} score",
                                    success=None,
                                )
                    return EvaluateResult(
                        score=result["score"],
                        reason=result.get("reason"),
                        metrics=metrics,
                    )
                else:
                    raise TypeError(
                        f"Invalid return type from reward function: {type(result)}. "
                        f"Expected EvaluateResult or (float, Dict[str, float]) tuple."
                    )

            except Exception as e:
                logger.error(f"Error calling local reward function: {str(e)}")
                raise

        elif self.mode in ["remote", "fireworks_hosted"]:
            if self.endpoint is None:
                raise ValueError(f"No endpoint provided for {self.mode} mode")

            # Prepare the payload
            payload = {
                "messages": messages,
                "original_messages": original_messages,
                **combined_kwargs,
            }

            # Get API key
            api_key = os.environ.get("FIREWORKS_API_KEY")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}" if api_key else "",
            }

            try:
                response = requests.post(self.endpoint, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()

                # Convert the result to EvaluateResult
                if isinstance(result, dict) and "score" in result:
                    # Create metrics dictionary
                    metrics = {}
                    if "metrics" in result:
                        for k, v in result["metrics"].items():
                            if isinstance(v, dict):
                                metrics[k] = MetricResult(
                                    score=v.get("score", 0.0),
                                    reason=v.get("reason", f"{k} score"),
                                    success=v.get("success", None),
                                )
                            else:
                                metrics[k] = MetricResult(
                                    score=float(v),
                                    reason=f"{k} score",
                                    success=None,
                                )

                    return EvaluateResult(
                        score=result["score"],
                        reason=result.get("reason"),
                        metrics=metrics,
                    )
                else:
                    raise ValueError(f"Invalid response from remote endpoint: {result}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Error calling remote endpoint: {str(e)}")
                raise

        raise ValueError(f"Invalid mode: {self.mode}")

    def get_trl_adapter(self) -> Callable:
        """
        Create an adapter function for use with TRL library.

        The TRL library expects a function that takes batch inputs and returns a batch of reward values.
        This adapter handles:
        1. Batch of messages (List[List[Dict]]) and original messages (List[List[Dict]])
        2. Batch of texts (List[str]) for simpler cases

        Returns:
            A callable function compatible with TRL's expected signature for reward functions.
        """

        def adapter(
            prompts: List[List[Dict]], completions: List[str], **kwargs
        ) -> List[float]:
            """
            Adapter function compatible with TRL's reward function signature.

            Args:
                prompts: A batch of prompt message lists.
                         e.g., [[{'role':'system',...}, {'role':'user',...}], ...]
                completions: A batch of generated completion strings by the model.
                **kwargs: Additional keyword arguments passed by TRL, potentially including
                          ground truth data like 'solution'. TRL typically passes these
                          as lists matching the batch size.

            Returns:
                A list of float reward scores for the batch.
            """
            results = []
            batch_size = len(prompts)
            if batch_size != len(completions):
                raise ValueError("Batch size mismatch between prompts and completions.")

            # Extract potential ground truth solutions if available
            # TRL passes columns from the dataset that weren't removed.
            # We expect 'solution' based on our grpo_example.py setup.
            solutions = kwargs.get("solution", [None] * batch_size)
            if not isinstance(solutions, list) or len(solutions) != batch_size:
                logger.warning(
                    f"Expected 'solution' kwarg to be a list of size {batch_size}, but got {type(solutions)}. Ground truth might not be passed correctly."
                )
                solutions = [None] * batch_size  # Fallback

            for i in range(batch_size):
                # Construct the full message list for this sample
                completion_input = completions[i]
                actual_completion_str = ""

                if isinstance(completion_input, list):
                    if completion_input:  # If the list is not empty
                        first_element = completion_input[0]
                        if (
                            isinstance(first_element, dict)
                            and "content" in first_element
                            and isinstance(first_element.get("role"), str)
                            and first_element.get("role") == "assistant"
                        ):
                            # Expected structure: completions[i] = [{'role': 'assistant', 'content': 'str_content'}]
                            actual_completion_str = str(first_element["content"])
                            logger.debug(
                                f"Adapter: completions[{i}] is a list with an assistant message dict. Extracted content."
                            )
                        else:
                            logger.warning(
                                f"Adapter: completions[{i}] is a list, but its first element "
                                f"is not the expected assistant message dict or is malformed: {first_element}. "
                                f"Using str(first_element) as content."
                            )
                            actual_completion_str = str(
                                first_element
                            )  # Fallback: stringify the element
                    else:
                        logger.warning(
                            f"Adapter: completions[{i}] is an empty list. Using empty string for content."
                        )
                        actual_completion_str = ""
                elif isinstance(completion_input, str):
                    actual_completion_str = completion_input  # It's already a string
                else:
                    # Fallback for other types (e.g. a direct dict, though less likely given warnings)
                    logger.warning(
                        f"Adapter: completions[{i}] is of unexpected type: {type(completion_input)}. "
                        f"Attempting to stringify for content: {completion_input}"
                    )
                    actual_completion_str = str(completion_input)

                messages = prompts[i] + [
                    {"role": "assistant", "content": actual_completion_str}
                ]

                # Prepare kwargs for the underlying reward function call for this specific sample
                call_kwargs = {}
                current_solution = solutions[
                    i
                ]  # Get the solution for the current sample

                # --- DEBUG PRINT ---
                debug_solution_val_str = (
                    str(current_solution) if current_solution is not None else "None"
                )
                logger.debug(
                    f"Adapter loop i={i}, type(current_solution)={type(current_solution)}, value='{debug_solution_val_str[:100]}...'"
                )
                # --- END DEBUG PRINT ---

                if current_solution is not None:
                    # Ensure it's actually a string before passing, handle potential lists defensively
                    if isinstance(current_solution, list):
                        logger.warning(
                            f"Sample {i} solution is a list, attempting to use first element: {current_solution}"
                        )
                        if current_solution:  # If list is not empty
                            call_kwargs["solution"] = str(
                                current_solution[0]
                            )  # Convert first element to string
                        else:
                            call_kwargs["solution"] = None  # Treat empty list as None
                    else:
                        call_kwargs["solution"] = str(
                            current_solution
                        )  # Ensure it's a string

                # Add any other necessary kwargs extraction here if needed in the future

                try:
                    # Call the underlying RewardFunction instance (__call__)
                    # Pass the constructed messages and the extracted kwargs for this sample
                    result = self(
                        messages=messages,
                        # original_messages are implicitly handled by self() if needed,
                        # as it defaults to messages[:-1]
                        **call_kwargs,
                    )
                    # Handle both RewardOutput and EvaluateResult
                    score = result.score
                    results.append(score)
                except Exception as e:
                    logger.error(
                        f"Error processing sample {i} in TRL adapter: {str(e)}"
                    )
                    # Append a default low score (e.g., 0.0) on error
                    results.append(0.0)

            return results

        return adapter


# The legacy_reward_function decorator has been removed as it is no longer needed.
# Use the reward_function decorator from typed_interface instead.
#
# For deployment functionality, use the RewardFunction class or the deployment
# methods from the evaluation module directly.


# The alias below is removed to ensure that `from .typed_interface import reward_function`
# is the one used throughout the library, thus avoiding the deprecation warning
# when using the @reward_function decorator.
