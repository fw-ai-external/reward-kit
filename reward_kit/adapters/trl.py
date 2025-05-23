"""
Adapters for integrating reward-kit with TRL trainers.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from reward_kit.models import (  # Assuming Message accessible
    Message,
)
from reward_kit.typed_interface import (  # For type hinting the reward_fn
    EvaluateFunction,
)

logger = logging.getLogger(__name__)


def create_trl_adapter(
    reward_fn: EvaluateFunction,
    dataset_to_reward_kwargs_map: Dict[str, str],
    static_reward_kwargs: Optional[Dict[str, Any]] = None,
    user_message_fn: Optional[
        Callable[[Any], str]
    ] = None,  # Function to construct user message content
    assistant_message_fn: Optional[
        Callable[[Any], str]
    ] = None,  # Function to construct assistant message content
) -> Callable[[List[Any], List[str]], List[float]]:
    """
    Creates an adapter function compatible with TRL trainers (e.g., GRPOTrainer)
    from a reward-kit reward function.

    The TRL trainer expects a reward function with the signature:
    (prompts: List[str], completions: List[str], **kwargs: Any) -> List[float]
    where **kwargs contains other columns from the HuggingFace dataset.

    Args:
        reward_fn: The reward-kit reward function to adapt. This function should
                   already be decorated with @reward_function or follow its
                   input/output conventions (takes List[Message] or List[Dict],
                   returns Dict with a 'score' key).
        dataset_to_reward_kwargs_map: A dictionary mapping dataset column names
                                      (which appear as keys in **kwargs from TRL
                                      ) to `reward_fn` parameter names.
                                      Example: {
                                          "test_cases_col": "test_cases_param"
                                      }
                                      This tells the adapter to take data from
                                      kwargs['test_cases_column'] and pass as
                                      the `test_cases_param` arg to `reward_fn`.
        static_reward_kwargs: Dict of static keyword args that will be
                              passed to `reward_fn` for each sample.
                              Example: {
                                  "language": "python",
                                  "timeout": 10
                              }
        user_message_fn: Optional function that takes prompt string and returns
                         the content for user message. If None, prompt is used
                         is used as content.
        assistant_message_fn: Optional function that takes completion string and
                              returns content for assistant message. If None,
                              the completion itself is used as content.


    Returns:
        An adapter function that can be passed to TRL trainers.
    """
    if static_reward_kwargs is None:
        static_reward_kwargs = {}

    def trl_reward_pipeline(
        prompts: List[Any],  # Changed from List[str] to List[Any]
        completions: Optional[List[str]] = None,
        **kwargs: Any,  # Contains other dataset columns, e.g., test_cases
    ) -> List[float]:
        """
        This is the actual function TRL will call.

        Note: completions param is optional if prompts already
        contain complete conversations.
        """
        scores: List[float] = []
        num_samples = len(prompts)

        # If completions is None, assume prompts contains complete conversations
        if completions is None:
            completions = [""] * num_samples

        if not (len(completions) == num_samples):
            logger.warning(
                f"Mismatch in lengths of prompts ({num_samples}) and "
                f"completions ({len(completions)}). Using min length."
            )
            num_samples = min(num_samples, len(completions))

        # Pre-extract data for all samples from kwargs based on the map
        # This makes it easier to access per-sample data in the loop
        mapped_kwargs_data: Dict[str, List[Any]] = {}
        for (
            dataset_col_name,
            reward_fn_param_name,
        ) in dataset_to_reward_kwargs_map.items():
            if dataset_col_name not in kwargs:
                logger.warning(
                    f"Dataset column '{dataset_col_name}' "
                    f"(mapped to reward_fn param "
                    f"'{reward_fn_param_name}') not found in TRL kwargs. "
                    f"Reward function will receive None for this parameter "
                    f"for all samples."
                )
                # Ensure key exists in mapped_kwargs_data with list of Nones
                mapped_kwargs_data[reward_fn_param_name] = [None] * num_samples
            else:
                # Ensure data from TRL kwargs is a list of correct length
                data_list = kwargs[dataset_col_name]
                if (
                    not isinstance(data_list, list)
                    or len(data_list) != num_samples
                ):
                    logger.error(
                        f"Data for dataset column '{dataset_col_name}' "
                        f"is not a list of "
                        f"length {num_samples}. "
                        f"Received: {data_list}. "
                        f"Reward function will receive None for this parameter "
                        f"for all samples."
                    )
                    mapped_kwargs_data[reward_fn_param_name] = [
                        None
                    ] * num_samples
                else:
                    mapped_kwargs_data[reward_fn_param_name] = data_list

        for i in range(num_samples):
            current_prompt_item: Any = prompts[i]  # Item from the prompts list
            current_completion: str = completions[i]

            # Construct messages
            # If user_message_fn, it converts current_prompt_item to string.
            # If not, and current_prompt_item is not string, may error.
            # Default: current_prompt_item is string if user_message_fn is None.
            user_content = (
                user_message_fn(current_prompt_item)
                if user_message_fn
                else str(current_prompt_item)
            )

            # Default extraction for assistant_content if not simple string
            final_assistant_str_content = ""
            if assistant_message_fn:
                final_assistant_str_content = assistant_message_fn(
                    current_completion
                )
            elif isinstance(current_completion, str):
                final_assistant_str_content = current_completion
            elif (
                isinstance(current_completion, list)
                and len(current_completion) == 1
                and isinstance(current_completion[0], dict)
                and "content" in current_completion[0]
                and isinstance(current_completion[0].get("content"), str)
            ):
                # Handles cases like [{'role':'assistant', 'content':'text'}]
                final_assistant_str_content = current_completion[0]["content"]
            else:
                # Fallback if current_completion is an unexpected type
                logger.warning(
                    f"Completion for assistant message was not a string or "
                    f"expected list/dict structure: {current_completion}. "
                    f"Using str()."
                )
                final_assistant_str_content = str(current_completion)

            # Ensure messages_for_reward is List[Message] per EvaluateFunction
            messages_for_reward: List[Message] = [
                Message(role="user", content=user_content),
                Message(role="assistant", content=final_assistant_str_content),
            ]

            # Prepare kwargs for the specific reward_fn call for this sample
            current_dynamic_kwargs: Dict[str, Any] = {}
            for (
                reward_fn_param_name,
                data_list_for_param,
            ) in mapped_kwargs_data.items():
                # data_list_for_param is list of Nones or actual data
                current_dynamic_kwargs[reward_fn_param_name] = (
                    data_list_for_param[i]
                )

            # Combine static and dynamic kwargs
            final_reward_fn_kwargs = {
                **static_reward_kwargs,
                **current_dynamic_kwargs,
            }

            try:
                # reward_fn is expected to be decorated with @reward_function
                # so it handles Message object creation if dicts are passed,
                # and returns a dict.
                reward_output_dict: Dict[str, Any] = reward_fn(
                    messages=messages_for_reward, **final_reward_fn_kwargs
                )

                score = reward_output_dict.get("score")
                if score is None:
                    logger.warning(
                        f"Sample {i}: 'score' key not found in "
                        f"reward_output_dict or is None. "
                        f"Output: {reward_output_dict}. Assigning 0.0."
                    )
                    scores.append(0.0)
                else:
                    scores.append(float(score))

            except Exception as e:
                logger.error(
                    f"Error calling reward_fn for sample {i} "
                    f"(prompt: '{str(current_prompt_item)[:50]}...'): {e}",
                    exc_info=True,
                )
                scores.append(0.0)  # Assign 0 on error

        if scores:
            logger.debug(
                f"Batch rewards calculated by TRL adapter. "
                f"Count: {len(scores)}, Min: {min(scores)}, "
                f"Max: {max(scores)}, Avg: {sum(scores) / len(scores):.2f}"
            )
        return scores

    return trl_reward_pipeline
