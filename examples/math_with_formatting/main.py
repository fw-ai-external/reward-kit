import math
import re

import numpy as np
import pydantic

from reward_kit.models import Message
from reward_kit.typed_interface import reward_function


@reward_function
def evaluate(messages: list[Message], ground_truth: str, **kwargs) -> dict:
    """
    Evaluates a single entry from the dataset. This function is required in the `main.py` file.
    This template is when you skip rollup in a multi-metrics evaluation.

    Args:
        messages: A list of dictionaries representing a single line from the dataset jsonl file.
        kwargs: Additional keyword arguments. Highly recommended to not remove this due to potential more keywords being passed to the function.
    Returns:
        dict: Evaluate result that should include is_score_valid, score, and reason
    """

    # ground_truth is now directly passed as a parameter due to @reward_function decorator.
    # Assuming messages[-1] is an object with a 'content' attribute
    completion = messages[-1].content

    # Evaluate accuracy and format compliance
    (accuracy_reward, extracted_completion_answer, extracted_ground_truth_answer) = (
        accuracy_reward_fn(completion, ground_truth, True)
    )

    format_reward = format_reward_fn(completion)

    return {
        "score": (accuracy_reward + format_reward) * 0.5,
        "is_score_valid": True,
        "reason": "This is the eval result for the score used",
        "extracted_completion_answer": extracted_completion_answer,
        "extracted_ground_truth_answer": extracted_ground_truth_answer,
        "metrics": {
            "accuracy_reward": {
                "is_score_valid": True,
                "score": accuracy_reward,
                "reason": "This is the eval result for result accuracy",
            },
            "format_reward": {
                "is_score_valid": True,
                "score": format_reward,
                "reason": "This is the eval result for format matching",
            },
        },
    }


def extract_last_number(text: str) -> float:
    """
    Extract the last number from the text.
    Returns the float value, or float('nan') if not found or not convertible.
    """
    num_match = re.findall(r"([-+]?\$?\d[\d,]*\.?\d*)", text)
    if num_match:
        answer = num_match[-1]
        answer = answer.replace("$", "").replace(",", "")
        try:
            return float(answer.strip())
        except ValueError:
            return float("nan")
    return float("nan")


def fraction_to_float(text: str) -> float:
    r"""
    Detects and converts LaTeX fraction strings to float.
    Supports \dfrac{}{} and \frac{}{} formats.
    """
    pattern = r"\\(?:d?frac)\{([\d.]+)\}\{([\d.]+)\}"
    match = re.search(pattern, text)
    if match:
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        try:
            return numerator / denominator
        except ZeroDivisionError:
            return float("nan")  # Handle division by zero
    return float("nan")  # Return NaN if no match is found


def extract_boxed_value(text: str) -> float:
    r"""
    Extract the value from the last \\boxed{} pattern in the text.
    Returns the float value, or float('nan') if not found or not convertible.
    First tries to convert directly to float, then tries fraction_to_float if that fails.

    Args:
        text (str): The input text containing \\boxed{} patterns.

    Returns:
        float: The extracted value from the last \\boxed{} pattern as a float,
               or float('nan') if no valid pattern is found or conversion fails.
    """

    boxed_match = list(re.finditer(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text))
    if boxed_match:
        answer = boxed_match[-1].group(1)
        answer = answer.replace("$", "").replace("\\!", "").replace(",", "")
        try:
            return float(answer.strip())
        except ValueError:
            # Try to extract as a fraction if direct float conversion fails
            return fraction_to_float(answer.strip())

    return float("nan")


def extract_math_answer(completion: str) -> float:
    """
    Extract the answer from the text.
    Returns the float value, or float('nan') if not found or not convertible.
    """

    # Check if the format is correct
    pattern = r".*<answer>([\s\S]*?)</answer>"
    match = re.search(pattern, completion, re.DOTALL)
    if match:
        completion = match.group(1).strip()

    answer = extract_boxed_value(completion)
    if not math.isnan(answer):
        return answer

    answer = fraction_to_float(completion)
    if not math.isnan(answer):
        return answer

    answer = extract_last_number(completion)
    if not math.isnan(answer):
        return answer
    return float("nan")


def format_reward_fn(completion: str) -> float:
    """Reward function that checks if the completion has a specific format."""
    # Match the entire string. Allows any characters between </think> and <answer>.
    pattern = r"^<think>[\s\S]*?</think>[\s\S]*?<answer>[\s\S]*?</answer>$"
    match = re.match(pattern, completion)
    return 1.0 if match else 0.0


def accuracy_reward_fn(
    completion: str, ground_truth: str, force_format_reward: bool = True
) -> tuple[float, float, float]:
    """
    Reward function that checks if the completion's answer matches the ground truth.

    Args:
        completion: The model completion text
        ground_truth: The ground truth text
        force_format_reward: If True, checks for correct format before computing the accuracy reward

    Returns:
        Tuple containing:
        - Reward value (1.0 if correct, 0.0 if incorrect)
        - Extracted completion answer
        - Extracted ground truth answer
    """

    extracted_completion_answer = extract_math_answer(completion)
    extracted_ground_truth_answer = extract_math_answer(ground_truth)

    if force_format_reward:
        # If the format is incorrect, return 0.0 accuracy reward
        format_reward = format_reward_fn(completion)
        if format_reward == 0.0:
            return 0.0, extracted_completion_answer, extracted_ground_truth_answer
        else:
            is_correct = (
                abs(extracted_completion_answer - extracted_ground_truth_answer) < 1e-6
            )
            return (
                float(is_correct),
                extracted_completion_answer,
                extracted_ground_truth_answer,
            )
    else:
        is_correct = (
            abs(extracted_completion_answer - extracted_ground_truth_answer) < 1e-6
        )
        return (
            float(is_correct),
            extracted_completion_answer,
            extracted_ground_truth_answer,
        )
