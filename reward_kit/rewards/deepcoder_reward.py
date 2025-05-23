"""
DeepCoder-style reward function for evaluating code correctness based on test cases.
"""

import json
import os
import re  # For function name extraction
from typing import Any, Dict, List, Optional, Union

from ..models import EvaluateResult, Message, MetricResult
from ..reward_function import reward_function
from .code_execution import (
    _HAS_E2B,
)  # Import _HAS_E2B to check E2B availability
from .code_execution import _run_test_cases  # Import the main test case runner
from .code_execution import (
    compare_outputs,
    execute_code_with_e2b,
    execute_javascript_code,
    execute_python_code,
    extract_code_blocks,
)


@reward_function
def deepcoder_code_reward(
    messages: List[
        Message
    ],  # Full conversation, model's response is messages[-1]
    ground_truth: List[Dict[str, Any]],  # This is the test_cases
    language: str,
    timeout: int = 10,  # DeepCoder paper mentions 6-12s, default to 10s
    environment: str = "local",
    api_key: Optional[str] = None,
    target_function: Optional[str] = None,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Evaluates code based on a set of test cases, DeepCoder-style.
    Returns 1.0 if all test cases pass, 0.0 otherwise.
    This version calls the shared _run_test_cases utility.

    Args:
        messages: List of conversation messages. The last message is assumed to be the
                  assistant's response containing the code.
        ground_truth: A list of dictionaries, each representing a test case with "input" (string)
                      and "expected_output" (string). This corresponds to the `test_cases`
                      parameter in the previous signature.
        language: Programming language of the code (e.g., "python", "javascript").
        timeout: Execution timeout per test case in seconds.
        environment: "local" or "e2b" for code execution.
        api_key: E2B API key, required if environment is "e2b".
        target_function: Optional name of the function to call within the code.
        **kwargs: Additional arguments.

    Returns:
        EvaluateResult with a score of 1.0 or 0.0 and detailed metrics.
    """
    metrics_dict: Dict[str, MetricResult] = {}

    if (
        not messages
        or not isinstance(messages[-1], Message)
        or messages[-1].role != "assistant"
        or messages[-1].content is None
    ):
        return EvaluateResult(
            score=0.0,
            reason="Invalid or missing assistant response in messages.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Last message not a valid assistant response.",
                )
            },
            is_score_valid=False,
        )

    assistant_content = messages[-1].content
    test_cases = (
        ground_truth  # The new ground_truth parameter is the test_cases
    )

    code_blocks = extract_code_blocks(assistant_content, language)
    if not code_blocks:
        return EvaluateResult(
            score=0.0,
            reason=f"No {language} code block found.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason=f"No {language} code block found.",
                )
            },
            is_score_valid=False,
        )

    code_to_execute = code_blocks[0]["code"]
    metrics_dict["extracted_code"] = MetricResult(
        score=0.0,
        is_score_valid=True,
        reason=f"Extracted code:\n```\n{code_to_execute}\n```",
    )

    if not test_cases:
        # Convert existing metrics_dict to string for details if needed, or just pass it.
        # For simplicity, let's pass the current metrics_dict.
        return EvaluateResult(
            score=0.0,
            reason="No test cases provided.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="No test cases provided.",
                ),
                **metrics_dict,  # Include already gathered metrics like extracted_code
            },
            is_score_valid=False,
        )

    # Use the explicitly passed target_function if available
    function_to_call = target_function
    if function_to_call:
        metrics_dict["target_function_provided"] = MetricResult(
            score=0.0,
            is_score_valid=True,
            reason=f"Using provided target function: {function_to_call}",
        )
    else:
        metrics_dict["target_function_missing"] = MetricResult(
            score=0.0,
            is_score_valid=False,
            reason="Target function name not provided in input data. Will attempt stdin/stdout.",
        )
        # Fallback to stdin/stdout mode will happen in _run_test_cases

    # Prepare kwargs for _run_test_cases, including the new function_to_call
    run_test_cases_kwargs = {
        "code": code_to_execute,
        "language": language,
        "test_cases": test_cases,
        "timeout": timeout,
        "environment": environment,
        "api_key": api_key,
        "function_to_call": function_to_call,
    }

    # Filter out None values from kwargs
    filtered_kwargs = {
        k: v for k, v in run_test_cases_kwargs.items() if v is not None
    }

    # _run_test_cases already returns EvaluateResult
    eval_result_from_tests: EvaluateResult = _run_test_cases(**filtered_kwargs)  # type: ignore

    # DeepCoder reward is sparse: 1.0 if all pass (score == 1.0 from _run_test_cases), 0.0 otherwise.
    final_score = 1.0 if eval_result_from_tests.score == 1.0 else 0.0

    # Combine metrics from _run_test_cases with metrics gathered here
    if eval_result_from_tests.metrics:
        metrics_dict.update(
            eval_result_from_tests.metrics
        )  # eval_result_from_tests.metrics is Dict[str, MetricResult]

    overall_reason = (
        "All tests passed."
        if final_score == 1.0
        else "One or more tests failed or an error occurred."
    )
    # If _run_test_cases had a top-level error, its reason might be more specific.
    if (
        eval_result_from_tests.reason and eval_result_from_tests.score == 0.0
    ):  # Check if there was an overarching error reason
        # Prefer the reason from _run_test_cases if it indicates a failure.
        # This might happen if _run_test_cases itself had an "error" metric.
        # The `overall_status` below will capture the pass/fail summary.
        # overall_reason is already set based on final_score
        pass
    metrics_dict["overall_status"] = MetricResult(
        score=final_score,
        is_score_valid=(final_score == 1.0),
        reason=overall_reason,
    )

    # The main reason for EvaluateResult should reflect the overall outcome.
    # If _run_test_cases provided a specific reason for failure, use that.
    # Otherwise, use the general pass/fail reason.
    final_reason = overall_reason
    if eval_result_from_tests.score != 1.0 and eval_result_from_tests.reason:
        final_reason = (
            eval_result_from_tests.reason
        )  # Use reason from test runner if it failed and provided one.

    return EvaluateResult(
        score=final_score,
        reason=final_reason,
        metrics=metrics_dict,
        is_score_valid=final_score == 1.0,
    )
