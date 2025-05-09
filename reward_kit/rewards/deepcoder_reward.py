"""
DeepCoder-style reward function for evaluating code correctness based on test cases.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union

from ..models import Message, EvaluateResult, MetricResult
from ..reward_function import reward_function
from .code_execution import (
    extract_code_blocks,
    execute_python_code,
    execute_javascript_code,
    execute_code_with_e2b,
    compare_outputs,
    _HAS_E2B, # Import _HAS_E2B to check E2B availability
    _run_test_cases # Import the main test case runner
)
import re # For function name extraction


@reward_function
def deepcoder_code_reward(
    messages: Union[List[Dict[str, Any]], List[Message]],
    language: str,
    test_cases: List[Dict[str, Any]],
    timeout: int = 10, # DeepCoder paper mentions 6-12s, default to 10s
    environment: str = "local",
    api_key: Optional[str] = None,
    original_messages: Optional[Union[List[Dict[str, Any]], List[Message]]] = None, # Kept for potential future use, but not needed for name extraction now
    target_function: Optional[str] = None, # Added explicit argument
    **kwargs: Any,
) -> EvaluateResult: # Changed to EvaluateResult
    """
    Evaluates code based on a set of test cases, DeepCoder-style.
    Returns 1.0 if all test cases pass, 0.0 otherwise.
    This version calls the shared _run_test_cases utility.

    Args:
        messages: List of conversation messages. The last assistant message should contain the code.
        language: Programming language of the code (e.g., "python", "javascript").
        test_cases: A list of dictionaries, each with "input" (string) and "expected_output" (string).
        timeout: Execution timeout per test case in seconds.
        environment: "local" or "e2b" for code execution.
        api_key: E2B API key, required if environment is "e2b".
        original_messages: Original conversation context, used to find the user prompt for function name extraction.
        **kwargs: Additional arguments.

    Returns:
        RewardOutput with a score of 1.0 or 0.0 and detailed metrics.
    """
    metrics_dict: Dict[str, MetricResult] = {} # Changed to MetricResult

    if not messages:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided.",
            metrics={"error": MetricResult(score=0.0, success=False, reason="No messages provided.")}
        )

    last_message = messages[-1]
    assistant_content: str
    assistant_role: Optional[str]

    if isinstance(last_message, Message):
        assistant_content = last_message.content
        assistant_role = last_message.role
    elif isinstance(last_message, dict):
        assistant_content = last_message.get("content", "")
        assistant_role = last_message.get("role")
    else:
        return EvaluateResult(
            score=0.0,
            reason="Last message is of an unexpected type.",
            metrics={"error": MetricResult(score=0.0, success=False, reason="Last message is of an unexpected type.")}
        )

    if assistant_role != "assistant":
        return EvaluateResult(
            score=0.0,
            reason="Last message is not from assistant.",
            metrics={"error": MetricResult(score=0.0, success=False, reason="Last message is not from assistant.")}
        )

    code_blocks = extract_code_blocks(assistant_content, language)
    if not code_blocks:
        return EvaluateResult(
            score=0.0,
            reason=f"No {language} code block found.",
            metrics={"error": MetricResult(score=0.0, success=False, reason=f"No {language} code block found.")}
        )
    
    code_to_execute = code_blocks[0]["code"]
    metrics_dict["extracted_code"] = MetricResult(score=0.0, success=True, reason=f"Extracted code:\n```\n{code_to_execute}\n```")


    if not test_cases:
        # Convert existing metrics_dict to string for details if needed, or just pass it.
        # For simplicity, let's pass the current metrics_dict.
        return EvaluateResult(
            score=0.0,
            reason="No test cases provided.",
            metrics={
                "error": MetricResult(score=0.0, success=False, reason="No test cases provided."),
                **metrics_dict # Include already gathered metrics like extracted_code
            }
        )

    # Use the explicitly passed target_function if available
    function_to_call = target_function
    if function_to_call:
         metrics_dict["target_function_provided"] = MetricResult(score=0.0, success=True, reason=f"Using provided target function: {function_to_call}")
    else:
         metrics_dict["target_function_missing"] = MetricResult(score=0.0, success=False, reason="Target function name not provided in input data. Will attempt stdin/stdout.")
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
    filtered_kwargs = {k: v for k, v in run_test_cases_kwargs.items() if v is not None}

    # _run_test_cases already returns EvaluateResult
    eval_result_from_tests: EvaluateResult = _run_test_cases(**filtered_kwargs) # type: ignore

    # DeepCoder reward is sparse: 1.0 if all pass (score == 1.0 from _run_test_cases), 0.0 otherwise.
    final_score = 1.0 if eval_result_from_tests.score == 1.0 else 0.0

    # Combine metrics from _run_test_cases with metrics gathered here
    if eval_result_from_tests.metrics:
        metrics_dict.update(eval_result_from_tests.metrics) # eval_result_from_tests.metrics is Dict[str, MetricResult]

    overall_reason = "All tests passed." if final_score == 1.0 else "One or more tests failed or an error occurred."
    # If _run_test_cases had a top-level error, its reason might be more specific.
    if eval_result_from_tests.reason and eval_result_from_tests.score == 0.0: # Check if there was an overarching error reason
        # Prefer the reason from _run_test_cases if it indicates a failure.
        # This might happen if _run_test_cases itself had an "error" metric.
        # The `overall_status` below will capture the pass/fail summary.
        # overall_reason is already set based on final_score
        pass
    metrics_dict["overall_status"] = MetricResult(score=final_score, success=(final_score == 1.0), reason=overall_reason)
    
    # The main reason for EvaluateResult should reflect the overall outcome.
    # If _run_test_cases provided a specific reason for failure, use that.
    # Otherwise, use the general pass/fail reason.
    final_reason = overall_reason
    if eval_result_from_tests.score != 1.0 and eval_result_from_tests.reason:
        final_reason = eval_result_from_tests.reason # Use reason from test runner if it failed and provided one.

    return EvaluateResult(score=final_score, reason=final_reason, metrics=metrics_dict)
