import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Import OpenAI at module level for mocking in tests
try:
    import openai
    from openai import OpenAI
except ImportError:
    # Type to mock in tests
    OpenAI = None  # type: ignore

import copy
from collections import Counter

from ..models import EvaluateResult, Message, MetricResult  # Added Message
from ..typed_interface import reward_function  # Added reward_function


def match_function_call(
    messages: List[Dict[str, Any]],  # messages is for context if needed
    function_name: str,
    parsed_arguments: Dict[str, Any],
    expected_call_schema: Dict[str, Any],
    argument_match_strictness: str = "exact",
    **kwargs,
) -> EvaluateResult:
    """
    Evaluate how well a function call matches an expected schema.

    Args:
        messages: The conversation messages (for context, not directly used for call parts).
        function_name: The parsed function name.
        parsed_arguments: The parsed arguments from the function call.
        expected_call_schema: The expected schema for the function call.
        argument_match_strictness: How strict to be with argument matching:
            - "exact": All arguments must match exactly
            - "partial": Only check provided arguments, ignore missing ones
            - "flexible": Allow extra arguments and type mismatches with penalty

    Returns:
        EvaluateResult with score and metrics
    """
    metrics = {}

    # 1. Function name match
    expected_name = expected_call_schema.get("name", "")
    name_match = function_name == expected_name
    name_score = 1.0 if name_match else 0.0
    name_reason = f"Function name {'matches' if name_match else 'does not match'}: expected '{expected_name}', got '{function_name}'"
    metrics["function_name_match"] = MetricResult(
        score=name_score, reason=name_reason, is_score_valid=name_match
    )

    # 2. Arguments match
    expected_args = expected_call_schema.get("arguments", {})
    arg_score = 0.0
    arg_details = []

    # We'll track different aspects of argument matching
    missing_args = []
    extra_args = []
    type_mismatches = []
    perfect_matches = []

    # Check for expected arguments
    for arg_name, arg_schema in expected_args.items():
        expected_type = arg_schema.get("type", "any")

        if arg_name not in parsed_arguments:
            missing_args.append(arg_name)
            arg_details.append(f"Missing argument: {arg_name}")
        else:
            arg_value = parsed_arguments[arg_name]
            # Basic type checking
            type_matched = True
            if expected_type == "string" and not isinstance(arg_value, str):
                type_mismatches.append(arg_name)
                arg_details.append(
                    f"Type mismatch for {arg_name}: expected string, got {type(arg_value).__name__}"
                )
                type_matched = False
            elif expected_type == "number" and not isinstance(arg_value, (int, float)):
                type_mismatches.append(arg_name)
                arg_details.append(
                    f"Type mismatch for {arg_name}: expected number, got {type(arg_value).__name__}"
                )
                type_matched = False
            elif expected_type == "boolean" and not isinstance(arg_value, bool):
                type_mismatches.append(arg_name)
                arg_details.append(
                    f"Type mismatch for {arg_name}: expected boolean, got {type(arg_value).__name__}"
                )
                type_matched = False
            elif expected_type == "array" and not isinstance(arg_value, list):
                type_mismatches.append(arg_name)
                arg_details.append(
                    f"Type mismatch for {arg_name}: expected array, got {type(arg_value).__name__}"
                )
                type_matched = False
            elif expected_type == "object" and not isinstance(arg_value, dict):
                type_mismatches.append(arg_name)
                arg_details.append(
                    f"Type mismatch for {arg_name}: expected object, got {type(arg_value).__name__}"
                )
                type_matched = False

            if type_matched:
                perfect_matches.append(arg_name)
                arg_details.append(
                    f"Argument {arg_name} matches expected type {expected_type}"
                )

    # Check for extra arguments
    for arg_name in parsed_arguments:
        if arg_name not in expected_args:
            extra_args.append(arg_name)
            arg_details.append(f"Unexpected argument: {arg_name}")

    # Calculate argument score based on strictness
    if argument_match_strictness == "exact":
        # All arguments must match exactly
        if missing_args or extra_args or type_mismatches:
            arg_score = 0.0
        else:
            arg_score = 1.0
    elif argument_match_strictness == "partial":
        # Only check provided arguments, ignore missing ones
        if extra_args or type_mismatches:
            arg_score = 0.0
        else:
            # We weight based on how many expected args were provided correctly
            total_provided = len(parsed_arguments)
            if total_provided == 0:
                arg_score = 0.0
            else:
                correct_args = len(perfect_matches)
                arg_score = correct_args / total_provided
    elif (
        argument_match_strictness == "permissive"
        or argument_match_strictness == "flexible"
    ):
        # For permissive mode, ignore extra arguments and just check that required ones are present
        # and have the correct type
        if missing_args or type_mismatches:
            arg_score = 0.0
        else:
            arg_score = 1.0
    else:
        raise ValueError(
            f"Invalid argument_match_strictness: {argument_match_strictness}"
        )

    arg_reason = "\n".join(arg_details)
    metrics["arguments_match"] = MetricResult(
        score=arg_score,
        reason=arg_reason,
        is_score_valid=arg_score == 1.0 if len(expected_args) > 0 else True,
    )

    # 3. Calculate final score (equally weighted between name and args)
    final_score = (name_score + arg_score) / 2.0
    final_reason = f"Overall score based on name match ({name_score:.2f}) and argument match ({arg_score:.2f})."

    return EvaluateResult(score=final_score, reason=final_reason, metrics=metrics)


def calculate_jaccard_similarity(set1: Set, set2: Set) -> float:
    """
    Calculate Jaccard similarity between two sets.

    Jaccard similarity is defined as the size of the intersection divided by the size of the union.

    Args:
        set1: First set
        set2: Second set

    Returns:
        Jaccard similarity score between 0.0 and 1.0
    """
    if not set1 and not set2:
        return 1.0  # Both empty means perfect similarity

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union


def extract_schema_properties(schema: Dict[str, Any]) -> Set[Tuple[str, str]]:
    """
    Extract properties from a JSON schema as a set of (name, type) tuples.

    Args:
        schema: JSON schema object

    Returns:
        Set of (property_name, property_type) tuples
    """
    properties = set()

    # Process schema properties (handles both root-level and nested properties)
    def process_properties(schema_obj: Dict[str, Any], prefix: str = ""):
        if not isinstance(schema_obj, dict):
            return

        # Handle properties field
        props = schema_obj.get("properties", {})
        for prop_name, prop_schema in props.items():
            prop_path = f"{prefix}.{prop_name}" if prefix else prop_name
            prop_type = prop_schema.get("type", "any")
            properties.add((prop_path, prop_type))

            # Recursively process object properties
            if prop_type == "object":
                process_properties(prop_schema, prop_path)

        # Handle patternProperties field
        pattern_props = schema_obj.get("patternProperties", {})
        for pattern, pattern_schema in pattern_props.items():
            prop_path = f"{prefix}[{pattern}]" if prefix else f"[{pattern}]"
            prop_type = pattern_schema.get("type", "any")
            properties.add((prop_path, prop_type))

            # Recursively process object pattern properties
            if prop_type == "object":
                process_properties(pattern_schema, prop_path)

        # Handle items for arrays
        items = schema_obj.get("items", {})
        if items and isinstance(items, dict):
            prop_path = f"{prefix}[]" if prefix else "[]"
            prop_type = items.get("type", "any")
            properties.add((prop_path, prop_type))

            # Recursively process array item properties
            if prop_type == "object":
                process_properties(items, prop_path)

    # Start processing at the root level
    process_properties(schema)
    return properties


def normalize_schema(schema: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Normalize schema to a standard dictionary format.

    Args:
        schema: JSON schema as dictionary or string

    Returns:
        Normalized schema dictionary
    """
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError:
            return {}

    if not isinstance(schema, dict):
        return {}

    return schema


# New Exact Tool Match Reward Function and Helpers
# VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV


def maybe_deserialize_tool_call_arguments(
    tool_calls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result = copy.deepcopy(tool_calls)
    for tool_call in result:
        # Sanity check
        assert (
            "function" in tool_call
        ), f"Tool call {tool_call} does not have a function"
        function = tool_call["function"]
        # Sanity check
        assert (
            "arguments" in function
        ), f"Function {function} does not have an arguments"
        # Sanity check, we should have converted None to ""
        assert (
            function["arguments"] is not None
        ), f"Tool call {tool_call} has a None arguments"
        if function["arguments"] == "":
            function["arguments"] = None
        else:
            try:
                arguments = json.loads(function["arguments"])
                function["arguments"] = arguments
            except Exception:  # pylint: disable=bare-except
                # In this case, leave it as string
                pass
    return result


def parse_tool_calls(completion: str) -> list:
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", completion, re.DOTALL)
    row_tool_calls = []
    for match in matches:
        try:
            tool_call_str = match.strip()
            row_tool_calls.append(json.loads(tool_call_str))
        except Exception:  # pylint: disable=bare-except
            # print("tool call parsing error")
            continue
    return row_tool_calls


def compare_tool_calls(generated_tool_calls: list, gt_tool_calls: list) -> bool:
    if len(generated_tool_calls) != len(gt_tool_calls):
        return False

    generated_tool_calls_serialized = [
        json.dumps(item, sort_keys=True) for item in generated_tool_calls
    ]
    gt_tool_calls_serialized = [
        json.dumps(item, sort_keys=True) for item in gt_tool_calls
    ]

    # Direct list comparison for order sensitivity
    return generated_tool_calls_serialized == gt_tool_calls_serialized


def eval_tool_call(generation: dict, ground_truth: dict) -> bool:
    if ground_truth is None or "tool_calls" not in ground_truth:
        expected_gt_tool_calls = []
    else:
        expected_gt_tool_calls = ground_truth["tool_calls"]

    ground_truth_tool_calls = maybe_deserialize_tool_call_arguments(
        expected_gt_tool_calls
    )
    ground_truth_functions = [item["function"] for item in ground_truth_tool_calls]

    generated_functions = []
    # Check if 'tool_calls' key exists and is not None or empty
    if generation.get("tool_calls"):
        processed_tool_calls = []
        for tc in generation["tool_calls"]:
            if hasattr(tc, "model_dump"):
                processed_tool_calls.append(tc.model_dump())
            elif isinstance(tc, dict):
                processed_tool_calls.append(tc)

        deserialized_tool_calls = maybe_deserialize_tool_call_arguments(
            processed_tool_calls
        )
        generated_functions = [item["function"] for item in deserialized_tool_calls]
    # Check 'content' only if 'tool_calls' is not present or is empty
    elif generation.get("content") and "<tool_call>" in generation["content"]:
        parsed_tool_calls_from_content = parse_tool_calls(generation["content"])
        if parsed_tool_calls_from_content:  # Check if parsing yielded any results
            deserialized_tool_calls_from_content = (
                maybe_deserialize_tool_call_arguments(parsed_tool_calls_from_content)
            )
            generated_functions = [
                item["function"] for item in deserialized_tool_calls_from_content
            ]
        # If parsing yields nothing, generated_functions remains empty, which is correct.

    return compare_tool_calls(generated_functions, ground_truth_functions)


@reward_function
def exact_tool_match_reward(
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> EvaluateResult:
    if not messages:
        return EvaluateResult(
            score=0.0, reason="No messages provided for evaluation.", metrics={}
        )

    generation_message_obj = messages[-1]
    generation_dict: Dict[str, Any]

    if isinstance(generation_message_obj, Message):
        generation_dict = {
            "role": generation_message_obj.role,
            "content": generation_message_obj.content,
        }
        if generation_message_obj.tool_calls:
            # Ensure tool_calls are dicts, not Pydantic models, for eval_tool_call
            generation_dict["tool_calls"] = [
                tc.model_dump() if hasattr(tc, "model_dump") else tc
                for tc in generation_message_obj.tool_calls
            ]
    elif isinstance(generation_message_obj, dict):
        generation_dict = generation_message_obj
    else:
        return EvaluateResult(
            score=0.0,
            reason=f"Unexpected type for generation message: {type(generation_message_obj)}",
            metrics={},
        )

    if ground_truth is None:
        has_generation_tool_calls = False
        # Check 'tool_calls' first
        if generation_dict.get("tool_calls"):
            has_generation_tool_calls = True
        # Then check 'content' if 'tool_calls' was not indicative
        elif "<tool_call>" in generation_dict.get("content", ""):
            if parse_tool_calls(generation_dict.get("content", "")):
                has_generation_tool_calls = True

        score = 1.0 if not has_generation_tool_calls else 0.0
        reason = "Ground truth not provided. Score based on absence (1.0) or presence (0.0) of tool calls in generation."
        return EvaluateResult(score=score, reason=reason, metrics={})

    if not isinstance(ground_truth, dict):
        return EvaluateResult(
            score=0.0,
            reason=f"Ground truth is not a dictionary: {type(ground_truth)}",
            metrics={},
        )

    score = float(eval_tool_call(generation_dict, ground_truth))
    reason = f"Exact tool match evaluation score: {score}"
    return EvaluateResult(score=score, reason=reason, metrics={})


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# End of New Exact Tool Match Reward Function and Helpers


@reward_function  # Added decorator
def schema_jaccard_reward(
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[
        Dict[str, Any]
    ] = None,  # Ensure ground_truth type is Dict for exact_tool_match_reward
    function_call: Optional[Dict[str, Any]] = None,  # Param becomes unused
    expected_schema: Optional[
        Union[Dict[str, Any], str]
    ] = None,  # Param becomes unused
    **kwargs,
) -> EvaluateResult:
    """
    DEPRECATED: This function is deprecated and will be removed in a future version.
    Please use `exact_tool_match_reward` for evaluating tool calls.

    NOTE: This function now delegates to exact_tool_match_reward.
    Original Jaccard similarity logic for function call schemas is bypassed.
    The helper functions for Jaccard similarity are kept in this file as they
    are used by reward_kit.rewards.json_schema.py.

    Args:
        messages: List of conversation messages.
        ground_truth: Expected assistant response as a dictionary.
        function_call: (Unused) Kept for signature compatibility.
        expected_schema: (Unused) Kept for signature compatibility.
        **kwargs: Additional keyword arguments.

    Returns:
        EvaluateResult from exact_tool_match_reward.
    """
    warnings.warn(
        "`schema_jaccard_reward` is deprecated and will be removed in a future version. "
        "Please use `exact_tool_match_reward`.",
        DeprecationWarning,
        stacklevel=2,
    )
    return exact_tool_match_reward(
        messages=messages, ground_truth=ground_truth, **kwargs
    )


@reward_function  # Added decorator
def llm_judge_reward(
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[Dict[str, Any]] = None,  # Ensure ground_truth type is Dict
    function_call: Optional[Dict[str, Any]] = None,  # Param becomes unused
    expected_schema: Optional[
        Union[Dict[str, Any], str]
    ] = None,  # Param becomes unused
    expected_behavior: Optional[str] = None,  # Param becomes unused
    openai_api_key: Optional[str] = None,  # Param becomes unused
    model: str = "gpt-4o-mini",  # Param becomes unused
    temperature: float = 0.0,  # Param becomes unused
    **kwargs,
) -> EvaluateResult:
    """
    DEPRECATED: This function is deprecated and will be removed in a future version.
    Please use `exact_tool_match_reward` for evaluating tool calls.

    NOTE: This function now delegates to exact_tool_match_reward.
    Original LLM judge logic is bypassed.

    Args:
        messages: List of conversation messages.
        ground_truth: Expected assistant response as a dictionary.
        function_call: (Unused) Kept for signature compatibility.
        expected_schema: (Unused) Kept for signature compatibility.
        expected_behavior: (Unused) Kept for signature compatibility.
        openai_api_key: (Unused) Kept for signature compatibility.
        model: (Unused) Kept for signature compatibility.
        temperature: (Unused) Kept for signature compatibility.
        **kwargs: Additional keyword arguments.

    Returns:
        EvaluateResult from exact_tool_match_reward.
    """
    warnings.warn(
        "`llm_judge_reward` is deprecated and will be removed in a future version. "
        "Please use `exact_tool_match_reward`.",
        DeprecationWarning,
        stacklevel=2,
    )
    return exact_tool_match_reward(
        messages=messages, ground_truth=ground_truth, **kwargs
    )


@reward_function
def composite_function_call_reward(
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[Dict[str, Any]] = None,  # Changed type here
    # The following parameters are now effectively unused due to delegation
    # but are kept for signature compatibility or future flexibility.
    function_call: Optional[Dict[str, Any]] = None,
    expected_schema: Optional[Union[Dict[str, Any], str]] = None,
    expected_behavior: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o-mini",
    weights: Optional[Dict[str, float]] = None,
    **kwargs,
) -> EvaluateResult:
    """
    DEPRECATED: This function is deprecated and will be removed in a future version.
    Please use `exact_tool_match_reward` for evaluating tool calls.

    This reward function now delegates to exact_tool_match_reward
    for an exact match evaluation of tool calls.
    The model's response (containing the function call) is assumed to be `messages[-1]`.

    Args:
        messages: List of conversation messages, where `messages[-1]` is the model's response.
        ground_truth: Expected assistant response as a dictionary, typically containing 'tool_calls'.
                      This is passed directly to exact_tool_match_reward.
        function_call: (Unused) Kept for signature compatibility.
        expected_schema: (Unused) Kept for signature compatibility.
        expected_behavior: (Unused) Kept for signature compatibility.
        openai_api_key: (Unused) Kept for signature compatibility.
        llm_model: (Unused) Kept for signature compatibility.
        weights: (Unused) Kept for signature compatibility.
        **kwargs: Additional keyword arguments passed to exact_tool_match_reward.

    Returns:
        EvaluateResult with score and metrics from exact_tool_match_reward.
    """
    warnings.warn(
        "`composite_function_call_reward` is deprecated and will be removed in a future version. "
        "Please use `exact_tool_match_reward`.",
        DeprecationWarning,
        stacklevel=2,
    )
    return exact_tool_match_reward(
        messages=messages, ground_truth=ground_truth, **kwargs
    )


# JSON schema reward functions have been moved to json_schema.py module
