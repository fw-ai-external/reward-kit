from typing import Dict, List, Any, Optional
import json
import re

from ..models import RewardOutput, MetricRewardOutput

def match_function_call(
    messages: List[Dict[str, str]],
    original_messages: List[Dict[str, str]],
    function_name: str,
    parsed_arguments: Dict[str, Any],
    expected_call_schema: Dict[str, Any],
    argument_match_strictness: str = "exact",
    **kwargs
) -> RewardOutput:
    """
    Evaluate how well a function call matches an expected schema.
    
    Args:
        messages: The conversation messages
        original_messages: Original conversation context
        function_name: The parsed function name
        parsed_arguments: The parsed arguments from the function call
        expected_call_schema: The expected schema for the function call
        argument_match_strictness: How strict to be with argument matching:
            - "exact": All arguments must match exactly
            - "partial": Only check provided arguments, ignore missing ones
            - "flexible": Allow extra arguments and type mismatches with penalty
            
    Returns:
        RewardOutput with score and metrics
    """
    metrics = {}
    
    # 1. Function name match
    expected_name = expected_call_schema.get("name", "")
    name_match = function_name == expected_name
    name_score = 1.0 if name_match else 0.0
    name_reason = f"Function name {'matches' if name_match else 'does not match'}: expected '{expected_name}', got '{function_name}'"
    metrics["function_name_match"] = MetricRewardOutput(score=name_score, reason=name_reason)
    
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
                arg_details.append(f"Type mismatch for {arg_name}: expected string, got {type(arg_value).__name__}")
                type_matched = False
            elif expected_type == "number" and not isinstance(arg_value, (int, float)):
                type_mismatches.append(arg_name)
                arg_details.append(f"Type mismatch for {arg_name}: expected number, got {type(arg_value).__name__}")
                type_matched = False
            elif expected_type == "boolean" and not isinstance(arg_value, bool):
                type_mismatches.append(arg_name)
                arg_details.append(f"Type mismatch for {arg_name}: expected boolean, got {type(arg_value).__name__}")
                type_matched = False
            elif expected_type == "array" and not isinstance(arg_value, list):
                type_mismatches.append(arg_name)
                arg_details.append(f"Type mismatch for {arg_name}: expected array, got {type(arg_value).__name__}")
                type_matched = False
            elif expected_type == "object" and not isinstance(arg_value, dict):
                type_mismatches.append(arg_name)
                arg_details.append(f"Type mismatch for {arg_name}: expected object, got {type(arg_value).__name__}")
                type_matched = False
            
            if type_matched:
                perfect_matches.append(arg_name)
                arg_details.append(f"Argument {arg_name} matches expected type {expected_type}")
    
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
    elif argument_match_strictness == "flexible":
        # Allow extra arguments and type mismatches with penalty
        total_expected = len(expected_args)
        if total_expected == 0:
            # If no args were expected, we only penalize for extras
            arg_score = 1.0 if not extra_args else 0.5
        else:
            # Calculate percentage of perfect matches out of expected args
            perfect_score = len(perfect_matches) / total_expected
            
            # Apply penalties
            penalty = 0.0
            if missing_args:
                penalty += 0.3  # 30% penalty for missing args
            if type_mismatches:
                penalty += 0.2  # 20% penalty for type mismatches
            if extra_args:
                penalty += 0.1  # 10% penalty for extra args
            
            # Ensure the score is between 0 and 1
            arg_score = max(0.0, perfect_score - penalty)
    else:
        raise ValueError(f"Invalid argument_match_strictness: {argument_match_strictness}")
    
    arg_reason = "\n".join(arg_details)
    metrics["arguments_match"] = MetricRewardOutput(score=arg_score, reason=arg_reason)
    
    # 3. Calculate final score (equally weighted between name and args)
    final_score = (name_score + arg_score) / 2.0
    
    return RewardOutput(score=final_score, metrics=metrics)