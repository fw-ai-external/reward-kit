"""
BFCL Reward Function

This module contains a reward function for evaluating BFCL agent performance.
"""

from typing import Any, Dict, List, Optional, Tuple, Union


def bfcl_reward(
    ground_truth_function_calls: List[List[str]],
    ground_truth_comparable_state: Dict[str, Any],
    model_history: List[Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate an agent's performance on a BFCL task by comparing:
    1. The agent's tool calls to the ground truth tool calls
    2. The final state to the ground truth state

    Args:
        ground_truth_function_calls: List of lists of function calls per turn
        ground_truth_comparable_state: Expected final state
        model_history: Agent's conversation and tool call history

    Returns:
        Dictionary with score and evaluation details
    """
    # Extract tool calls from model history
    model_function_calls = _extract_function_calls_from_history(model_history)

    # Calculate function call score
    function_call_score, function_call_details = _evaluate_function_calls(
        ground_truth_function_calls, model_function_calls
    )

    # Extract final state from model history
    model_final_state = (
        model_history[-1].get("final_state", {}) if model_history else {}
    )

    # Compare states
    state_score, state_comparison = _evaluate_state_match(
        ground_truth_comparable_state, model_final_state
    )

    # Calculate overall score (with function calls weighted more)
    overall_score = 0.6 * function_call_score + 0.4 * state_score

    return {
        "score": overall_score,
        "function_call_score": function_call_score,
        "state_score": state_score,
        "function_call_details": function_call_details,
        "state_comparison": state_comparison,
        "reason": _generate_reason(function_call_score, state_score),
    }


def _extract_function_calls_from_history(
    model_history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract function calls from model history."""
    function_calls = []

    for message in model_history:
        if message.get("role") == "assistant" and "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                if "function" in tool_call:
                    function_calls.append(
                        {
                            "name": tool_call["function"]["name"],
                            "arguments": tool_call["function"]["arguments"],
                        }
                    )

    return function_calls


def _evaluate_function_calls(
    ground_truth_calls: List[List[str]], model_calls: List[Dict[str, Any]]
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate how well the model's function calls match the ground truth.

    Returns:
        Tuple of (score, details)
    """
    if not ground_truth_calls:
        return 0.0, {"error": "No ground truth function calls provided"}

    if not model_calls:
        return 0.0, {"error": "No model function calls found"}

    # Flatten ground truth calls
    flat_gt_calls = []
    for turn_calls in ground_truth_calls:
        flat_gt_calls.extend(turn_calls)

    # Count matches
    matches = 0
    match_details = []
    for gt_call in flat_gt_calls:
        best_match = None
        best_score = 0.0

        for model_call in model_calls:
            match_score = _function_call_match_score(gt_call, model_call)
            if match_score > best_score:
                best_score = match_score
                best_match = model_call

        if best_score > 0.8:  # Consider a strong match
            matches += 1
            match_details.append(
                {"ground_truth": gt_call, "model_call": best_match, "score": best_score}
            )

    # Calculate overall score
    score = matches / len(flat_gt_calls) if flat_gt_calls else 0.0

    return score, {
        "matches": matches,
        "total_expected": len(flat_gt_calls),
        "match_details": match_details,
    }


def _function_call_match_score(gt_call: str, model_call: Dict[str, Any]) -> float:
    """
    Calculate a match score between a ground truth function call and a model function call.

    Args:
        gt_call: String representation of ground truth call like "cd(folder='document')"
        model_call: Dict with "name" and "arguments" keys

    Returns:
        Score between 0.0 and 1.0
    """
    # Parse ground truth function name and arguments
    parts = gt_call.split("(", 1)
    if len(parts) != 2 or not parts[1].endswith(")"):
        return 0.0

    gt_name = parts[0]
    gt_args_str = parts[1][:-1]  # Remove trailing )

    # Check if function names match
    if gt_name != model_call["name"]:
        return 0.0

    # If no arguments, perfect match if names match
    if not gt_args_str:
        return 1.0

    # Parse arguments crudely (this is a simplified version)
    gt_args = {}
    for arg_pair in gt_args_str.split(","):
        if "=" in arg_pair:
            key, value = arg_pair.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Handle string values
            if (value.startswith("'") and value.endswith("'")) or (
                value.startswith('"') and value.endswith('"')
            ):
                value = value[1:-1]

            gt_args[key] = value

    # Get model arguments
    model_args = model_call.get("arguments", {})
    if isinstance(model_args, str):
        try:
            import json

            model_args = json.loads(model_args)
        except json.JSONDecodeError:
            model_args = {}

    # Compare arguments
    if not gt_args and not model_args:
        return 1.0

    # Count matching arguments
    matches = 0
    total = len(gt_args)

    for key, gt_value in gt_args.items():
        if key in model_args:
            model_value = str(model_args[key])
            if gt_value == model_value:
                matches += 1

    return matches / total if total > 0 else 0.0


def _evaluate_state_match(
    ground_truth_state: Dict[str, Any], model_state: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """
    Compare the model's final state to the ground truth state.

    Returns:
        Tuple of (score, details)
    """
    if not ground_truth_state:
        return 0.0, {"error": "No ground truth state provided"}

    if not model_state:
        return 0.0, {"error": "No model state found"}

    # For simplicity, we'll use direct comparison
    # In a real implementation, this would be more nuanced
    if ground_truth_state == model_state:
        return 1.0, {"match": True, "differences": []}
    else:
        return 0.0, {"match": False, "differences": ["States do not match exactly"]}


def _generate_reason(function_call_score: float, state_score: float) -> str:
    """Generate a human-readable reason for the scores."""
    reasons = []

    # Function call assessment
    if function_call_score >= 0.9:
        reasons.append("The agent executed nearly all required functions correctly.")
    elif function_call_score >= 0.7:
        reasons.append("The agent executed most required functions correctly.")
    elif function_call_score >= 0.4:
        reasons.append(
            "The agent executed some required functions correctly, but missed or incorrectly executed others."
        )
    else:
        reasons.append("The agent failed to execute most required functions correctly.")

    # State assessment
    if state_score >= 0.9:
        reasons.append("The final environment state matches the expected state.")
    elif state_score >= 0.5:
        reasons.append(
            "The final environment state partially matches the expected state."
        )
    else:
        reasons.append("The final environment state does not match the expected state.")

    return " ".join(reasons)
