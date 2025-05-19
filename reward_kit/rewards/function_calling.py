import json

# from collections import defaultdict # Unused import
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Import OpenAI at module level for mocking in tests
try:
    import openai
    from openai import OpenAI
except ImportError:
    # Type to mock in tests
    OpenAI = None  # type: ignore

from ..models import EvaluateResult, Message, MetricResult  # Added Message
from ..typed_interface import reward_function  # Added reward_function


def match_function_call(
    messages: List[
        Dict[str, str]
    ],  # messages is for context if needed, not directly used here for func call parts
    # original_messages: List[Dict[str, str]], # Removed
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
        score=name_score, reason=name_reason, success=name_match
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
        success=arg_score == 1.0 if len(expected_args) > 0 else True,
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


@reward_function  # Added decorator
def schema_jaccard_reward(
    messages: Union[List[Message], List[Dict[str, Any]]],  # Updated type
    ground_truth: Optional[Union[List[Message], List[Dict[str, Any]]]] = None,  # Added
    function_call: Optional[Dict[str, Any]] = None,
    expected_schema: Optional[Union[Dict[str, Any], str]] = None,
    **kwargs,
) -> EvaluateResult:
    """
    Evaluate a function call using Jaccard similarity between actual and expected schema.
    The model's response (containing the function call) is assumed to be `messages[-1]`.

    This reward function compares the structure of a function call against an expected schema
    and calculates a similarity score using Jaccard similarity.

    Args:
        messages: List of conversation messages, where `messages[-1]` is the model's response.
        ground_truth: Optional. Expected assistant response trajectory. If this contains an expected
                      function call, it might be used in future versions or by specific setups.
        function_call: The function call to evaluate (if not provided, extracts from `messages[-1]`).
        expected_schema: The expected schema for the function call.
        **kwargs: Additional keyword arguments.

    Returns:
        EvaluateResult with score and metrics
    """
    metrics = {}

    # Extract function call from messages if not provided directly
    if function_call is None:
        if not messages:
            return EvaluateResult(
                score=0.0,
                reason="No messages provided to extract function call.",
                metrics={
                    "error": MetricResult(
                        score=0.0, reason="No messages provided", success=False
                    )
                },
            )

        last_message = messages[-1]
        extracted_fc_from_message = None

        if isinstance(last_message, Message):
            if last_message.role == "assistant":
                if last_message.function_call:
                    extracted_fc_from_message = last_message.function_call
                elif last_message.tool_calls:
                    for tc in last_message.tool_calls:
                        if tc.type == "function":
                            extracted_fc_from_message = (
                                tc.function
                            )  # Assuming tc.function is a dict {name, arguments}
                            if isinstance(
                                extracted_fc_from_message, dict
                            ):  # Ensure it's a dict
                                break
                            else:  # If tc.function is not a dict (e.g. pydantic model), convert or handle
                                extracted_fc_from_message = {
                                    "name": getattr(
                                        extracted_fc_from_message, "name", ""
                                    ),
                                    "arguments": getattr(
                                        extracted_fc_from_message, "arguments", "{}"
                                    ),
                                }
                                break

        elif isinstance(last_message, dict):
            if last_message.get("role") == "assistant":
                if "function_call" in last_message and isinstance(
                    last_message["function_call"], dict
                ):
                    extracted_fc_from_message = last_message["function_call"]
                elif "tool_calls" in last_message and isinstance(
                    last_message["tool_calls"], list
                ):
                    for tc in last_message["tool_calls"]:
                        if isinstance(tc, dict) and tc.get("type") == "function":
                            if "function" in tc and isinstance(tc["function"], dict):
                                extracted_fc_from_message = tc["function"]
                                break
        else:
            return EvaluateResult(
                score=0.0,
                reason=f"Unexpected type for last message: {type(last_message)}.",
                metrics={
                    "error": MetricResult(
                        score=0.0,
                        reason="Invalid message type for function call extraction.",
                        success=False,
                    )
                },
            )

        if extracted_fc_from_message:
            function_call = extracted_fc_from_message

        if not function_call:  # Check again if function_call is still None or empty
            return EvaluateResult(
                score=0.0,
                reason="No function call found in messages.",
                metrics={
                    "error": MetricResult(
                        score=0.0,
                        reason="No function call found in messages",
                        success=False,
                    )
                },
            )

    # Normalize expected schema
    if expected_schema is None:
        return EvaluateResult(
            score=0.0,
            reason="No expected schema provided for comparison.",
            metrics={
                "error": MetricResult(
                    score=0.0, reason="No expected schema provided", success=False
                )
            },
        )

    expected_schema = normalize_schema(expected_schema)

    # Extract function name and arguments
    function_name = function_call.get("name", "")
    arguments_str = function_call.get("arguments", "{}")

    # Parse arguments JSON
    try:
        if isinstance(arguments_str, str):
            parsed_arguments = json.loads(arguments_str)
        else:
            parsed_arguments = arguments_str
    except json.JSONDecodeError:
        return EvaluateResult(
            score=0.0,
            reason=f"Invalid JSON in function arguments: {arguments_str}",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    reason=f"Invalid JSON in function arguments: {arguments_str}",
                    success=False,
                )
            },
        )

    # 1. Function name match
    expected_name = expected_schema.get("name", "")
    name_match = function_name == expected_name
    name_score = 1.0 if name_match else 0.0
    name_reason = f"Function name {'matches' if name_match else 'does not match'}: expected '{expected_name}', got '{function_name}'"
    metrics["function_name_match"] = MetricResult(
        score=name_score, reason=name_reason, success=name_match
    )

    # If function name doesn't match, return low score immediately
    if not name_match:
        return EvaluateResult(
            score=0.1,  # Some small score for partial matching attempts
            reason=name_reason,
            metrics=metrics,
        )

    # 2. Create schemas for comparison
    expected_args_schema = expected_schema.get("arguments", {})

    # Create a schema representation of the actual arguments
    actual_args_schema: Dict[str, Any] = {}
    for arg_name, arg_value in parsed_arguments.items():
        # Infer type from value
        if isinstance(arg_value, str):
            arg_type = "string"
        elif isinstance(arg_value, (int, float)):
            arg_type = "number"
        elif isinstance(arg_value, bool):
            arg_type = "boolean"
        elif isinstance(arg_value, list):
            arg_type = "array"
        elif isinstance(arg_value, dict):
            arg_type = "object"
        elif arg_value is None:
            arg_type = "null"
        else:
            arg_type = "any"

        actual_args_schema[arg_name] = {"type": arg_type}

        # For nested objects, create subschema
        if arg_type == "object" and isinstance(arg_value, dict):
            properties_dict: Dict[str, Any] = {}
            actual_args_schema[arg_name]["properties"] = properties_dict

            for sub_name, sub_value in arg_value.items():
                if isinstance(sub_value, str):
                    sub_type = "string"
                elif isinstance(sub_value, (int, float)):
                    sub_type = "number"
                elif isinstance(sub_value, bool):
                    sub_type = "boolean"
                elif isinstance(sub_value, list):
                    sub_type = "array"
                elif isinstance(sub_value, dict):
                    sub_type = "object"
                elif sub_value is None:
                    sub_type = "null"
                else:
                    sub_type = "any"

                properties_dict[sub_name] = {"type": sub_type}

    # 3. Extract schema properties
    expected_properties = extract_schema_properties(
        {"properties": expected_args_schema}
    )
    actual_properties = extract_schema_properties({"properties": actual_args_schema})

    # 4. Calculate Jaccard similarity
    schema_similarity = calculate_jaccard_similarity(
        expected_properties, actual_properties
    )

    # 5. Create detailed comparison report
    missing_props = expected_properties - actual_properties
    extra_props = actual_properties - expected_properties
    matching_props = expected_properties.intersection(actual_properties)

    comparison_details = []

    if matching_props:
        comparison_details.append(f"Matching properties ({len(matching_props)}):")
        for prop, prop_type in sorted(matching_props):
            comparison_details.append(f"  - {prop}: {prop_type}")

    if missing_props:
        comparison_details.append(f"Missing properties ({len(missing_props)}):")
        for prop, prop_type in sorted(missing_props):
            comparison_details.append(f"  - {prop}: {prop_type}")

    if extra_props:
        comparison_details.append(f"Extra properties ({len(extra_props)}):")
        for prop, prop_type in sorted(extra_props):
            comparison_details.append(f"  - {prop}: {prop_type}")

    schema_comparison_reason = "\n".join(comparison_details)

    metrics["schema_similarity"] = MetricResult(
        score=schema_similarity,
        reason=f"Schema similarity: {schema_similarity:.2f}\n{schema_comparison_reason}",
        success=schema_similarity == 1.0,
    )

    # 6. Calculate final score
    # Name match is critical but schema similarity is also important
    # Weight: 30% name match, 70% schema similarity
    final_score = (name_score * 0.3) + (schema_similarity * 0.7)
    final_reason = f"Final score based on name match ({name_score*0.3:.2f}) and schema similarity ({schema_similarity*0.7:.2f})."
    return EvaluateResult(score=final_score, reason=final_reason, metrics=metrics)


@reward_function  # Added decorator
def llm_judge_reward(
    messages: Union[List[Message], List[Dict[str, Any]]],  # Updated type
    ground_truth: Optional[Union[List[Message], List[Dict[str, Any]]]] = None,  # Added
    function_call: Optional[Dict[str, Any]] = None,
    expected_schema: Optional[Union[Dict[str, Any], str]] = None,
    expected_behavior: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    **kwargs,
) -> EvaluateResult:
    """
    Evaluate a function call using an LLM (GPT-4o-mini) as a judge.
    The model's response (containing the function call) is assumed to be `messages[-1]`.

    This reward function sends the function call and expected behavior to an LLM
    to evaluate the quality and correctness of the function call.

    Args:
        messages: List of conversation messages, where `messages[-1]` is the model's response.
        ground_truth: Optional. Expected assistant response trajectory.
        function_call: The function call to evaluate (if not provided, extracts from `messages[-1]`).
        expected_schema: The expected schema for the function call.
        expected_behavior: Description of the expected behavior for the function call.
        openai_api_key: OpenAI API key (if not provided, uses environment variable)
        model: Model to use for evaluation (default: gpt-4o-mini)
        temperature: Temperature for the model generation (default: 0.0)
        **kwargs: Additional keyword arguments

    Returns:
        EvaluateResult with score and metrics
    """
    # Check if OpenAI is available
    if OpenAI is None:
        return EvaluateResult(
            score=0.0,
            reason="OpenAI package not installed.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    reason="OpenAI package not installed. Install it with: pip install openai",
                    success=False,
                )
            },
        )

    metrics = {}

    # Extract function call from messages if not provided directly
    if function_call is None:
        if not messages:
            return EvaluateResult(
                score=0.0,
                reason="No messages provided to extract function call.",
                metrics={
                    "error": MetricResult(
                        score=0.0, reason="No messages provided", success=False
                    )
                },
            )

        last_message = messages[-1]
        extracted_fc_from_message = None

        if isinstance(last_message, Message):
            if last_message.role == "assistant":
                if last_message.function_call:
                    extracted_fc_from_message = last_message.function_call
                elif last_message.tool_calls:
                    for tc in last_message.tool_calls:
                        if tc.type == "function":
                            extracted_fc_from_message = tc.function
                            if isinstance(extracted_fc_from_message, dict):
                                break
                            else:
                                extracted_fc_from_message = {
                                    "name": getattr(
                                        extracted_fc_from_message, "name", ""
                                    ),
                                    "arguments": getattr(
                                        extracted_fc_from_message, "arguments", "{}"
                                    ),
                                }
                                break
        elif isinstance(last_message, dict):
            if last_message.get("role") == "assistant":
                if "function_call" in last_message and isinstance(
                    last_message["function_call"], dict
                ):
                    extracted_fc_from_message = last_message["function_call"]
                elif "tool_calls" in last_message and isinstance(
                    last_message["tool_calls"], list
                ):
                    for tc in last_message["tool_calls"]:
                        if isinstance(tc, dict) and tc.get("type") == "function":
                            if "function" in tc and isinstance(tc["function"], dict):
                                extracted_fc_from_message = tc["function"]
                                break
        else:
            return EvaluateResult(
                score=0.0,
                reason=f"Unexpected type for last message: {type(last_message)}.",
                metrics={
                    "error": MetricResult(
                        score=0.0,
                        reason="Invalid message type for function call extraction.",
                        success=False,
                    )
                },
            )

        if extracted_fc_from_message:
            function_call = extracted_fc_from_message

        if not function_call:  # Check again if function_call is still None or empty
            return EvaluateResult(
                score=0.0,
                reason="No function call found in messages.",
                metrics={
                    "error": MetricResult(
                        score=0.0,
                        reason="No function call found in messages",
                        success=False,
                    )
                },
            )

    # Get API key
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return EvaluateResult(
            score=0.0,
            reason="OpenAI API key not provided.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    reason="OpenAI API key not provided. Set it as openai_api_key parameter or OPENAI_API_KEY environment variable.",
                    success=False,
                )
            },
        )

    # Extract function name and arguments
    function_name = function_call.get("name", "")
    arguments_str = function_call.get("arguments", "{}")

    # Parse arguments JSON for formatting
    try:
        if isinstance(arguments_str, str):
            parsed_arguments = json.loads(arguments_str)
            # Format for readability
            arguments_str = json.dumps(parsed_arguments, indent=2)
        else:
            # Already parsed
            arguments_str = json.dumps(arguments_str, indent=2)
    except json.JSONDecodeError:
        arguments_str = str(arguments_str)  # Use as is if not valid JSON

    # Normalize expected schema
    if expected_schema:
        expected_schema = normalize_schema(expected_schema)
        expected_schema_str = json.dumps(expected_schema, indent=2)
    else:
        expected_schema_str = "No schema provided"

    # Set expected behavior if not provided
    if not expected_behavior:
        expected_behavior = "No specific behavior guidance provided"

    # Construct prompt for LLM
    conversation_msg = "No conversation context provided"
    if messages:
        conversation_parts = []
        for msg in messages[:-1]:  # Exclude the last message with function call
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content:
                conversation_parts.append(f"{role}: {content}")

        if conversation_parts:
            conversation_msg = "\n".join(conversation_parts)

    prompt = f"""You are evaluating the quality of a function call made by an AI assistant.
Your job is to assess whether the function call is appropriate, correctly formatted, and follows the expected behavior.

CONVERSATION CONTEXT:
{conversation_msg}

FUNCTION CALL:
Name: {function_name}
Arguments:
{arguments_str}

EXPECTED SCHEMA:
{expected_schema_str}

EXPECTED BEHAVIOR:
{expected_behavior}

Evaluate the function call and provide:
1. A score from 0.0 to 1.0 (where 1.0 is perfect)
2. A detailed explanation of your rating
3. Specific issues or strengths of the function call

Format your response as:
SCORE: [number between 0.0 and 1.0]
EXPLANATION: [your detailed explanation]
"""

    try:
        # Create OpenAI client
        client = OpenAI(api_key=api_key)

        # Call the API
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract the response
        llm_response = response.choices[0].message.content or ""

        # Parse the score and explanation
        score_match = re.search(r"SCORE:\s*([\d.]+)", llm_response)
        explanation_match = re.search(r"EXPLANATION:\s*(.*)", llm_response, re.DOTALL)

        if score_match:
            try:
                score = float(score_match.group(1))
                # Ensure score is in range [0, 1]
                score = max(0.0, min(score, 1.0))
            except ValueError:
                score = 0.5  # Default if parsing fails
        else:
            score = 0.5  # Default if no score found

        explanation = (
            explanation_match.group(1).strip()
            if explanation_match
            else "No explanation provided"
        )

        # Create metrics
        metrics["llm_judge"] = MetricResult(
            score=score,
            reason=explanation,
            success=score >= 0.8,  # Assuming high score means success
        )

        return EvaluateResult(score=score, reason=explanation, metrics=metrics)

    except Exception as e:
        return EvaluateResult(
            score=0.0,
            reason=f"Error calling OpenAI API: {str(e)}",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    reason=f"Error calling OpenAI API: {str(e)}",
                    success=False,
                )
            },
        )


@reward_function  # Added decorator
def composite_function_call_reward(
    messages: Union[List[Message], List[Dict[str, Any]]],  # Updated type
    ground_truth: Optional[Union[List[Message], List[Dict[str, Any]]]] = None,  # Added
    function_call: Optional[Dict[str, Any]] = None,
    expected_schema: Optional[Union[Dict[str, Any], str]] = None,
    expected_behavior: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o-mini",
    weights: Optional[Dict[str, float]] = None,
    **kwargs,
) -> EvaluateResult:
    """
    Combined reward function that evaluates function calls using both schema validation and LLM judgment.
    The model's response (containing the function call) is assumed to be `messages[-1]`.

    Args:
        messages: List of conversation messages, where `messages[-1]` is the model's response.
        ground_truth: Optional. Expected assistant response trajectory.
        function_call: The function call to evaluate (if not provided, extracts from `messages[-1]`).
        expected_schema: The expected schema for the function call.
        expected_behavior: Description of the expected behavior for the function call.
        openai_api_key: OpenAI API key (if not provided, uses environment variable)
        llm_model: Model to use for LLM evaluation (default: gpt-4o-mini)
        weights: Dictionary of weights for each component (default: {"schema": 0.5, "llm": 0.5})
        **kwargs: Additional keyword arguments

    Returns:
        EvaluateResult with score and metrics
    """
    # Default weights
    if weights is None:
        weights = {"schema": 0.5, "llm": 0.5}

    # Ensure weights sum to 1.0
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    # Run schema validation
    schema_result = schema_jaccard_reward(
        messages=messages,
        ground_truth=ground_truth,  # Pass through
        function_call=function_call,
        expected_schema=expected_schema,
        **kwargs,
    )

    # Run LLM judge evaluation if expected_behavior is provided
    if expected_behavior:
        llm_result = llm_judge_reward(
            messages=messages,
            ground_truth=ground_truth,  # Pass through
            function_call=function_call,
            expected_schema=expected_schema,
            expected_behavior=expected_behavior,
            openai_api_key=openai_api_key,
            model=llm_model,
            **kwargs,
        )
    else:
        # Skip LLM evaluation if no behavior specified
        llm_result = EvaluateResult(
            score=0.0,
            reason="LLM judge skipped: No expected behavior provided.",
            metrics={
                "llm_judge": MetricResult(
                    score=0.0,
                    reason="Skipped: No expected behavior provided",
                    success=True,  # Success because it's an expected skip
                )
            },
        )

    # Combine metrics
    combined_metrics = {}

    # Add schema metrics with "schema_" prefix
    for (
        key,
        metric_val,
    ) in schema_result.metrics.items():  # Renamed to metric_val to avoid conflict
        combined_metrics[f"schema_{key}"] = metric_val

    # Add llm metrics with "llm_" prefix
    for key, metric_val in llm_result.metrics.items():  # Renamed to metric_val
        combined_metrics[f"llm_{key}"] = metric_val

    # Add summary metrics
    combined_metrics["schema_score"] = MetricResult(
        score=schema_result.score,
        reason=f"Schema validation score: {schema_result.score:.2f}",
        success=schema_result.score == 1.0,
    )

    combined_metrics["llm_score"] = MetricResult(
        score=llm_result.score,
        reason=f"LLM judge score: {llm_result.score:.2f}",
        success=llm_result.score >= 0.8,  # Assuming high score means success
    )

    # Calculate weighted final score
    schema_weight = normalized_weights.get("schema", 0.5)
    llm_weight = normalized_weights.get("llm", 0.5)

    final_score = (schema_result.score * schema_weight) + (
        llm_result.score * llm_weight
    )
    final_reason = f"Composite score. Schema ({schema_result.score:.2f} * {schema_weight:.2f}) + LLM ({llm_result.score:.2f} * {llm_weight:.2f})."

    # Add weight information
    combined_metrics["weights"] = MetricResult(
        score=0.0,  # Not a real score
        reason=f"Weights used - Schema: {schema_weight:.2f}, LLM: {llm_weight:.2f}",
        success=True,  # Informational metric
    )

    return EvaluateResult(
        score=final_score, reason=final_reason, metrics=combined_metrics
    )


# JSON schema reward functions have been moved to json_schema.py module
