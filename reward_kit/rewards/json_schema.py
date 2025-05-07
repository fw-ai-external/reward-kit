from typing import Dict, List, Any, Optional, Union
import json
import re

from ..models import RewardOutput, MetricRewardOutput
from .function_calling import (
    calculate_jaccard_similarity,
    extract_schema_properties,
    normalize_schema,
)


def json_schema_reward(
    messages: List[Dict[str, str]],
    original_messages: Optional[List[Dict[str, str]]] = None,
    json_content: Optional[Union[Dict[str, Any], str]] = None,
    expected_schema: Optional[Union[Dict[str, Any], str]] = None,
    **kwargs,
) -> RewardOutput:
    """
    Evaluate JSON content against an expected schema using Jaccard similarity.

    This reward function compares the structure of JSON content against an
    expected schema and calculates a similarity score using Jaccard similarity.
    It repurposes the same approach used for function calling validation but for
    general JSON schema validation.

    Args:
        messages: The conversation messages
        original_messages: Original conversation context (optional)
        json_content: The JSON content to evaluate (if not provided, extracts
                      from message)
        expected_schema: The expected schema for the JSON content
        **kwargs: Additional keyword arguments

    Returns:
        RewardOutput with score and metrics
    """
    metrics = {}

    # Extract JSON content from messages if not provided directly
    if json_content is None:
        if not messages:
            return RewardOutput(
                score=0.0,
                metrics={
                    "error": MetricRewardOutput(
                        score=0.0, reason="No messages provided"
                    )
                },
            )

        last_message = messages[-1]
        content = last_message.get("content", "")

        # Try to extract JSON from the message content
        if content:
            try:
                # First look for JSON code blocks
                pattern = r"```(?:json)?\s*([\s\S]*?)```"
                code_blocks = re.findall(pattern, content)
                if code_blocks:
                    json_content = code_blocks[0]
                else:
                    # Try to find JSON-like content in the message
                    json_matches = re.findall(r"\{.*\}", content, re.DOTALL)
                    if json_matches:
                        json_content = json_matches[0]
            except Exception:
                pass

        if not json_content:
            return RewardOutput(
                score=0.0,
                metrics={
                    "error": MetricRewardOutput(
                        score=0.0, reason="No JSON content found in messages"
                    )
                },
            )

    # Normalize expected schema
    if expected_schema is None:
        return RewardOutput(
            score=0.0,
            metrics={
                "error": MetricRewardOutput(
                    score=0.0, reason="No expected schema provided"
                )
            },
        )

    expected_schema = normalize_schema(expected_schema)

    # Parse JSON content
    try:
        if isinstance(json_content, str):
            parsed_content = json.loads(json_content)
        else:
            parsed_content = json_content
    except json.JSONDecodeError:
        return RewardOutput(
            score=0.0,
            metrics={
                "error": MetricRewardOutput(
                    score=0.0, reason=f"Invalid JSON content: {json_content}"
                )
            },
        )

    # Function to recursively build a schema from content
    def build_schema_from_content(content: Any) -> Dict[str, Any]:
        if isinstance(content, dict):
            schema: Dict[str, Any] = {"type": "object", "properties": {}}
            for key, value in content.items():
                if isinstance(schema["properties"], dict):
                    schema["properties"][key] = build_schema_from_content(value)
            return schema
        elif isinstance(content, list):
            if content:
                # Use the first item as reference for array items
                return {
                    "type": "array",
                    "items": build_schema_from_content(content[0]),
                }
            return {"type": "array"}
        elif isinstance(content, str):
            return {"type": "string"}
        elif isinstance(content, bool):
            return {"type": "boolean"}
        elif isinstance(content, (int, float)):
            return {"type": "number"}
        elif content is None:
            return {"type": "null"}
        else:
            return {"type": "any"}

    # Build schema for the actual content
    content_schema = build_schema_from_content(parsed_content)

    # Extract schema properties
    expected_properties = extract_schema_properties(expected_schema)
    actual_properties = extract_schema_properties(content_schema)

    # Calculate Jaccard similarity
    schema_similarity = calculate_jaccard_similarity(
        expected_properties, actual_properties
    )

    # Create detailed comparison report
    missing_props = expected_properties - actual_properties
    extra_props = actual_properties - expected_properties
    matching_props = expected_properties.intersection(actual_properties)

    comparison_details = []

    if matching_props:
        comparison_details.append(
            f"Matching properties ({len(matching_props)}):"
        )
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

    metrics["schema_similarity"] = MetricRewardOutput(
        score=schema_similarity,
        reason=f"Schema similarity: {schema_similarity:.2f}\n{schema_comparison_reason}",
    )

    # Calculate final score based on schema similarity
    final_score = schema_similarity

    return RewardOutput(score=final_score, metrics=metrics)


def json_schema_reward_with_llm_judge(
    messages: List[Dict[str, str]],
    original_messages: Optional[List[Dict[str, str]]] = None,
    json_content: Optional[Union[Dict[str, Any], str]] = None,
    expected_schema: Optional[Union[Dict[str, Any], str]] = None,
    expected_behavior: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
    **kwargs,
) -> RewardOutput:
    """
    Combined reward function that evaluates JSON content using both schema
    validation and LLM judgment.

    Args:
        messages: The conversation messages
        original_messages: Original conversation context (optional)
        json_content: The JSON content to evaluate (if not provided, extracts
                      from message)
        expected_schema: The expected schema for the JSON content
        expected_behavior: Description of the expected behavior/content
        openai_api_key: OpenAI API key (if not provided, uses environment variable)
        model: Model to use for LLM evaluation (default: gpt-4o-mini)
        temperature: Temperature for the model generation (default: 0.0)
        weights: Dictionary of weights for each component
                (default: {"schema": 0.7, "llm": 0.3})
        **kwargs: Additional keyword arguments

    Returns:
        RewardOutput with score and metrics
    """
    # Import OpenAI at call time to make this optional
    try:
        from openai import OpenAI
    except ImportError:
        return RewardOutput(
            score=0.0,
            metrics={
                "error": MetricRewardOutput(
                    score=0.0,
                    reason="OpenAI package not installed. Install it with: pip install openai",
                )
            },
        )

    # Default weights
    if weights is None:
        weights = {"schema": 0.7, "llm": 0.3}

    # Ensure weights sum to 1.0
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    # Run schema validation
    schema_result = json_schema_reward(
        messages=messages,
        original_messages=original_messages,
        json_content=json_content,
        expected_schema=expected_schema,
        **kwargs,
    )

    # Skip LLM evaluation if no behavior specified or OpenAI is not available
    if not expected_behavior:
        llm_score = 0.0
        llm_reason = "Skipped: No expected behavior provided"
    else:
        # Extract and parse JSON content if not done already
        if json_content is None:
            # Use error from schema validation if it failed to extract JSON
            if "error" in schema_result.metrics:
                return schema_result

            # Otherwise, try to get JSON content from the last message
            last_message = messages[-1]
            content = last_message.get("content", "")

            # Try to extract JSON from the message content
            json_str = ""
            try:
                # First look for JSON code blocks
                pattern = r"```(?:json)?\s*([\s\S]*?)```"
                code_blocks = re.findall(pattern, content)
                if code_blocks:
                    json_str = code_blocks[0]
                else:
                    # Try to find JSON-like content in the message
                    json_matches = re.findall(r"\{.*\}", content, re.DOTALL)
                    if json_matches:
                        json_str = json_matches[0]
            except Exception:
                pass

            # Try to parse the extracted content
            try:
                if json_str:
                    json_content = json.loads(json_str)
            except json.JSONDecodeError:
                json_content = json_str

        # Format JSON content for readability if it's a dictionary
        if isinstance(json_content, dict):
            json_str = json.dumps(json_content, indent=2)
        else:
            json_str = str(json_content)

        # Format expected schema for prompt
        expected_schema_str = (
            json.dumps(expected_schema, indent=2)
            if expected_schema
            else "No schema provided"
        )

        # Construct prompt for LLM
        conversation_msg = "No conversation context provided"
        if messages:
            conversation_parts = []
            # Exclude the last message with JSON response
            for msg in messages[:-1]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role and content:
                    conversation_parts.append(f"{role}: {content}")

            if conversation_parts:
                conversation_msg = "\n".join(conversation_parts)

        prompt = f"""You are evaluating the quality of JSON content provided by an AI assistant. 
Your job is to assess whether the JSON structure and content is appropriate, correctly formatted, 
and follows the expected schema and behavior.

CONVERSATION CONTEXT:
{conversation_msg}

JSON CONTENT:
{json_str}

EXPECTED SCHEMA:
{expected_schema_str}

EXPECTED BEHAVIOR/CONTENT:
{expected_behavior}

Evaluate the JSON content and provide:
1. A score from 0.0 to 1.0 (where 1.0 is perfect)
2. A detailed explanation of your rating
3. Specific issues or strengths of the JSON content

Format your response as:
SCORE: [number between 0.0 and 1.0]
EXPLANATION: [your detailed explanation]
"""

        try:
            # Get API key
            import os

            api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided")

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
            explanation_match = re.search(
                r"EXPLANATION:\s*(.*)", llm_response, re.DOTALL
            )

            if score_match:
                try:
                    llm_score = float(score_match.group(1))
                    # Ensure score is in range [0, 1]
                    llm_score = max(0.0, min(llm_score, 1.0))
                except ValueError:
                    llm_score = 0.5  # Default if parsing fails
            else:
                llm_score = 0.5  # Default if no score found

            llm_reason = (
                explanation_match.group(1).strip()
                if explanation_match
                else "No explanation provided"
            )

        except Exception as e:
            llm_score = 0.0
            llm_reason = f"Error calling OpenAI API: {str(e)}"

    # Combine metrics
    combined_metrics = {}

    # Add schema metrics with "schema_" prefix
    for key, metric in schema_result.metrics.items():
        if key != "schema_similarity":
            combined_metrics[f"schema_{key}"] = metric
        else:
            combined_metrics[key] = metric

    # Add llm metrics
    combined_metrics["llm_judge"] = MetricRewardOutput(
        score=llm_score, reason=llm_reason
    )

    # Add summary metrics
    combined_metrics["schema_score"] = MetricRewardOutput(
        score=schema_result.score,
        reason=f"Schema validation score: {schema_result.score:.2f}",
    )

    combined_metrics["llm_score"] = MetricRewardOutput(
        score=llm_score, reason=f"LLM judge score: {llm_score:.2f}"
    )

    # Calculate weighted final score
    schema_weight = normalized_weights.get("schema", 0.7)
    llm_weight = normalized_weights.get("llm", 0.3)

    final_score = (schema_result.score * schema_weight) + (
        llm_score * llm_weight
    )

    # Add weight information
    combined_metrics["weights"] = MetricRewardOutput(
        score=0.0,  # Not a real score
        reason=f"Weights used - Schema: {schema_weight:.2f}, LLM: {llm_weight:.2f}",
    )

    return RewardOutput(score=final_score, metrics=combined_metrics)
