import json
from typing import Any, Dict, List, Tuple, cast
from unittest.mock import MagicMock, patch

import pytest

from reward_kit.models import EvaluateResult  # Changed
from reward_kit.rewards.function_calling import (
    calculate_jaccard_similarity,
    composite_function_call_reward,
    extract_schema_properties,
    llm_judge_reward,
    match_function_call,
    schema_jaccard_reward,
)


class TestFunctionCalling:
    """Tests for the function_calling reward module."""

    def test_exact_match(self):
        """Test exact match of function name and arguments."""
        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
        }

        parsed_name = "get_weather"
        parsed_args = {"location": "New York", "unit": "celsius"}

        result = match_function_call(
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "What's the weather?"},
                    {"role": "assistant", "content": "Let me check the weather."},
                ],
            ),
            # original_messages removed
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="exact",
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert result.score == 1.0
        assert "function_name_match" in result.metrics
        assert "arguments_match" in result.metrics
        assert result.metrics["function_name_match"].score == 1.0
        assert result.metrics["arguments_match"].score == 1.0
        # Dictionary access
        assert result["score"] == 1.0
        assert result["metrics"]["function_name_match"]["score"] == 1.0
        assert result["metrics"]["arguments_match"]["score"] == 1.0

    def test_wrong_function_name(self):
        """Test with incorrect function name."""
        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
        }

        parsed_name = "fetch_weather"  # Wrong name
        parsed_args = {"location": "New York", "unit": "celsius"}

        result = match_function_call(
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "What's the weather?"},
                    {"role": "assistant", "content": "Let me check the weather."},
                ],
            ),
            # original_messages removed
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="exact",
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert result.score < 1.0
        assert "function_name_match" in result.metrics
        assert result.metrics["function_name_match"].score == 0.0
        assert (
            result.metrics["function_name_match"].reason is not None
            and "Function name does not match"
            in result.metrics["function_name_match"].reason
        )
        # Dictionary access
        assert result["score"] < 1.0
        assert result["metrics"]["function_name_match"]["score"] == 0.0
        assert (
            result["metrics"]["function_name_match"]["reason"] is not None
            and "Function name does not match"
            in result["metrics"]["function_name_match"]["reason"]
        )

    def test_missing_required_argument(self):
        """Test with missing required argument."""
        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
        }

        parsed_name = "get_weather"
        parsed_args = {
            "location": "New York"
            # Missing "unit" argument
        }

        result = match_function_call(
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "What's the weather?"},
                    {"role": "assistant", "content": "Let me check the weather."},
                ],
            ),
            # original_messages removed
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="exact",
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert result.score < 1.0
        assert "arguments_match" in result.metrics
        assert result.metrics["arguments_match"].score < 1.0
        assert (
            result.metrics["arguments_match"].reason is not None
            and "Missing argument" in result.metrics["arguments_match"].reason
        )
        # Dictionary access
        assert result["score"] < 1.0
        assert result["metrics"]["arguments_match"]["score"] < 1.0
        assert (
            result["metrics"]["arguments_match"]["reason"] is not None
            and "Missing argument" in result["metrics"]["arguments_match"]["reason"]
        )

    def test_extra_argument(self):
        """Test with extra argument not in schema."""
        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
        }

        parsed_name = "get_weather"
        parsed_args = {
            "location": "New York",
            "unit": "celsius",
            "extra_param": "value",  # Extra argument
        }

        result = match_function_call(
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "What's the weather?"},
                    {"role": "assistant", "content": "Let me check the weather."},
                ],
            ),
            # original_messages removed
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="exact",
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert result.score < 1.0
        assert "arguments_match" in result.metrics
        assert result.metrics["arguments_match"].score < 1.0
        assert (
            result.metrics["arguments_match"].reason is not None
            and "Unexpected argument" in result.metrics["arguments_match"].reason
        )
        # Dictionary access
        assert result["score"] < 1.0
        assert result["metrics"]["arguments_match"]["score"] < 1.0
        assert (
            result["metrics"]["arguments_match"]["reason"] is not None
            and "Unexpected argument" in result["metrics"]["arguments_match"]["reason"]
        )

    def test_permissive_mode(self):
        """Test permissive mode with extra arguments."""
        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
        }

        parsed_name = "get_weather"
        parsed_args = {
            "location": "New York",
            "unit": "celsius",
            "extra_param": "value",  # Extra argument
        }

        result = match_function_call(
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "What's the weather?"},
                    {"role": "assistant", "content": "Let me check the weather."},
                ],
            ),
            # original_messages removed
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="permissive",  # Permissive mode
        )

        assert isinstance(result, EvaluateResult)
        # In permissive mode, extra arguments are allowed
        # Attribute access
        assert result.score == 1.0
        assert "function_name_match" in result.metrics
        assert "arguments_match" in result.metrics
        assert result.metrics["function_name_match"].score == 1.0
        assert result.metrics["arguments_match"].score == 1.0
        # Dictionary access
        assert result["score"] == 1.0
        assert result["metrics"]["function_name_match"]["score"] == 1.0
        assert result["metrics"]["arguments_match"]["score"] == 1.0

    def test_wrong_argument_value_type(self):
        """Test with wrong argument value type."""
        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "temperature": {"type": "number"},
            },
        }

        parsed_name = "get_weather"
        parsed_args = {
            "location": "New York",
            "temperature": "25",  # String instead of number
        }

        result = match_function_call(
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "What's the weather?"},
                    {"role": "assistant", "content": "Let me check the weather."},
                ],
            ),
            # original_messages removed
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="exact",
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert result.score < 1.0
        assert "arguments_match" in result.metrics
        assert result.metrics["arguments_match"].score < 1.0
        assert (
            result.metrics["arguments_match"].reason is not None
            and "Type mismatch" in result.metrics["arguments_match"].reason
        )
        # Dictionary access
        assert result["score"] < 1.0
        assert result["metrics"]["arguments_match"]["score"] < 1.0
        assert (
            result["metrics"]["arguments_match"]["reason"] is not None
            and "Type mismatch" in result["metrics"]["arguments_match"]["reason"]
        )

    def test_calculate_jaccard_similarity(self):
        """Test Jaccard similarity calculation."""
        # Perfect match
        set1 = {"a", "b", "c"}
        set2 = {"a", "b", "c"}
        similarity = calculate_jaccard_similarity(set1, set2)
        assert similarity == 1.0

        # No overlap
        set1 = {"a", "b", "c"}
        set2 = {"d", "e", "f"}
        similarity = calculate_jaccard_similarity(set1, set2)
        assert similarity == 0.0

        # Partial overlap
        set1 = {"a", "b", "c"}
        set2 = {"b", "c", "d"}
        similarity = calculate_jaccard_similarity(set1, set2)
        assert similarity == 0.5  # 2/4 = 0.5

        # Empty sets
        set1 = set()
        set2 = set()
        similarity = calculate_jaccard_similarity(set1, set2)
        assert similarity == 1.0  # Both empty should be perfect match

    def test_extract_schema_properties(self):
        """Test extraction of properties from JSON schema."""
        # Simple schema
        schema = {
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
            }
        }
        properties = extract_schema_properties(schema)
        assert len(properties) == 2
        assert ("name", "string") in properties
        assert ("age", "number") in properties

        # Nested schema
        nested_schema: Dict[str, Any] = {
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "firstName": {"type": "string"},
                        "lastName": {"type": "string"},
                    },
                }
            }
        }
        properties = extract_schema_properties(nested_schema)
        assert len(properties) == 3
        assert ("user", "object") in properties
        assert ("user.firstName", "string") in properties
        assert ("user.lastName", "string") in properties

    def test_schema_jaccard_reward_exact_match(self):
        """Test schema_jaccard_reward now delegates to exact_tool_match_reward - Perfect Match."""
        # This test now verifies exact_tool_match_reward's behavior via schema_jaccard_reward
        assistant_message_content_with_tool_call = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps(
                            {"location": "New York", "unit": "celsius"}
                        ),
                    },
                }
            ],
        }
        ground_truth_data = {
            "role": "assistant",  # Role for ground_truth is illustrative, exact_tool_match_reward uses tool_calls from it
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps(
                            {"location": "New York", "unit": "celsius"}
                        ),
                    },
                }
            ],
        }

        result = schema_jaccard_reward(  # This now calls exact_tool_match_reward
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "What's the weather?"},
                    assistant_message_content_with_tool_call,
                ],
            ),
            ground_truth=ground_truth_data,
            # expected_schema is no longer used by the delegated function
        )

        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        assert isinstance(result.reason, str)
        assert "Exact tool match evaluation score: 1.0" in result.reason

    def test_schema_jaccard_reward_mismatch(self):
        """Test schema_jaccard_reward now delegates to exact_tool_match_reward - Mismatch."""
        assistant_message_content_with_tool_call = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps(
                            {"location": "New York", "unit": "fahrenheit"}
                        ),
                    },  # Different unit
                }
            ],
        }
        ground_truth_data = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps(
                            {"location": "New York", "unit": "celsius"}
                        ),
                    },
                }
            ]
        }

        result = schema_jaccard_reward(
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "What's the weather?"},
                    assistant_message_content_with_tool_call,
                ],
            ),
            ground_truth=ground_truth_data,
        )
        assert isinstance(result, EvaluateResult)
        assert result.score == 0.0
        assert isinstance(result.reason, str)
        assert "Exact tool match evaluation score: 0.0" in result.reason

    def test_schema_jaccard_reward_wrong_function_name(self):
        """Test schema_jaccard_reward (delegating) with wrong function name."""
        assistant_message_content_with_tool_call = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "fetch_weather_data",
                        "arguments": json.dumps(
                            {"location": "New York", "unit": "celsius"}
                        ),
                    },
                }
            ],
        }
        ground_truth_data = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps(
                            {"location": "New York", "unit": "celsius"}
                        ),
                    },
                }
            ]
        }
        result = schema_jaccard_reward(
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "What's the weather?"},
                    assistant_message_content_with_tool_call,
                ],
            ),
            ground_truth=ground_truth_data,
        )
        assert isinstance(result, EvaluateResult)
        assert result.score == 0.0
        assert isinstance(result.reason, str)
        assert "Exact tool match evaluation score: 0.0" in result.reason

    def test_nested_schema_exact_match(self):  # Renamed for clarity
        """Test exact_tool_match_reward (via schema_jaccard_reward) with nested objects."""
        assistant_message_content_with_tool_call = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "create_user",
                        "arguments": json.dumps(
                            {
                                "user": {
                                    "firstName": "John",
                                    "lastName": "Doe",
                                    "age": 30,
                                }
                            }
                        ),
                    },
                }
            ],
        }
        ground_truth_data = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "create_user",
                        "arguments": json.dumps(
                            {
                                "user": {
                                    "firstName": "John",
                                    "lastName": "Doe",
                                    "age": 30,
                                }
                            }
                        ),
                    },
                }
            ]
        }
        result = schema_jaccard_reward(  # This now calls exact_tool_match_reward
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "Create a user for John Doe"},
                    assistant_message_content_with_tool_call,
                ],
            ),
            ground_truth=ground_truth_data,
        )
        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        assert isinstance(result.reason, str)
        assert "Exact tool match evaluation score: 1.0" in result.reason

    # Remove @patch for OpenAI as llm_judge_reward now delegates
    def test_llm_judge_reward_delegation(self):  # Renamed and simplified
        """Test llm_judge_reward now delegates to exact_tool_match_reward."""
        assistant_message_content_with_tool_call = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps(
                            {"location": "New York", "unit": "celsius"}
                        ),
                    },
                }
            ],
        }
        ground_truth_data = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps(
                            {"location": "New York", "unit": "celsius"}
                        ),
                    },
                }
            ]
        }

        result = llm_judge_reward(  # This now calls exact_tool_match_reward
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "What's the weather?"},
                    assistant_message_content_with_tool_call,
                ],
            ),
            ground_truth=ground_truth_data,
            # Other params like expected_schema, expected_behavior, openai_api_key are no longer used by the core logic
        )

        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        assert isinstance(result.reason, str)
        assert "Exact tool match evaluation score: 1.0" in result.reason
        # Ensure no LLM-specific metrics are present if the delegation is clean
        assert "llm_judge" not in result.metrics

    # Remove @patch for OpenAI as composite_function_call_reward now delegates
    def test_composite_function_call_reward_delegation(self):  # Renamed and simplified
        """Test composite_function_call_reward now delegates to exact_tool_match_reward."""
        assistant_message_content_with_tool_call = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps(
                            {"location": "New York", "unit": "celsius"}
                        ),
                    },
                }
            ],
        }
        ground_truth_data = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps(
                            {"location": "New York", "unit": "celsius"}
                        ),
                    },
                }
            ]
        }

        result = composite_function_call_reward(  # This now calls exact_tool_match_reward
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "What's the weather?"},
                    assistant_message_content_with_tool_call,
                ],
            ),
            ground_truth=ground_truth_data,
            # Other params like expected_schema, expected_behavior, weights are no longer used by the core logic
        )

        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        assert isinstance(result.reason, str)
        assert "Exact tool match evaluation score: 1.0" in result.reason
        # Ensure no composite-specific metrics (like schema_score, llm_score, weights) are present
        assert "schema_score" not in result.metrics
        assert "llm_score" not in result.metrics
        assert "weights" not in result.metrics


# The JSON schema tests have been moved to tests/test_json_schema.py
