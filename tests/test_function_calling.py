import json
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pytest

from reward_kit.models import EvaluateResult
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
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "Let me check the weather."},
            ],
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="exact",
        )

        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        assert "function_name_match" in result.metrics
        assert "arguments_match" in result.metrics
        assert result.metrics["function_name_match"].score == 1.0
        assert result.metrics["arguments_match"].score == 1.0
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

        parsed_name = "fetch_weather"
        parsed_args = {"location": "New York", "unit": "celsius"}

        result = match_function_call(
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "Let me check the weather."},
            ],
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="exact",
        )

        assert isinstance(result, EvaluateResult)
        assert result.score < 1.0
        assert "function_name_match" in result.metrics
        fn_match_metric = result.metrics["function_name_match"]
        assert fn_match_metric.score == 0.0
        reason_text = fn_match_metric.reason
        assert reason_text is not None
        assert "Function name does not match" in reason_text  # type: ignore[operator]

        assert result["score"] < 1.0
        assert result["metrics"]["function_name_match"]["score"] == 0.0
        fn_match_metric_dict_reason = result["metrics"]["function_name_match"]["reason"]
        assert fn_match_metric_dict_reason is not None
        assert "Function name does not match" in fn_match_metric_dict_reason  # type: ignore[operator]

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
        parsed_args = {"location": "New York"}

        result = match_function_call(
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "Let me check the weather."},
            ],
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="exact",
        )

        assert isinstance(result, EvaluateResult)
        assert result.score < 1.0
        assert "arguments_match" in result.metrics
        arg_match_metric = result.metrics["arguments_match"]
        assert arg_match_metric.score < 1.0
        reason_text = arg_match_metric.reason
        assert reason_text is not None
        assert "Missing argument" in reason_text  # type: ignore[operator]

        assert result["score"] < 1.0
        assert result["metrics"]["arguments_match"]["score"] < 1.0
        arg_match_metric_dict_reason = result["metrics"]["arguments_match"]["reason"]
        assert arg_match_metric_dict_reason is not None
        assert "Missing argument" in arg_match_metric_dict_reason  # type: ignore[operator]

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
            "extra_param": "value",
        }

        result = match_function_call(
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "Let me check the weather."},
            ],
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="exact",
        )

        assert isinstance(result, EvaluateResult)
        assert result.score < 1.0
        assert "arguments_match" in result.metrics
        arg_match_metric = result.metrics["arguments_match"]
        assert arg_match_metric.score < 1.0
        reason_text = arg_match_metric.reason
        assert reason_text is not None
        assert "Unexpected argument" in reason_text  # type: ignore[operator]

        assert result["score"] < 1.0
        assert result["metrics"]["arguments_match"]["score"] < 1.0
        arg_match_metric_dict_reason = result["metrics"]["arguments_match"]["reason"]
        assert arg_match_metric_dict_reason is not None
        assert "Unexpected argument" in arg_match_metric_dict_reason  # type: ignore[operator]

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
            "extra_param": "value",
        }

        result = match_function_call(
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "Let me check the weather."},
            ],
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="permissive",
        )

        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        assert "function_name_match" in result.metrics
        assert "arguments_match" in result.metrics
        assert result.metrics["function_name_match"].score == 1.0
        assert result.metrics["arguments_match"].score == 1.0
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
            "temperature": "25",
        }

        result = match_function_call(
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "Let me check the weather."},
            ],
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="exact",
        )

        assert isinstance(result, EvaluateResult)
        assert result.score < 1.0
        assert "arguments_match" in result.metrics
        arg_match_metric = result.metrics["arguments_match"]
        assert arg_match_metric.score < 1.0
        reason_text = arg_match_metric.reason
        assert reason_text is not None
        assert "Type mismatch" in reason_text  # type: ignore[operator]

        assert result["score"] < 1.0
        assert result["metrics"]["arguments_match"]["score"] < 1.0
        arg_match_metric_dict_reason = result["metrics"]["arguments_match"]["reason"]
        assert arg_match_metric_dict_reason is not None
        assert "Type mismatch" in arg_match_metric_dict_reason  # type: ignore[operator]

    def test_calculate_jaccard_similarity(self):
        """Test Jaccard similarity calculation."""
        set1 = {"a", "b", "c"}
        set2 = {"a", "b", "c"}
        assert calculate_jaccard_similarity(set1, set2) == 1.0

        set1 = {"a", "b", "c"}
        set2 = {"d", "e", "f"}
        assert calculate_jaccard_similarity(set1, set2) == 0.0

        set1 = {"a", "b", "c"}
        set2 = {"b", "c", "d"}
        assert calculate_jaccard_similarity(set1, set2) == 0.5

        set1 = set()
        set2 = set()
        assert calculate_jaccard_similarity(set1, set2) == 1.0

    def test_extract_schema_properties(self):
        """Test extraction of properties from JSON schema."""
        schema = {
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
            }
        }
        properties = extract_schema_properties(schema)
        assert properties == {("name", "string"), ("age", "number")}

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
        assert properties == {
            ("user", "object"),
            ("user.firstName", "string"),
            ("user.lastName", "string"),
        }

    def test_schema_jaccard_reward_exact_match(self):
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
        test_messages: List[Dict[str, Any]] = [
            {"role": "user", "content": "What's the weather?"},
            assistant_message_content_with_tool_call,
        ]
        result = schema_jaccard_reward(
            messages=test_messages,
            ground_truth=ground_truth_data,
        )
        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        reason_text = result.reason
        assert reason_text is not None
        assert "Exact tool match evaluation score: 1.0" in reason_text  # type: ignore[operator]

    def test_schema_jaccard_reward_mismatch(self):
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
        test_messages: List[Dict[str, Any]] = [
            {"role": "user", "content": "What's the weather?"},
            assistant_message_content_with_tool_call,
        ]
        result = schema_jaccard_reward(
            messages=test_messages,
            ground_truth=ground_truth_data,
        )
        assert isinstance(result, EvaluateResult)
        assert result.score == 0.0
        reason_text = result.reason
        assert reason_text is not None
        assert "Exact tool match evaluation score: 0.0" in reason_text  # type: ignore[operator]

    def test_schema_jaccard_reward_wrong_function_name(self):
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
        test_messages: List[Dict[str, Any]] = [
            {"role": "user", "content": "What's the weather?"},
            assistant_message_content_with_tool_call,
        ]
        result = schema_jaccard_reward(
            messages=test_messages,
            ground_truth=ground_truth_data,
        )
        assert isinstance(result, EvaluateResult)
        assert result.score == 0.0
        reason_text = result.reason
        assert reason_text is not None
        assert "Exact tool match evaluation score: 0.0" in reason_text  # type: ignore[operator]

    def test_nested_schema_exact_match(self):
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
        test_messages: List[Dict[str, Any]] = [
            {"role": "user", "content": "Create a user for John Doe"},
            assistant_message_content_with_tool_call,
        ]
        result = schema_jaccard_reward(
            messages=test_messages,
            ground_truth=ground_truth_data,
        )
        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        reason_text = result.reason
        assert reason_text is not None
        assert "Exact tool match evaluation score: 1.0" in reason_text  # type: ignore[operator]

    def test_llm_judge_reward_delegation(self):
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
        test_messages: List[Dict[str, Any]] = [
            {"role": "user", "content": "What's the weather?"},
            assistant_message_content_with_tool_call,
        ]
        result = llm_judge_reward(
            messages=test_messages,
            ground_truth=ground_truth_data,
        )
        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        reason_text = result.reason
        assert reason_text is not None
        assert "Exact tool match evaluation score: 1.0" in reason_text  # type: ignore[operator]
        assert "llm_judge" not in result.metrics

    def test_composite_function_call_reward_delegation(self):
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
        test_messages: List[Dict[str, Any]] = [
            {"role": "user", "content": "What's the weather?"},
            assistant_message_content_with_tool_call,
        ]
        result = composite_function_call_reward(
            messages=test_messages,
            ground_truth=ground_truth_data,
        )
        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        reason_text = result.reason
        assert reason_text is not None
        assert "Exact tool match evaluation score: 1.0" in reason_text  # type: ignore[operator]
        assert "schema_score" not in result.metrics
        assert "llm_score" not in result.metrics
        assert "weights" not in result.metrics
