import pytest
import json
from reward_kit.rewards.function_calling import (
    match_function_call,
    schema_jaccard_reward,
    calculate_jaccard_similarity,
    extract_schema_properties,
    llm_judge_reward,
    composite_function_call_reward,
)
from reward_kit.models import EvaluateResult # Changed
from unittest.mock import patch, MagicMock


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
        assert result['score'] == 1.0
        assert result['metrics']["function_name_match"]['score'] == 1.0
        assert result['metrics']["arguments_match"]['score'] == 1.0

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
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "Let me check the weather."},
            ],
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
        assert result.metrics["function_name_match"].reason is not None and \
            "Function name does not match" in result.metrics["function_name_match"].reason
        # Dictionary access
        assert result['score'] < 1.0
        assert result['metrics']["function_name_match"]['score'] == 0.0
        assert result['metrics']["function_name_match"]['reason'] is not None and \
            "Function name does not match" in result['metrics']["function_name_match"]['reason']

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
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "Let me check the weather."},
            ],
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
        assert result.metrics["arguments_match"].reason is not None and \
            "Missing argument" in result.metrics["arguments_match"].reason
        # Dictionary access
        assert result['score'] < 1.0
        assert result['metrics']["arguments_match"]['score'] < 1.0
        assert result['metrics']["arguments_match"]['reason'] is not None and \
            "Missing argument" in result['metrics']["arguments_match"]['reason']

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
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "Let me check the weather."},
            ],
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
        assert result.metrics["arguments_match"].reason is not None and \
            "Unexpected argument" in result.metrics["arguments_match"].reason
        # Dictionary access
        assert result['score'] < 1.0
        assert result['metrics']["arguments_match"]['score'] < 1.0
        assert result['metrics']["arguments_match"]['reason'] is not None and \
            "Unexpected argument" in result['metrics']["arguments_match"]['reason']

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
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "Let me check the weather."},
            ],
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
        assert result['score'] == 1.0
        assert result['metrics']["function_name_match"]['score'] == 1.0
        assert result['metrics']["arguments_match"]['score'] == 1.0

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
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "Let me check the weather."},
            ],
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
        assert result.metrics["arguments_match"].reason is not None and \
            "Type mismatch" in result.metrics["arguments_match"].reason
        # Dictionary access
        assert result['score'] < 1.0
        assert result['metrics']["arguments_match"]['score'] < 1.0
        assert result['metrics']["arguments_match"]['reason'] is not None and \
            "Type mismatch" in result['metrics']["arguments_match"]['reason']

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
        schema = {
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
        properties = extract_schema_properties(schema)
        assert len(properties) == 3
        assert ("user", "object") in properties
        assert ("user.firstName", "string") in properties
        assert ("user.lastName", "string") in properties

    def test_schema_jaccard_reward_exact_match(self):
        """Test schema Jaccard reward with exact match."""
        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
        }

        function_call = {
            "name": "get_weather",
            "arguments": json.dumps(
                {"location": "New York", "unit": "celsius"}
            ),
        }

        result = schema_jaccard_reward(
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "function_call": function_call},
            ],
            ground_truth=None,
            expected_schema=expected_schema,
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert result.score > 0.9  # Should be very high
        assert "function_name_match" in result.metrics
        assert "schema_similarity" in result.metrics
        assert result.metrics["function_name_match"].score == 1.0
        assert result.metrics["schema_similarity"].score > 0.9
        # Dictionary access
        assert result['score'] > 0.9
        assert result['metrics']["function_name_match"]['score'] == 1.0
        assert result['metrics']["schema_similarity"]['score'] > 0.9

    def test_schema_jaccard_reward_partial_match(self):
        """Test schema Jaccard reward with partial match."""
        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
                "date": {"type": "string"},
            },
        }

        function_call = {
            "name": "get_weather",
            "arguments": json.dumps(
                {
                    "location": "New York",
                    "unit": "celsius",
                    "extraParam": "value",  # Extra param instead of date
                }
            ),
        }

        result = schema_jaccard_reward(
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "function_call": function_call},
            ],
            ground_truth=None,
            expected_schema=expected_schema,
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert 0.3 < result.score < 0.9  # Should be middle range
        assert "function_name_match" in result.metrics
        assert "schema_similarity" in result.metrics
        assert result.metrics["function_name_match"].score == 1.0
        assert 0.3 < result.metrics["schema_similarity"].score < 0.9
        assert result.metrics["schema_similarity"].reason is not None and \
            "Missing properties" in result.metrics["schema_similarity"].reason
        assert result.metrics["schema_similarity"].reason is not None and \
            "Extra properties" in result.metrics["schema_similarity"].reason
        # Dictionary access
        assert 0.3 < result['score'] < 0.9
        assert result['metrics']["function_name_match"]['score'] == 1.0
        assert 0.3 < result['metrics']["schema_similarity"]['score'] < 0.9
        assert result['metrics']["schema_similarity"]['reason'] is not None and \
            "Missing properties" in result['metrics']["schema_similarity"]['reason']
        assert result['metrics']["schema_similarity"]['reason'] is not None and \
            "Extra properties" in result['metrics']["schema_similarity"]['reason']

    def test_schema_jaccard_reward_wrong_function_name(self):
        """Test schema Jaccard reward with wrong function name."""
        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
        }

        function_call = {
            "name": "fetch_weather",  # Wrong name
            "arguments": json.dumps(
                {"location": "New York", "unit": "celsius"}
            ),
        }

        result = schema_jaccard_reward(
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "function_call": function_call},
            ],
            ground_truth=None,
            expected_schema=expected_schema,
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert result.score < 0.2  # Should be very low
        assert "function_name_match" in result.metrics
        assert result.metrics["function_name_match"].score == 0.0
        assert result.metrics["function_name_match"].reason is not None and \
            "Function name does not match" in result.metrics["function_name_match"].reason
        # Dictionary access
        assert result['score'] < 0.2
        assert result['metrics']["function_name_match"]['score'] == 0.0
        assert result['metrics']["function_name_match"]['reason'] is not None and \
            "Function name does not match" in result['metrics']["function_name_match"]['reason']

    def test_nested_schema(self):
        """Test schema Jaccard reward with nested objects."""
        expected_schema = {
            "name": "create_user",
            "arguments": {
                "user": {
                    "type": "object",
                    "properties": {
                        "firstName": {"type": "string"},
                        "lastName": {"type": "string"},
                        "age": {"type": "number"},
                    },
                }
            },
        }

        function_call = {
            "name": "create_user",
            "arguments": json.dumps(
                {"user": {"firstName": "John", "lastName": "Doe", "age": 30}}
            ),
        }

        result = schema_jaccard_reward(
            messages=[
                {"role": "user", "content": "Create a user for John Doe"},
                {"role": "assistant", "function_call": function_call},
            ],
            ground_truth=None,
            expected_schema=expected_schema,
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert result.score > 0.9  # Should be very high
        assert "schema_similarity" in result.metrics
        assert result.metrics["schema_similarity"].score > 0.9
        # Dictionary access
        assert result['score'] > 0.9
        assert "schema_similarity" in result['metrics']
        assert result['metrics']["schema_similarity"]['score'] > 0.9

    @patch("reward_kit.rewards.function_calling.OpenAI")
    def test_llm_judge_reward_mock(self, mock_openai):
        """Test LLM judge reward with a mocked OpenAI client."""
        # Mock the OpenAI client response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "SCORE: 0.85\nEXPLANATION: This is a good function call that matches the expected schema."
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
        }

        function_call = {
            "name": "get_weather",
            "arguments": json.dumps(
                {"location": "New York", "unit": "celsius"}
            ),
        }

        result = llm_judge_reward(
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "function_call": function_call},
            ],
            ground_truth=None,
            expected_schema=expected_schema,
            expected_behavior="Get the weather for the specified location with the specified unit",
            openai_api_key="fake_key_for_testing",
        )

        assert isinstance(result, EvaluateResult)
        assert mock_client.chat.completions.create.called
        # Attribute access
        assert result.score == 0.85
        assert "llm_judge" in result.metrics
        assert result.metrics["llm_judge"].score == 0.85
        assert result.metrics["llm_judge"].reason is not None and \
            "This is a good function call" in result.metrics["llm_judge"].reason
        # Dictionary access
        assert result['score'] == 0.85
        assert "llm_judge" in result['metrics']
        assert result['metrics']["llm_judge"]['score'] == 0.85
        assert result['metrics']["llm_judge"]['reason'] is not None and \
            "This is a good function call" in result['metrics']["llm_judge"]['reason']

    @patch("reward_kit.rewards.function_calling.OpenAI")
    def test_composite_function_call_reward(self, mock_openai):
        """Test composite function call reward."""
        # Mock the OpenAI client response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "SCORE: 0.70\nEXPLANATION: This is an acceptable function call but could be improved."
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
        }

        function_call = {
            "name": "get_weather",
            "arguments": json.dumps(
                {"location": "New York", "unit": "celsius"}
            ),
        }

        result = composite_function_call_reward(
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "function_call": function_call},
            ],
            ground_truth=None,
            expected_schema=expected_schema,
            expected_behavior="Get the weather for the specified location with the specified unit",
            openai_api_key="fake_key_for_testing",
            weights={"schema": 0.7, "llm": 0.3},
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert "schema_score" in result.metrics
        assert "llm_score" in result.metrics
        assert "weights" in result.metrics
        assert result.score > 0.8
        assert result.metrics["weights"].reason is not None and \
            "0.70" in result.metrics["weights"].reason  # Schema weight
        assert result.metrics["weights"].reason is not None and \
            "0.30" in result.metrics["weights"].reason  # LLM weight
        # Dictionary access
        assert "schema_score" in result['metrics']
        assert "llm_score" in result['metrics']
        assert "weights" in result['metrics']
        assert result['score'] > 0.8
        assert result['metrics']["weights"]['reason'] is not None and \
            "0.70" in result['metrics']["weights"]['reason']
        assert result['metrics']["weights"]['reason'] is not None and \
            "0.30" in result['metrics']["weights"]['reason']


# The JSON schema tests have been moved to tests/test_json_schema.py
