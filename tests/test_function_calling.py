import pytest
from reward_kit.rewards.function_calling import match_function_call
from reward_kit.models import RewardOutput


class TestFunctionCalling:
    """Tests for the function_calling reward module."""

    def test_exact_match(self):
        """Test exact match of function name and arguments."""
        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "unit": {"type": "string"}
            }
        }
        
        parsed_name = "get_weather"
        parsed_args = {
            "location": "New York",
            "unit": "celsius"
        }
        
        result = match_function_call(
            messages=[{"role": "user", "content": "What's the weather?"}, 
                      {"role": "assistant", "content": "Let me check the weather."}],
            original_messages=[{"role": "user", "content": "What's the weather?"}],
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="exact"
        )
        
        assert result.score == 1.0
        assert "function_name_match" in result.metrics
        assert "arguments_match" in result.metrics
        assert result.metrics["function_name_match"].score == 1.0
        assert result.metrics["arguments_match"].score == 1.0

    def test_wrong_function_name(self):
        """Test with incorrect function name."""
        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "unit": {"type": "string"}
            }
        }
        
        parsed_name = "fetch_weather"  # Wrong name
        parsed_args = {
            "location": "New York",
            "unit": "celsius"
        }
        
        result = match_function_call(
            messages=[{"role": "user", "content": "What's the weather?"}, 
                      {"role": "assistant", "content": "Let me check the weather."}],
            original_messages=[{"role": "user", "content": "What's the weather?"}],
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="exact"
        )
        
        assert result.score < 1.0
        assert "function_name_match" in result.metrics
        assert result.metrics["function_name_match"].score == 0.0
        assert "Function name does not match" in result.metrics["function_name_match"].reason

    def test_missing_required_argument(self):
        """Test with missing required argument."""
        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "unit": {"type": "string"}
            }
        }
        
        parsed_name = "get_weather"
        parsed_args = {
            "location": "New York"
            # Missing "unit" argument
        }
        
        result = match_function_call(
            messages=[{"role": "user", "content": "What's the weather?"}, 
                      {"role": "assistant", "content": "Let me check the weather."}],
            original_messages=[{"role": "user", "content": "What's the weather?"}],
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="exact"
        )
        
        assert result.score < 1.0
        assert "arguments_match" in result.metrics
        assert result.metrics["arguments_match"].score < 1.0
        assert "Missing argument" in result.metrics["arguments_match"].reason

    def test_extra_argument(self):
        """Test with extra argument not in schema."""
        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "unit": {"type": "string"}
            }
        }
        
        parsed_name = "get_weather"
        parsed_args = {
            "location": "New York",
            "unit": "celsius",
            "extra_param": "value"  # Extra argument
        }
        
        result = match_function_call(
            messages=[{"role": "user", "content": "What's the weather?"}, 
                      {"role": "assistant", "content": "Let me check the weather."}],
            original_messages=[{"role": "user", "content": "What's the weather?"}],
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="exact"
        )
        
        assert result.score < 1.0
        assert "arguments_match" in result.metrics
        assert result.metrics["arguments_match"].score < 1.0
        assert "Unexpected argument" in result.metrics["arguments_match"].reason

    def test_permissive_mode(self):
        """Test permissive mode with extra arguments."""
        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "unit": {"type": "string"}
            }
        }
        
        parsed_name = "get_weather"
        parsed_args = {
            "location": "New York",
            "unit": "celsius",
            "extra_param": "value"  # Extra argument
        }
        
        result = match_function_call(
            messages=[{"role": "user", "content": "What's the weather?"}, 
                      {"role": "assistant", "content": "Let me check the weather."}],
            original_messages=[{"role": "user", "content": "What's the weather?"}],
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="permissive"  # Permissive mode
        )
        
        # In permissive mode, extra arguments are allowed
        assert result.score == 1.0
        assert "function_name_match" in result.metrics
        assert "arguments_match" in result.metrics
        assert result.metrics["function_name_match"].score == 1.0
        assert result.metrics["arguments_match"].score == 1.0

    def test_wrong_argument_value_type(self):
        """Test with wrong argument value type."""
        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "temperature": {"type": "number"}
            }
        }
        
        parsed_name = "get_weather"
        parsed_args = {
            "location": "New York",
            "temperature": "25"  # String instead of number
        }
        
        result = match_function_call(
            messages=[{"role": "user", "content": "What's the weather?"}, 
                      {"role": "assistant", "content": "Let me check the weather."}],
            original_messages=[{"role": "user", "content": "What's the weather?"}],
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="exact"
        )
        
        assert result.score < 1.0
        assert "arguments_match" in result.metrics
        assert result.metrics["arguments_match"].score < 1.0
        assert "Type mismatch" in result.metrics["arguments_match"].reason