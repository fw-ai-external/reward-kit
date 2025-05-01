"""
Tests for the typed interface functionality.
"""

import pytest
from typing import List, Dict, Any

from reward_kit.models import Message, EvaluateResult, MetricResult
from reward_kit.typed_interface import reward_function


def test_typed_interface_basic():
    """Test that the typed_interface decorator works with basic inputs."""
    
    @reward_function
    def sample_evaluator(messages: List[Message], **kwargs) -> EvaluateResult:
        """Sample evaluator that returns a hardcoded result."""
        return EvaluateResult({
            "test": MetricResult(success=True, score=0.8, reason="Test reason")
        })
    
    # Test with valid messages
    valid_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    result = sample_evaluator(messages=valid_messages)
    
    # Check the output
    assert isinstance(result, dict)
    assert "test" in result
    assert result["test"]["success"] is True
    assert result["test"]["score"] == 0.8
    assert result["test"]["reason"] == "Test reason"


def test_typed_interface_input_validation():
    """
    Test that invalid input messages are properly handled.
    
    Note: With our OpenAI compatibility update, we're now more permissive with message formats
    to match OpenAI's flexibility. This test now checks that even unusual formats are processed
    without errors.
    """
    
    @reward_function
    def sample_evaluator(messages: List[Message], **kwargs) -> EvaluateResult:
        # Check that we can access the messages
        assert len(messages) > 0
        
        return EvaluateResult(root={
            "test": MetricResult(success=True, score=0.5, reason="Test")
        })
    
    # Test with unusual but now acceptable messages
    unusual_messages = [
        {"content": "Hello"},  # Missing 'role' but we handle this gracefully now
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    # This should now work without raising an error
    result = sample_evaluator(messages=unusual_messages)
    assert result["test"]["score"] == 0.5
    
    # Test with other unusual formats
    unusual_role_messages = [
        {"role": "custom_role", "content": "Hello"},  # Non-standard role
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    # This should also work without raising an error
    result = sample_evaluator(messages=unusual_role_messages)
    assert result["test"]["score"] == 0.5


def test_typed_interface_output_validation():
    """Test that the typed_interface validates output correctly."""
    
    @reward_function
    def invalid_evaluator(messages: List[Message], **kwargs) -> EvaluateResult:
        # Return an incomplete metric (missing required fields)
        # This should be caught by the output validation
        return EvaluateResult({
            "test": {"score": 0.5}  # Missing 'success' and 'reason'
        })  # type: ignore
    
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    with pytest.raises(ValueError) as excinfo:
        invalid_evaluator(messages=messages)
    
    # For pydantic v2, the error format has changed but should contain 'validation errors'
    assert "validation error" in str(excinfo.value).lower()


def test_typed_interface_kwargs():
    """Test that the typed_interface correctly passes through kwargs."""
    
    @reward_function
    def kwargs_evaluator(messages: List[Message], **kwargs) -> EvaluateResult:
        # Return the kwargs in the reason field
        return EvaluateResult({
            "test": MetricResult(
                success=True, 
                score=0.5, 
                reason=f"Got kwargs: {sorted([k for k in kwargs.keys()])}"
            )
        })
    
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    result = kwargs_evaluator(messages=messages, param1="test", param2=123)
    
    assert "test" in result
    assert "Got kwargs: ['param1', 'param2']" == result["test"]["reason"]