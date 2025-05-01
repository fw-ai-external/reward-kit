"""
Tests for OpenAI message type compatibility.
"""

import sys
import os
import unittest
from typing import List, Dict, Any

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reward_kit import Message, reward_function, EvaluateResult, MetricResult
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam, ChatCompletionAssistantMessageParam


class OpenAICompatibilityTest(unittest.TestCase):
    """Test compatibility with OpenAI message types."""

    def test_message_type_compatibility(self):
        """Test that our Message type is compatible with OpenAI's."""
        # Check that Message is an alias for ChatCompletionMessageParam
        self.assertEqual(Message, ChatCompletionMessageParam)

    def test_openai_message_in_decorator(self):
        """Test that the reward_function decorator can handle OpenAI message types."""
        
        @reward_function
        def sample_evaluator(messages: List[Message], **kwargs: Any) -> EvaluateResult:
            """Sample evaluator that uses OpenAI message types."""
            # Check if the last message is from the assistant
            last_message = messages[-1]
            
            # Simple evaluation - check if the response is not empty and from the assistant
            success = (last_message.get("role") == "assistant" and 
                      last_message.get("content", "") != "")
            
            return EvaluateResult(root={
                "not_empty": MetricResult(
                    success=success,
                    score=1.0 if success else 0.0,
                    reason="Response is not empty" if success else "Response is empty or not from assistant"
                )
            })
        
        # Test with OpenAI message types
        system_message = ChatCompletionSystemMessageParam(role="system", content="You are a helpful assistant.")
        user_message = ChatCompletionUserMessageParam(role="user", content="Hello!")
        assistant_message = ChatCompletionAssistantMessageParam(role="assistant", content="Hi there!")
        
        # Convert to dict for the decorated function
        messages_dict = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        # Test the evaluator with dict messages
        result = sample_evaluator(messages=messages_dict)
        
        # Verify the result
        self.assertIn("not_empty", result)
        self.assertEqual(result["not_empty"]["score"], 1.0)
        self.assertEqual(result["not_empty"]["reason"], "Response is not empty")


if __name__ == "__main__":
    unittest.main()