import pytest
from unittest.mock import MagicMock, patch
import json
from typing import List, Dict, Any, Optional

from reward_kit.models import EvaluateResult, MetricResult # Changed
from reward_kit.reward_function import RewardFunction


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_messages(self):
        """Test handling of empty messages arrays."""

        def reward_func(messages, original_messages, **kwargs):
            """Function that expects non-empty messages."""
            if not messages or not original_messages:
                raise ValueError("Messages cannot be empty")
            return EvaluateResult(score=0.5, reason="Test reason", metrics={}) # Changed

        reward_fn = RewardFunction(func=reward_func, mode="local")

        # Test with empty messages
        with pytest.raises(ValueError):
            reward_fn(
                messages=[],
                original_messages=[{"role": "user", "content": "Hello"}],
            )

        # Test with empty original_messages
        with pytest.raises(ValueError):
            reward_fn(
                messages=[{"role": "user", "content": "Hello"}],
                original_messages=[],
            )

    def test_invalid_message_structure(self):
        """Test handling of invalid message structure."""

        def reward_func(messages, original_messages, **kwargs):
            """Function that validates message structure."""
            for msg in messages + original_messages:
                if "role" not in msg or "content" not in msg:
                    raise ValueError("Invalid message structure")
            return EvaluateResult(score=0.5, reason="Test reason", metrics={}) # Changed

        reward_fn = RewardFunction(func=reward_func, mode="local")

        # Test with missing role
        with pytest.raises(ValueError):
            reward_fn(
                messages=[{"content": "Hello"}],
                original_messages=[{"role": "user", "content": "Hello"}],
            )

        # Test with missing content
        with pytest.raises(ValueError):
            reward_fn(
                messages=[{"role": "user"}],
                original_messages=[{"role": "user", "content": "Hello"}],
            )

    def test_remote_error_handling(self):
        """Test error handling in remote mode."""
        with patch("reward_kit.reward_function.requests.post") as mock_post:
            # Mock the response to simulate an error
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_post.return_value = mock_response

            reward_fn = RewardFunction(
                endpoint="https://example.com/reward", mode="remote"
            )

            # Should raise an exception due to server error
            with pytest.raises(Exception):
                reward_fn(
                    messages=[{"role": "user", "content": "Hello"}],
                    original_messages=[{"role": "user", "content": "Hello"}],
                )

    def test_large_message_handling(self):
        """Test handling of very large messages."""

        def reward_func(messages, original_messages, **kwargs):
            """Function that processes large messages."""
            # Just calculate based on message length
            content_length = len(messages[-1]["content"])
            length_score = min(content_length / 10000.0, 1.0)
            metrics = {
                "length": MetricResult( # Changed
                    score=length_score,
                    reason=f"Message length: {content_length} chars",
                    success=length_score == 1.0
                )
            }
            return EvaluateResult(score=0.5, reason="Large message processed", metrics=metrics) # Changed

        reward_fn = RewardFunction(func=reward_func, mode="local")

        # Generate a large message (10KB)
        large_content = "x" * 10000

        # This should not raise any exceptions
        result = reward_fn(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": large_content},
            ],
            original_messages=[{"role": "user", "content": "Hello"}],
        )

        assert result.score == 0.5
        assert "length" in result.metrics
        assert result.metrics["length"].score == 1.0  # Max score due to length

    def test_unicode_handling(self):
        """Test handling of unicode characters in messages."""

        def reward_func(messages, original_messages, **kwargs):
            """Function that handles Unicode."""
            # Just return a simple score and the message
            content = messages[-1]["content"]
            metrics = {
                "content": MetricResult( # Changed
                    score=0.5, reason=f"Processed: {content}", success=True # Assuming success for this metric
                )
            }
            return EvaluateResult(score=0.5, reason="Unicode message processed", metrics=metrics) # Changed

        reward_fn = RewardFunction(func=reward_func, mode="local")

        # Test with various Unicode characters
        unicode_message = "Hello 你好 こんにちは Привет 👋 🌍 ☀️"

        result = reward_fn(
            messages=[
                {"role": "user", "content": "Greet me in different languages"},
                {"role": "assistant", "content": unicode_message},
            ],
            original_messages=[
                {"role": "user", "content": "Greet me in different languages"}
            ],
        )

        assert result.score == 0.5
        assert "content" in result.metrics
        assert unicode_message in result.metrics["content"].reason

        # Ensure the output can be serialized to JSON and back
        json_str = json.dumps(result.model_dump()) # Changed to model_dump()
        parsed = json.loads(json_str)
        assert (
            parsed["metrics"]["content"]["reason"]
            == f"Processed: {unicode_message}"
        )
