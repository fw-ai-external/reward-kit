import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from typing import List, Dict, Any, Optional

from reward_kit.models import EvaluateResult, MetricResult # Changed
from reward_kit.reward_function import RewardFunction, reward_function
from reward_kit.server import create_app


@reward_function
def sample_reward_function(
    messages: List[Dict[str, str]],
    original_messages: List[Dict[str, str]],
    **kwargs,
) -> EvaluateResult: # Changed
    """Sample reward function that checks message length and keywords."""
    # The @reward_function decorator from typed_interface.py ensures 'messages'
    # are List[Message] Pydantic objects when this function is called.
    last_message_obj = messages[-1]
    last_message_content = last_message_obj.content if last_message_obj.content is not None else ""
    metrics = {}

    # Check message length
    length_score = min(len(last_message_content) / 100.0, 1.0)
    metrics["length"] = MetricResult( # Changed
        score=length_score, reason=f"Message length: {len(last_message_content)} chars", success=length_score == 1.0
    )

    # Check for keywords
    has_helpful_keywords = any(
        keyword in last_message_content.lower()
        for keyword in ["help", "assist", "support"]
    )
    keyword_score = 0.8 if has_helpful_keywords else 0.2
    metrics["keywords"] = MetricResult( # Changed
        score=keyword_score,
        reason=(
            "Contains helpful keywords"
            if has_helpful_keywords
            else "Missing helpful keywords"
        ),
        success=has_helpful_keywords
    )

    # Calculate final score (70% keywords, 30% length)
    final_score = 0.7 * keyword_score + 0.3 * length_score
    final_reason = "Overall score based on keyword presence and message length."

    return EvaluateResult(score=final_score, reason=final_reason, metrics=metrics) # Changed


class TestIntegration:
    """Integration tests for reward-kit components."""

    def test_reward_function_to_server(self):
        """Test flow from reward function to server."""
        # Create a FastAPI app with our reward function
        app = create_app(sample_reward_function)
        client = TestClient(app)

        # Test request with helpful keywords
        helpful_payload = {
            "messages": [
                {"role": "user", "content": "Can you help me with Python?"},
                {
                    "role": "assistant",
                    "content": "I'd be happy to assist you with Python. What do you need help with?",
                },
            ],
            "original_messages": [
                {"role": "user", "content": "Can you help me with Python?"}
            ],
        }

        helpful_response = client.post("/reward", json=helpful_payload)
        assert helpful_response.status_code == 200
        helpful_data = helpful_response.json()

        # Score should be high due to helpful keywords
        assert helpful_data["score"] > 0.7
        assert "keywords" in helpful_data["metrics"]
        assert helpful_data["metrics"]["keywords"]["score"] == 0.8

        # Test request without helpful keywords
        unhelpful_payload = {
            "messages": [
                {"role": "user", "content": "Can you help me with Python?"},
                {
                    "role": "assistant",
                    "content": "Python is a programming language.",
                },
            ],
            "original_messages": [
                {"role": "user", "content": "Can you help me with Python?"}
            ],
        }

        unhelpful_response = client.post("/reward", json=unhelpful_payload)
        assert unhelpful_response.status_code == 200
        unhelpful_data = unhelpful_response.json()

        # Score should be lower due to missing helpful keywords
        assert unhelpful_data["score"] < 0.5
        assert "keywords" in unhelpful_data["metrics"]
        assert unhelpful_data["metrics"]["keywords"]["score"] == 0.2

    def test_reward_function_to_trl_adapter(self):
        """Test converting a reward function to TRL adapter."""
        # Create a RewardFunction instance
        reward_fn = RewardFunction(func=sample_reward_function, mode="local")

        # Get the TRL adapter
        trl_adapter = reward_fn.get_trl_adapter()

        # Test with batch inputs (as expected by TRL)
        batch_messages = [
            [
                {"role": "user", "content": "Can you help me with Python?"},
                {
                    "role": "assistant",
                    "content": "I'd be happy to assist you with Python programming!",
                },
            ],
            [
                {"role": "user", "content": "Tell me about Python."},
                {
                    "role": "assistant",
                    "content": "Python is a programming language.",
                },
            ],
        ]

        batch_original_messages = [
            [{"role": "user", "content": "Can you help me with Python?"}],
            [{"role": "user", "content": "Tell me about Python."}],
        ]

        # Call the TRL adapter
        scores = trl_adapter(batch_messages, batch_original_messages)

        # Verify the results
        assert isinstance(scores, list)
        assert len(scores) == 2
        assert (
            scores[0] > scores[1]
        )  # First message should score higher (has "assist")
