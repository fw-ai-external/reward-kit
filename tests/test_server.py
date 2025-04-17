import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from typing import List, Dict, Any, Optional

from reward_kit.server import create_app
from reward_kit.models import RewardOutput, MetricRewardOutput


@pytest.fixture
def test_reward_func():
    """Fixture that returns a test reward function."""
    def _reward_func(
        messages: List[Dict[str, str]], 
        original_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> RewardOutput:
        """Test reward function that returns a simple score."""
        metrics = {
            "test": MetricRewardOutput(
                score=0.5,
                reason="Test reason"
            )
        }
        return RewardOutput(score=0.5, metrics=metrics)
    
    return _reward_func


class TestServer:
    """Tests for the FastAPI server."""
    
    @pytest.fixture
    def client(self, test_reward_func):
        """Create a test client for the FastAPI app."""
        app = create_app(test_reward_func)
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    def test_reward_endpoint(self, client):
        """Test the reward endpoint."""
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ],
            "original_messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        
        response = client.post("/reward", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["score"] == 0.5
        assert "metrics" in data
        assert "test" in data["metrics"]
        assert data["metrics"]["test"]["score"] == 0.5
        assert data["metrics"]["test"]["reason"] == "Test reason"
    
    def test_reward_endpoint_with_metadata(self, client):
        """Test the reward endpoint with metadata."""
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ],
            "original_messages": [
                {"role": "user", "content": "Hello"}
            ],
            "metadata": {
                "test_key": "test_value"
            }
        }
        
        response = client.post("/reward", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["score"] == 0.5
    
    def test_reward_endpoint_missing_required_fields(self, client):
        """Test the reward endpoint with missing required fields."""
        # Empty payload without messages field
        payload = {}
        
        response = client.post("/reward", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_reward_endpoint_malformed_messages(self, client):
        """Test the reward endpoint with malformed messages."""
        # Malformed messages - missing role
        payload = {
            "messages": [
                {"content": "Hello"},  # Missing role
                {"role": "assistant", "content": "Hi there"}
            ],
            "original_messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        
        response = client.post("/reward", json=payload)
        assert response.status_code == 422  # Validation error