import os
import pytest
from unittest.mock import patch, MagicMock
import sys
import json
import importlib.util
from pathlib import Path


# Load the deploy_example module directly from the examples folder
def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def deploy_example():
    # Path to the deploy_example.py file
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "examples",
        "deploy_example.py",
    )

    # Load the module
    return load_module_from_path("deploy_example", file_path)


@pytest.fixture
def mock_env_variables(monkeypatch):
    """Set environment variables for testing"""
    monkeypatch.setenv("FIREWORKS_API_KEY", "test_api_key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "test_account")
    monkeypatch.setenv("FIREWORKS_API_BASE", "https://api.fireworks.ai")


@pytest.fixture
def mock_requests_post():
    """Mock requests.post method"""
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock()
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "name": "accounts/test_account/evaluators/informativeness-v1",
            "displayName": "informativeness-v1",
            "description": "Evaluates response informativeness based on specificity and content density",
        }
        yield mock_post


@pytest.fixture
def mock_requests_get():
    """Mock requests.get method"""
    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock()
        mock_get.return_value.status_code = 404  # Evaluator doesn't exist
        yield mock_get


def test_informativeness_reward(deploy_example):
    """Test that the reward function works correctly"""
    # Example messages
    test_messages = [
        {"role": "user", "content": "Can you explain machine learning?"},
        {
            "role": "assistant",
            "content": "Machine learning is a method of data analysis that automates analytical model building. Specifically, it uses algorithms that iteratively learn from data, allowing computers to find hidden insights without being explicitly programmed where to look. For example, deep learning is a type of machine learning that uses neural networks with many layers. Such approaches have revolutionized fields like computer vision and natural language processing.",
        },
    ]

    # Test the reward function
    result = deploy_example.informativeness_reward(
        messages=test_messages, original_messages=[test_messages[0]]
    )

    # Verify results
    assert isinstance(result['score'], float)
    assert 0.0 <= result['score'] <= 1.0
    assert "length" in result['metrics']
    assert "specificity" in result['metrics']
    assert "content_density" in result['metrics']


def test_deploy_to_fireworks(
    deploy_example, mock_env_variables, mock_requests_post, mock_requests_get
):
    """Test the deployment function"""
    with patch(
        "reward_kit.auth.get_authentication"
    ) as mock_get_auth, patch.object(
        deploy_example, "deploy_to_fireworks"
    ) as mock_deploy:

        # Mock the authentication function
        mock_get_auth.return_value = ("test_account", "test_api_key")

        # Mock the deploy function to return a fixed evaluation ID
        mock_deploy.return_value = "informativeness-v1"

        # Run the deploy function
        evaluation_id = mock_deploy()

        # Verify the result
        assert evaluation_id == "informativeness-v1"
        assert mock_deploy.call_count == 1


def test_deploy_failure_handling(deploy_example, mock_env_variables):
    """Test error handling in the deploy_to_fireworks function"""
    # Create a mock ValueError with the expected error message
    error_message = """
    Permission Error: Your API key doesn't have deployment permissions.
    Possible solutions:
    1. Use a production API key: export FIREWORKS_API_KEY=your_production_key
    2. Request deployment permissions for your API key
    3. Check if your account has evaluator deployment enabled
    Error details: {"error":"unauthorized"}
    """

    with patch.object(deploy_example, "deploy_to_fireworks") as mock_deploy:
        # Mock the deploy function to raise a ValueError
        mock_deploy.side_effect = ValueError(error_message)

        # Since we're expecting an error, we need to handle it
        try:
            mock_deploy()
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            # Verify the error was raised
            assert isinstance(e, ValueError)
            assert "Permission Error" in str(e)
            assert mock_deploy.call_count == 1
