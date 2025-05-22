import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Load the deploy_example module directly from the examples folder
def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ImportError(f"Could not load spec for module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Spec for module {name} has no loader")
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

    # Import the updated reward function
    from examples.informativeness_metric.main import (
        evaluate as informativeness_evaluate_function,
    )
    from reward_kit import Message  # For converting dict messages to Message objects

    # Convert test messages to Message objects
    test_messages_objects = [Message(**msg) for msg in test_messages]

    # Test the reward function
    # The informativeness_evaluate_function expects List[Message] and **kwargs.
    # It does not explicitly take original_messages.
    result = informativeness_evaluate_function(messages=test_messages_objects)

    # Verify results (EvaluateResult uses attribute access)
    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0
    assert "length" in result.metrics
    assert "specificity" in result.metrics
    assert "content_density" in result.metrics


def test_deploy_to_fireworks(
    deploy_example, mock_env_variables, mock_requests_post, mock_requests_get
):
    """Test the deployment function using the refactored deploy_example"""
    # mock_env_variables fixture sets FIREWORKS_API_KEY and FIREWORKS_ACCOUNT_ID
    # mock_requests_post fixture mocks requests.post to simulate successful API call
    # mock_requests_get fixture is also available if needed (e.g. for checking existence before PUT/POST)

    # Call the actual deploy_to_fireworks function from the loaded example module
    # This will use create_evaluation, which internally uses the new auth and requests.post
    evaluation_id = deploy_example.deploy_to_fireworks()

    # Assert based on the mocked response in mock_requests_post and deploy_example logic
    # The mock_requests_post returns:
    # {
    #     "name": "accounts/test_account/evaluators/informativeness-v1",
    #     "displayName": "informativeness-v1", ...
    # }
    # The refactored deploy_example.py now deploys with ID "informativeness-metric-example-v1"
    # and the mock should reflect what the actual API call would do for that ID.
    # The mock_requests_post currently returns a name based on "informativeness-v1".
    # Let's update the assertion to match the new ID used in deploy_example.py.
    # The deploy_example.py also extracts the ID from the response name.
    # If mock_requests_post.return_value.json.return_value["name"] is
    # "accounts/test_account/evaluators/informativeness-metric-example-v1",
    # then deploy_example.py should extract "informativeness-metric-example-v1".

    # Adjust mock to return the new ID in the 'name' field of the response
    mock_requests_post.return_value.json.return_value = {
        "name": "accounts/test_account/evaluators/informativeness-metric-example-v1",
        "displayName": "Informativeness Metric (Example V1)",
        "description": "Evaluates response informativeness based on specificity and content density.",
        # ... other fields if necessary
    }

    evaluation_id = deploy_example.deploy_to_fireworks()
    assert evaluation_id == "informativeness-metric-example-v1"

    # Verify that requests.post was called (by create_evaluation)
    mock_requests_post.assert_called()


def test_deploy_failure_handling(deploy_example, mock_env_variables, monkeypatch):
    """Test error handling in the deploy_to_fireworks function when API call fails"""

    # Configure mock_requests_post to simulate an API error
    mock_response_one = MagicMock()  # Renamed to avoid conflict
    mock_response_one.status_code = 401  # Unauthorized
    mock_response_one.text = '{"error": "unauthorized"}'
    mock_response_one.json.return_value = {"error": "unauthorized"}
    # Import requests for requests.exceptions.HTTPError
    import requests

    mock_response_one.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "API Error", response=mock_response_one
    )

    with patch("requests.post", return_value=mock_response_one) as mock_post_failure:
        # Call the actual deploy_to_fireworks function
        result = deploy_example.deploy_to_fireworks()

        # The example's deploy_to_fireworks catches exceptions and prints, then returns None
        assert result is None
        mock_post_failure.assert_called()  # Ensure the API call was attempted

    # Test another failure case, e.g. non-JSON response or other requests exception
    with patch(
        "requests.post",
        side_effect=requests.exceptions.ConnectionError("Connection failed"),
    ) as mock_post_conn_error:
        result = deploy_example.deploy_to_fireworks()
        assert result is None
        mock_post_conn_error.assert_called()

    # Create a mock ValueError with the expected error message (this part of original test seems less relevant now)
    # error_message = """
    # Permission Error: Your API key doesn't have deployment permissions.
    # ...
    # """
    # The refactored deploy_example.py's error handling is more generic now,
    # catching Exception and printing str(e).

    # If we want to test specific error messages printed to console, we'd need to mock 'print'
    # For now, verifying it returns None on error is sufficient for this test's scope.
