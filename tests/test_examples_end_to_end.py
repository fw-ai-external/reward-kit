import importlib.util
import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest


# Helper function to import modules from file paths
def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def examples_path():
    """Return the path to the examples directory"""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")


@pytest.fixture
def mock_env_variables(monkeypatch):
    """Set environment variables for testing"""
    monkeypatch.setenv("FIREWORKS_API_KEY", "test_api_key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "test_account")
    monkeypatch.setenv("FIREWORKS_API_BASE", "https://api.fireworks.ai")


@pytest.fixture
def mock_requests():
    """Mock all requests methods with appropriate responses"""
    with patch("requests.post") as mock_post, patch("requests.get") as mock_get, patch(
        "requests.delete"
    ) as mock_delete:

        # Configure mock_post for different use cases
        def post_side_effect(*args, **kwargs):
            url = args[0]
            mock_resp = MagicMock()

            # For preview API
            if "previewEvaluator" in url:
                mock_resp.status_code = 200
                mock_resp.json.return_value = {
                    "totalSamples": 2,
                    "totalRuntimeMs": 1234,
                    "results": [
                        {
                            "success": True,
                            "score": 0.26,
                            "perMetricEvals": {
                                "word_count": {
                                    "score": 0.26,
                                    "reason": "Word count: 26",
                                }
                            },
                        },
                        {
                            "success": True,
                            "score": 0.22,
                            "perMetricEvals": {
                                "word_count": {
                                    "score": 0.22,
                                    "reason": "Word count: 22",
                                }
                            },
                        },
                    ],
                }
            # For evaluator creation
            elif "/evaluators" in url:
                mock_resp.status_code = 200
                mock_resp.json.return_value = {
                    "name": "accounts/test_account/evaluators/test-eval",
                    "displayName": "Test Evaluator",
                    "description": "Test description",
                }
            # For reward function deployment
            else:
                mock_resp.status_code = 200
                mock_resp.json.return_value = {
                    "name": "accounts/test_account/evaluators/informativeness-v1",
                    "displayName": "informativeness-v1",
                    "description": "Informativeness Evaluator",
                }

            return mock_resp

        mock_post.side_effect = post_side_effect

        # Configure mock_get
        mock_get.return_value.status_code = 404  # Evaluator doesn't exist by default

        # Configure mock_delete
        mock_delete.return_value.status_code = 200

        yield (mock_post, mock_get, mock_delete)


@pytest.fixture
def temp_examples_dir(examples_path):
    """Create a temporary directory with copies of the examples"""
    temp_dir = tempfile.mkdtemp()

    # Copy all example files to the temp directory
    for item in os.listdir(examples_path):
        src = os.path.join(examples_path, item)
        dst = os.path.join(temp_dir, item)

        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    yield temp_dir

    # Clean up
    shutil.rmtree(temp_dir)


def test_basic_reward_example(mock_env_variables):
    """Test the ability to import and analyze the basic_reward.py example"""
    basic_reward_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "examples",
        "basic_reward.py",
    )

    # Load the module without executing functions
    module = load_module_from_path("basic_reward", basic_reward_path)

    # Verify the module has the expected components
    assert hasattr(module, "calculate_base_score")
    assert hasattr(module, "calculate_safety_score")
    assert hasattr(module, "combined_reward")


def test_folder_based_evaluation_example(mock_env_variables, mock_requests):
    """Test the folder_based_evaluation_example.py can be loaded"""
    # Import the module directly
    folder_eval_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "examples",
        "folder_based_evaluation_example.py",
    )

    # Load the module without executing __main__
    module = load_module_from_path("folder_eval", folder_eval_path)

    # Verify the module has the expected components
    assert hasattr(module, "setup_sample_evaluator")
    assert hasattr(module, "clean_up")
    assert hasattr(module, "main")


def test_server_example_endpoint(mock_env_variables):
    """Test that the reward server example can be loaded"""
    server_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "examples",
        "server_example.py",
    )

    # We'll patch the serve function to prevent it from actually running
    with patch("reward_kit.server.serve"):
        # Load the module
        module = load_module_from_path("server_example", server_path)

        # Verify the module has the expected components
        assert hasattr(module, "server_reward")


class TestExamplesIntegration:
    """Integration tests that run examples together in different combinations"""

    def test_reward_then_deploy(self, mock_env_variables, mock_requests):
        """Test loading reward modules and deployment modules"""
        # Load the basic reward example
        basic_reward_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "examples",
            "basic_reward.py",
        )
        basic_module = load_module_from_path("basic_reward", basic_reward_path)

        # Check that modules exist
        assert hasattr(basic_module, "combined_reward")

        # Load the deployment example
        deploy_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "examples",
            "deploy_example.py",
        )

        # Load the deploy module
        deploy_module = load_module_from_path("deploy_example", deploy_path)

        # Verify the deploy module loaded correctly
        assert hasattr(deploy_module, "informativeness_reward")
        assert hasattr(deploy_module, "deploy_to_fireworks")
