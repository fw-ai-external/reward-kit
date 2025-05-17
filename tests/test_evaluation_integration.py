import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reward_kit.evaluation import Evaluator, create_evaluation, preview_evaluation


def create_test_folder():
    """Create a temporary folder with a main.py file for testing"""
    tmp_dir = tempfile.mkdtemp()

    # Create main.py
    with open(os.path.join(tmp_dir, "main.py"), "w") as f:
        f.write(
            """
def evaluate(messages, original_messages=None, tools=None, **kwargs):
    if not messages:
        return {'score': 0.0, 'reason': 'No messages found'}

    last_message = messages[-1]
    content = last_message.get('content', '')

    word_count = len(content.split())
    score = min(word_count / 100, 1.0)

    return {
        'score': score,
        'reason': f'Word count: {word_count}'
    }
"""
        )

    return tmp_dir


def create_sample_file():
    """Create a temporary sample file for testing"""
    fd, path = tempfile.mkstemp(suffix=".jsonl")

    samples = [
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": "Hi there! How can I help you today?",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is AI?"},
                {
                    "role": "assistant",
                    "content": "AI stands for Artificial Intelligence.",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search for information",
                    },
                }
            ],
        },
    ]

    with os.fdopen(fd, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return path


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
        # Default response for creation API
        default_response = {
            "name": "accounts/test_account/evaluators/test-eval",
            "displayName": "Test Evaluator",
            "description": "Test description",
            "multiMetrics": False,
        }

        # For preview API, we need to include results
        preview_response = {
            "totalSamples": 2,
            "totalRuntimeMs": 1234,
            "results": [
                {
                    "success": True,
                    "score": 0.7,
                    "perMetricEvals": {
                        "quality": 0.8,
                        "relevance": 0.7,
                        "safety": 0.9,
                    },
                },
                {
                    "success": True,
                    "score": 0.5,
                    "perMetricEvals": {
                        "quality": 0.6,
                        "relevance": 0.4,
                        "safety": 0.8,
                    },
                },
            ],
        }

        # Configure mock to return different responses based on URL
        def side_effect(*args, **kwargs):
            url = args[0]
            response = mock_post.return_value

            if "previewEvaluator" in url:
                response.json.return_value = preview_response
            else:
                response.json.return_value = default_response

            return response

        mock_post.side_effect = side_effect
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = default_response

        yield mock_post


def test_integration_single_metric(mock_env_variables, mock_requests_post):
    """Test the integration path for a single metric evaluator"""
    tmp_dir = create_test_folder()
    sample_file = create_sample_file()

    try:
        # Preview the evaluation
        preview_result = preview_evaluation(
            metric_folders=[f"test_metric={tmp_dir}"],
            sample_file=sample_file,
            max_samples=2,
        )

        assert preview_result.total_samples == 2
        assert len(preview_result.results) == 2

        # Create the evaluation
        evaluator = create_evaluation(
            evaluator_id="test-eval",
            metric_folders=[f"test_metric={tmp_dir}"],
            display_name="Test Evaluator",
            description="Test description",
        )

        assert evaluator["name"] == "accounts/test_account/evaluators/test-eval"
        assert evaluator["displayName"] == "Test Evaluator"

        # Verify API call
        assert mock_requests_post.call_count >= 1

        # Get the last call (for creation)
        args, kwargs = mock_requests_post.call_args_list[-1]
        url = args[0]
        payload = kwargs.get("json")

        assert "api.fireworks.ai/v1/accounts/test_account/evaluators" in url

        # Handle different payload formats in different environments
        # The dev API uses "evaluator" while the old API uses "evaluation"
        if "evaluator" in payload:
            assert "evaluator" in payload
            assert "evaluatorId" in payload
            assert payload["evaluatorId"] == "test-eval"
            assert "criteria" in payload["evaluator"]

            # Check criteria format
            criteria = payload["evaluator"]["criteria"]
            assert len(criteria) > 0
            assert "type" in criteria[0]
            assert criteria[0]["type"] == "CODE_SNIPPETS"
            assert "codeSnippets" in criteria[0]
        else:
            assert "evaluation" in payload
            assert payload["evaluation"]["evaluationType"] == "code_assertion"
            assert payload["evaluationId"] == "test-eval"
            assert "assertions" in payload["evaluation"]

            # Check assertion format
            assertions = payload["evaluation"]["assertions"]
            assert len(assertions) > 0
            assert "assertionType" in assertions[0]
            assert assertions[0]["assertionType"] == "CODE"
            assert "codeAssertion" in assertions[0]
            assert "metricName" in assertions[0]

    finally:
        # Clean up
        os.unlink(os.path.join(tmp_dir, "main.py"))
        os.rmdir(tmp_dir)
        os.unlink(sample_file)


def test_integration_multi_metrics(mock_env_variables, mock_requests_post):
    """Test the integration path for a multi-metrics evaluator"""
    tmp_dir = create_test_folder()
    sample_file = create_sample_file()

    try:
        # Preview the evaluation
        preview_result = preview_evaluation(
            multi_metrics=True,
            folder=tmp_dir,
            sample_file=sample_file,
            max_samples=2,
        )

        assert preview_result.total_samples == 2
        assert len(preview_result.results) == 2

        # Check that we get expected metrics in multi-metrics mode
        assert "quality" in preview_result.results[0]["per_metric_evals"]
        assert "relevance" in preview_result.results[0]["per_metric_evals"]
        assert "safety" in preview_result.results[0]["per_metric_evals"]

        # Create the evaluation
        mock_requests_post.reset_mock()
        default_response = {
            "name": "accounts/test_account/evaluators/test-eval",
            "displayName": "Multi Metrics Evaluator",
            "description": "Test multi-metrics evaluator",
            "multiMetrics": True,
        }
        mock_requests_post.return_value.json.return_value = default_response

        evaluator = create_evaluation(
            evaluator_id="multi-metrics-eval",
            multi_metrics=True,
            folder=tmp_dir,
            display_name="Multi Metrics Evaluator",
            description="Test multi-metrics evaluator",
        )

        assert evaluator["name"] == "accounts/test_account/evaluators/test-eval"

        # Verify API call
        assert mock_requests_post.call_count >= 1

        # Get the last call (for creation)
        args, kwargs = mock_requests_post.call_args_list[-1]
        url = args[0]
        payload = kwargs.get("json")

        assert "api.fireworks.ai/v1/accounts/test_account/evaluators" in url

        # Handle different payload formats in different environments
        if "evaluator" in payload:
            assert "evaluator" in payload
            assert "evaluatorId" in payload
            assert payload["evaluatorId"] == "multi-metrics-eval"
            assert "criteria" in payload["evaluator"]
            assert payload["evaluator"]["multiMetrics"] == True

            # Check criteria format
            criteria = payload["evaluator"]["criteria"]
            assert len(criteria) > 0
        else:
            assert "evaluationId" in payload
            assert payload["evaluationId"] == "multi-metrics-eval"
            assert "evaluation" in payload
            assert "assertions" in payload["evaluation"]

            # Check assertion format for production API - not dev
            assertions = payload["evaluation"]["assertions"]
            assert len(assertions) > 0
            assert "assertionType" in assertions[0]
            assert assertions[0]["assertionType"] == "CODE"
            assert "codeAssertion" in assertions[0]
            assert "metricName" in assertions[0]

    finally:
        # Clean up
        os.unlink(os.path.join(tmp_dir, "main.py"))
        os.rmdir(tmp_dir)
        os.unlink(sample_file)


@patch("sys.exit")
def test_integration_cli_commands(mock_exit, mock_env_variables, mock_requests_post):
    """Test CLI integration by directly calling the CLI command functions"""
    from reward_kit.cli import deploy_command, preview_command

    # Make sys.exit a pass-through instead of raising an exception
    mock_exit.return_value = None
    mock_exit.side_effect = lambda code=0: None

    tmp_dir = create_test_folder()
    sample_file = create_sample_file()

    try:
        # Test preview command
        with patch(
            "reward_kit.cli_commands.preview.preview_evaluation"
        ) as mock_preview:  # Corrected patch target
            # Create mock preview result
            mock_preview_result = MagicMock()
            mock_preview_result.display = MagicMock()
            mock_preview.return_value = mock_preview_result

            # Create args
            args = MagicMock()
            args.metrics_folders = [f"test_metric={tmp_dir}"]
            args.samples = sample_file
            args.max_samples = 2
            # Add HuggingFace attributes with None values
            args.huggingface_dataset = None
            args.huggingface_split = "train"
            args.huggingface_prompt_key = "prompt"
            args.huggingface_response_key = "response"
            args.huggingface_key_map = None

            # Run preview command
            with patch(
                "reward_kit.cli_commands.preview.Path.exists", return_value=True
            ):  # Corrected patch target for Path
                result = preview_command(args)

                # Verify the result
                assert result == 0
                mock_preview.assert_called_once_with(
                    metric_folders=[f"test_metric={tmp_dir}"],
                    sample_file=sample_file,
                    max_samples=2,
                    huggingface_dataset=None,
                    huggingface_split="train",
                    huggingface_prompt_key="prompt",
                    huggingface_response_key="response",
                    huggingface_message_key_map=None,
                )
                mock_preview_result.display.assert_called_once()

        # Test deploy command
        with patch(
            "reward_kit.cli_commands.deploy.create_evaluation"
        ) as mock_create:  # Corrected patch target
            # Configure the mock
            mock_create.return_value = {
                "name": "accounts/test_account/evaluators/test-eval",
                "displayName": "Test Evaluator",
                "description": "Test description",
                "multiMetrics": False,
            }

            # Create args
            args = MagicMock()
            args.metrics_folders = [f"test_metric={tmp_dir}"]
            args.id = "test-eval"
            args.display_name = "Test Evaluator"
            args.description = "Test description"
            args.force = False
            # Add HuggingFace attributes with None values
            args.huggingface_dataset = None
            args.huggingface_split = "train"
            args.huggingface_prompt_key = "prompt"
            args.huggingface_response_key = "response"
            args.huggingface_key_map = None

            # Run deploy command
            result = deploy_command(args)

            # Verify the result
            assert result == 0
            mock_create.assert_called_once_with(
                evaluator_id="test-eval",
                metric_folders=[f"test_metric={tmp_dir}"],
                display_name="Test Evaluator",
                description="Test description",
                force=False,
                huggingface_dataset=None,
                huggingface_split="train",
                huggingface_message_key_map=None,
                huggingface_prompt_key="prompt",
                huggingface_response_key="response",
            )

    finally:
        # Clean up
        os.unlink(os.path.join(tmp_dir, "main.py"))
        os.rmdir(tmp_dir)
        os.unlink(sample_file)
