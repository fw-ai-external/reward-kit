import os
import json
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch

from reward_kit.evaluation import Evaluator, preview_evaluation, create_evaluation


def create_test_folder():
    """Create a temporary folder with a main.py file for testing"""
    tmp_dir = tempfile.mkdtemp()
    
    # Create main.py
    with open(os.path.join(tmp_dir, "main.py"), "w") as f:
        f.write("""
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
""")
    
    return tmp_dir


def create_sample_file():
    """Create a temporary sample file for testing"""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    
    samples = [
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there! How can I help you today?"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI stands for Artificial Intelligence."}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search for information"
                    }
                }
            ]
        }
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
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "name": "accounts/test_account/evaluators/test-eval",
            "displayName": "Test Evaluator",
            "description": "Test description",
            "multiMetrics": False
        }
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
            max_samples=2
        )
        
        assert preview_result.total_samples == 2
        assert len(preview_result.results) == 2
        
        # Create the evaluation
        evaluator = create_evaluation(
            evaluator_id="test-eval",
            metric_folders=[f"test_metric={tmp_dir}"],
            display_name="Test Evaluator",
            description="Test description"
        )
        
        assert evaluator["name"] == "accounts/test_account/evaluators/test-eval"
        assert evaluator["displayName"] == "Test Evaluator"
        
        # Verify API call
        mock_requests_post.assert_called_once()
        args, kwargs = mock_requests_post.call_args
        url = args[0]
        payload = kwargs.get('json')
        
        assert "api.fireworks.ai/v1/accounts/test_account/evaluators" in url
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
            max_samples=2
        )
        
        assert preview_result.total_samples == 2
        assert len(preview_result.results) == 2
        
        # Check that we get expected metrics in multi-metrics mode
        assert "quality" in preview_result.results[0]["per_metric_evals"]
        assert "relevance" in preview_result.results[0]["per_metric_evals"]
        assert "safety" in preview_result.results[0]["per_metric_evals"]
        
        # Create the evaluation
        mock_requests_post.reset_mock()
        mock_requests_post.return_value.json.return_value["multiMetrics"] = True
        
        evaluator = create_evaluation(
            evaluator_id="multi-metrics-eval",
            multi_metrics=True,
            folder=tmp_dir,
            display_name="Multi Metrics Evaluator",
            description="Test multi-metrics evaluator"
        )
        
        assert evaluator["name"] == "accounts/test_account/evaluators/test-eval"
        
        # Verify API call
        mock_requests_post.assert_called_once()
        args, kwargs = mock_requests_post.call_args
        payload = kwargs.get('json')
        
        assert payload["evaluationId"] == "multi-metrics-eval"
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


@patch('typer.Exit')
def test_integration_cli_commands(mock_exit, mock_env_variables, mock_requests_post):
    """Test CLI integration by directly calling the CLI command functions"""
    from reward_kit.cli import preview_cmd, create_cmd
    import typer
    from typing import List
    
    # Make typer.Exit a pass-through instead of raising an exception
    mock_exit.return_value = None
    mock_exit.side_effect = lambda code=0: None
    
    tmp_dir = create_test_folder()
    sample_file = create_sample_file()
    
    try:
        # Mock typer.echo to capture output
        with patch('typer.echo') as mock_echo:
            # Test preview command
            preview_cmd(
                metric_folder=[f"test_metric={tmp_dir}"],
                sample_file=sample_file,
                multi_metrics=False,
                folder=None,
                max_samples=2
            )
            
            # Create a new mock for create_evaluation to avoid real calls
            with patch('reward_kit.cli.create_evaluation') as mock_create:
                # Configure the mock
                mock_create.return_value = {
                    "name": "accounts/test_account/evaluators/test-eval",
                    "displayName": "Test Evaluator",
                    "description": "Test description",
                    "multiMetrics": False
                }
                
                # Test create command
                create_cmd(
                    eval_id="test-eval",
                    metric_folder=[f"test_metric={tmp_dir}"],
                    multi_metrics=False,
                    folder=None,
                    display_name="Test Evaluator",
                    description="Test description"
                )
                
                # Verify the mock was called correctly
                mock_create.assert_called_once_with(
                    evaluator_id="test-eval",
                    metric_folders=[f"test_metric={tmp_dir}"],
                    multi_metrics=False,
                    folder=None,
                    display_name="Test Evaluator",
                    description="Test description"
                )
                
                # Verify echo was called with success message
                success_message = f"Successfully created evaluator: accounts/test_account/evaluators/test-eval"
                mock_echo.assert_any_call(success_message)
            
    finally:
        # Clean up
        os.unlink(os.path.join(tmp_dir, "main.py"))
        os.rmdir(tmp_dir)
        os.unlink(sample_file)