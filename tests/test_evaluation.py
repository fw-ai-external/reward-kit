import os
import json
import tempfile
from pathlib import Path
import pytest

from reward_kit.evaluation import (
    Evaluator,
    preview_evaluation,
    create_evaluation,
)


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
            "original_messages": [
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


def test_evaluator_load_metric_folder():
    """Test loading metric folder"""
    tmp_dir = create_test_folder()
    try:
        evaluator = Evaluator()
        files = evaluator.load_metric_folder("test_metric", tmp_dir)

        assert "main.py" in files
        assert "test_metric" in evaluator.metric_folders
        assert "test_metric/main.py" in evaluator.code_files
        assert "evaluate" in evaluator.code_files["test_metric/main.py"]
    finally:
        # Clean up
        os.unlink(os.path.join(tmp_dir, "main.py"))
        os.rmdir(tmp_dir)


def test_evaluator_load_multi_metrics_folder():
    """Test loading multi-metrics folder"""
    tmp_dir = create_test_folder()
    try:
        evaluator = Evaluator(multi_metrics=True)
        files = evaluator.load_multi_metrics_folder(tmp_dir)

        assert "main.py" in files
        assert "main.py" in evaluator.code_files
        assert "evaluate" in evaluator.code_files["main.py"]
    finally:
        # Clean up
        os.unlink(os.path.join(tmp_dir, "main.py"))
        os.rmdir(tmp_dir)


def test_evaluator_update_evaluate_signature():
    """Test the evaluate signature updating function"""
    evaluator = Evaluator()

    # Test with old style signature
    old_code = """
def evaluate(entry):
    messages = entry.get('messages', [])
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

    updated_code = evaluator._update_evaluate_signature(old_code)

    # Check that signature was updated
    assert (
        "def evaluate(messages, original_messages=None, tools=None, **kwargs)"
        in updated_code
    )
    assert "entry = {" in updated_code
    assert "original_messages = messages" in updated_code

    # Test with new style signature - should not change
    new_code = """
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

    unchanged_code = evaluator._update_evaluate_signature(new_code)
    assert new_code == unchanged_code


def test_evaluator_preview():
    """Test preview functionality"""
    tmp_dir = create_test_folder()
    sample_file = create_sample_file()

    try:
        evaluator = Evaluator()
        evaluator.load_metric_folder("test_metric", tmp_dir)

        preview_result = evaluator.preview(sample_file, max_samples=2)

        assert preview_result.total_samples == 2
        assert preview_result.total_runtime_ms > 0
        assert len(preview_result.results) == 2

        # Check first result
        assert preview_result.results[0]["index"] == 0
        assert preview_result.results[0]["success"] is True
        assert "score" in preview_result.results[0]
        assert "per_metric_evals" in preview_result.results[0]
    finally:
        # Clean up
        os.unlink(os.path.join(tmp_dir, "main.py"))
        os.rmdir(tmp_dir)
        os.unlink(sample_file)


def test_preview_evaluation_helper():
    """Test the preview_evaluation helper function"""
    tmp_dir = create_test_folder()
    sample_file = create_sample_file()

    try:
        preview_result = preview_evaluation(
            metric_folders=[f"test_metric={tmp_dir}"],
            sample_file=sample_file,
            max_samples=2,
        )

        assert preview_result.total_samples == 2
        assert len(preview_result.results) == 2
    finally:
        # Clean up
        os.unlink(os.path.join(tmp_dir, "main.py"))
        os.rmdir(tmp_dir)
        os.unlink(sample_file)


def test_create_evaluation_helper(monkeypatch):
    """Test the create_evaluation helper function"""
    tmp_dir = create_test_folder()

    # Mock authentication and API endpoint
    monkeypatch.setenv("FIREWORKS_API_KEY", "test_api_key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "test_account")
    monkeypatch.setenv(
        "FIREWORKS_API_BASE", "https://api.fireworks.ai"
    )  # Ensure standard API format

    # Mock requests.post to avoid actual API calls
    class MockResponse:
        def __init__(self, json_data, status_code=200):
            self.json_data = json_data
            self.status_code = status_code
            self.text = json.dumps(json_data)

        def json(self):
            return self.json_data

        def raise_for_status(self):
            if self.status_code != 200:
                raise Exception("API Error")

    def mock_post(*args, **kwargs):
        # Check payload format
        payload = kwargs.get("json", {})
        assertions = payload.get("evaluation", {}).get("assertions", [])

        assert len(assertions) > 0
        assert "assertionType" in assertions[0]
        assert assertions[0]["assertionType"] == "CODE"
        assert "codeAssertion" in assertions[0]

        return MockResponse(
            {
                "name": "accounts/test_account/evaluators/test-eval",
                "displayName": "Test Evaluator",
                "description": "Test description",
                "multiMetrics": False,
            }
        )

    # Apply the monkey patch
    monkeypatch.setattr("requests.post", mock_post)

    try:
        evaluator = create_evaluation(
            evaluator_id="test-eval",
            metric_folders=[f"test_metric={tmp_dir}"],
            display_name="Test Evaluator",
            description="Test description",
        )

        assert evaluator["name"] == "accounts/test_account/evaluators/test-eval"
        assert evaluator["displayName"] == "Test Evaluator"
        assert evaluator["description"] == "Test description"
    finally:
        # Clean up
        os.unlink(os.path.join(tmp_dir, "main.py"))
        os.rmdir(tmp_dir)
