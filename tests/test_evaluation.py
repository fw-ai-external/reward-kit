import json
import os
import tempfile
from pathlib import Path

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


def test_evaluator_preview(monkeypatch):
    """Test preview functionality"""
    # Mock authentication and API endpoint for preview
    monkeypatch.setenv("FIREWORKS_API_KEY", "test_preview_api_key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "test_preview_account")
    # Using a mock API base to prevent real calls and simulate fallback
    monkeypatch.setenv("FIREWORKS_API_BASE", "http://localhost:12345/mock_api")

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
        # Assuming preview_result.results[0] is an object, use attribute access
        assert preview_result.results[0].index == 0
        assert preview_result.results[0].success is True
        assert hasattr(
            preview_result.results[0], "score"
        )  # Check if 'score' attribute exists
        assert hasattr(
            preview_result.results[0], "per_metric_evals"
        )  # Check for per_metric_evals
    finally:
        # Clean up
        os.unlink(os.path.join(tmp_dir, "main.py"))
        os.rmdir(tmp_dir)
        os.unlink(sample_file)


def test_preview_evaluation_helper(monkeypatch):
    """Test the preview_evaluation helper function"""
    # Mock authentication and API endpoint for preview
    monkeypatch.setenv("FIREWORKS_API_KEY", "test_helper_api_key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "test_helper_account")
    # Using a mock API base to prevent real calls and simulate fallback
    monkeypatch.setenv("FIREWORKS_API_BASE", "http://localhost:12345/mock_api_helper")

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
        # Check payload format for the new structure
        payload = kwargs.get("json", {})
        assert "evaluator" in payload
        assert "evaluatorId" in payload

        evaluator_data = payload["evaluator"]
        assert "criteria" in evaluator_data
        criteria = evaluator_data["criteria"]

        assert len(criteria) > 0, "Criteria list should not be empty"
        criterion = criteria[0]
        assert "type" in criterion
        assert criterion["type"] == "CODE_SNIPPETS"
        assert "codeSnippets" in criterion
        assert "fileContents" in criterion["codeSnippets"]
        assert (
            "main.py" in criterion["codeSnippets"]["fileContents"]
        )  # Assuming test_metric/main.py

        # Return a mock response consistent with the new structure if needed,
        # or a generic success response. The test asserts on the returned evaluator object.
        # The create_evaluation function returns the result of response.json().
        # The actual API returns a structure like:
        # { "evaluator": { "name": "...", "displayName": "...", ... } }
        # However, the old test was asserting on top-level keys from a flatter structure.
        # Let's make the mock response match what the SUT now expects from the API.
        # The SUT's create() method returns response.json() directly.
        # The test asserts evaluator["name"], evaluator["displayName"] etc.
        # This implies the mock response should be the content of the "evaluator" object itself,
        # or the test assertions need to change.
        # Given the SUT returns response.json(), the mock should return the full API response.

        # The `create_evaluation` helper calls `evaluator.create`, which returns `result = response.json()`.
        # The test then asserts `evaluator["name"]`. This means `result` should have a "name" key.
        # The actual API response for a successful creation is typically the full evaluator resource.
        # e.g. { "name": "accounts/...", "displayName": "...", ... }
        # Or if it's the new structure: { "evaluator": { "name": "...", ... } }
        # The `deploy_example.py` expects `deployment_result.get("evaluator", {}).get("name", ...)`
        # This suggests the API returns the nested structure.

        # Let's assume the mock should return the nested structure.
        return MockResponse(
            {  # This is the full response.json()
                "evaluator": {
                    "name": "accounts/test_account/evaluators/test-eval",
                    "displayName": "Test Evaluator",
                    "description": "Test description",
                    "multiMetrics": False,
                    # other fields like criteria, etc.
                },
                "evaluatorId": "test-eval",  # if this is part of the response
            }
        )

    # Apply the monkey patch
    monkeypatch.setattr("requests.post", mock_post)

    try:
        # create_evaluation returns the result of response.json()
        api_response = create_evaluation(
            evaluator_id="test-eval",
            metric_folders=[f"test_metric={tmp_dir}"],
            display_name="Test Evaluator",
            description="Test description",
        )

        # Assertions should now reflect the (potentially nested) structure of api_response
        # Based on deploy_example.py, it seems the actual evaluator data is under an "evaluator" key.
        assert "evaluator" in api_response
        created_evaluator_data = api_response["evaluator"]

        assert (
            created_evaluator_data["name"]
            == "accounts/test_account/evaluators/test-eval"
        )
        assert created_evaluator_data["displayName"] == "Test Evaluator"
        assert created_evaluator_data["description"] == "Test description"
    finally:
        # Clean up
        os.unlink(os.path.join(tmp_dir, "main.py"))
        os.rmdir(tmp_dir)
