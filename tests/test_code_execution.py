"""
Tests for code execution reward functions.
"""

import pytest
from reward_kit.rewards.code_execution import (
    extract_code_blocks,
    local_code_execution_reward,
    execute_python_code,
    execute_javascript_code,
    compare_outputs,
    string_similarity,
    e2b_code_execution_reward,
    execute_code_with_e2b,
    _HAS_E2B,
)
from reward_kit.models import RewardOutput


@pytest.mark.skipif(not _HAS_E2B, reason="E2B not installed")
class TestE2BCodeExecution:
    def test_e2b_reward_function_missing_e2b(self, monkeypatch):
        # Patch _HAS_E2B to False to simulate missing E2B package
        monkeypatch.setattr("reward_kit.rewards.code_execution._HAS_E2B", False)

        messages = [
            {"role": "user", "content": "Write a function to add two numbers"},
            {
                "role": "assistant",
                "content": """Here's a function to add two numbers:

```python
def add(a, b):
    return a + b

print(add(2, 3))
```

This will output `5`.
""",
            },
        ]

        result = e2b_code_execution_reward(
            messages=messages, expected_output="5", language="python"
        )

        assert isinstance(result, RewardOutput)
        assert result.score == 0.0
        assert "E2B package not installed" in result.metrics["error"].reason

        # Restore _HAS_E2B to its original value
        monkeypatch.setattr(
            "reward_kit.rewards.code_execution._HAS_E2B", _HAS_E2B
        )

    @pytest.mark.skipif(not _HAS_E2B, reason="E2B not installed")
    def test_execute_code_with_e2b_authentication(self, monkeypatch):
        """Test that authentication error is properly handled."""
        # Force E2B_API_KEY to None for this test
        monkeypatch.delenv("E2B_API_KEY", raising=False)

        code = "print('Hello, world!')"
        result = execute_code_with_e2b(code, language="python", api_key=None)

        assert result["success"] is False
        assert "API key is required" in result["error"]

    @pytest.mark.skipif(not _HAS_E2B, reason="E2B not installed")
    def test_e2b_reward_function_no_api_key(self, monkeypatch):
        """Test that missing API key is properly handled in the reward function."""
        # Ensure E2B_API_KEY is not set in environment
        monkeypatch.delenv("E2B_API_KEY", raising=False)

        messages = [
            {"role": "user", "content": "Write a function to add two numbers"},
            {
                "role": "assistant",
                "content": """```python
def add(a, b):
    return a + b

print(add(2, 3))
```""",
            },
        ]

        result = e2b_code_execution_reward(
            messages=messages,
            expected_output="5",
            language="python",
            api_key=None,
        )

        assert isinstance(result, RewardOutput)
        assert result.score == 0.0
        assert "API key is required" in result.metrics["error"].reason


class TestExtractCodeBlocks:
    def test_extract_python_code(self):
        text = """Here's a simple Python function:
        
```python
def add(a, b):
    return a + b
    
print(add(2, 3))
```

This will output `5`."""

        code_blocks = extract_code_blocks(text)

        assert len(code_blocks) == 1
        assert code_blocks[0]["language"] == "python"
        assert "def add(a, b):" in code_blocks[0]["code"]
        assert "print(add(2, 3))" in code_blocks[0]["code"]

    def test_extract_javascript_code(self):
        text = """Here's a simple JavaScript function:
        
```javascript
function add(a, b) {
    return a + b;
}

console.log(add(2, 3));
```

This will output `5`."""

        code_blocks = extract_code_blocks(text)

        assert len(code_blocks) == 1
        assert code_blocks[0]["language"] == "javascript"
        assert "function add(a, b) {" in code_blocks[0]["code"]
        assert "console.log(add(2, 3));" in code_blocks[0]["code"]

    def test_extract_multiple_code_blocks(self):
        text = """Here are some code examples:
        
```python
print("Hello from Python")
```

And another example:

```javascript
console.log("Hello from JavaScript");
```
"""

        code_blocks = extract_code_blocks(text)

        assert len(code_blocks) == 2
        assert code_blocks[0]["language"] == "python"
        assert code_blocks[1]["language"] == "javascript"

    def test_extract_with_language_filter(self):
        text = """Here are some code examples:
        
```python
print("Hello from Python")
```

And another example:

```javascript
console.log("Hello from JavaScript");
```
"""

        code_blocks = extract_code_blocks(text, language="python")

        assert len(code_blocks) == 1
        assert code_blocks[0]["language"] == "python"
        assert "Hello from Python" in code_blocks[0]["code"]

    def test_extract_with_no_language_specified(self):
        text = """Here's a code block with no language specified:
        
```
print("Hello, world!")
```
"""

        code_blocks = extract_code_blocks(text)

        assert len(code_blocks) == 1
        assert code_blocks[0]["language"] == "unknown"
        assert "Hello, world!" in code_blocks[0]["code"]


class TestExecutePythonCode:
    def test_simple_python_execution(self):
        code = "print('Hello, world!')"
        result = execute_python_code(code)

        assert result["success"] is True
        assert result["output"] == "Hello, world!"
        assert result["error"] is None

    def test_python_execution_with_error(self):
        code = "print(undefined_variable)"
        result = execute_python_code(code)

        assert result["success"] is False
        assert result["output"] is None
        assert "NameError" in result["error"]

    def test_python_execution_with_timeout(self):
        code = "import time; time.sleep(10); print('This should timeout')"
        result = execute_python_code(code, timeout=1)

        assert result["success"] is False
        assert result["output"] is None
        assert (
            "timeout" in result["error"].lower()
            or "timed out" in result["error"].lower()
        )


# Skip these tests if Node.js is not installed
# Since Node.js is available, we'll let these tests run normally
# @pytest.mark.xfail(reason="Skipping if Node.js not installed")
class TestExecuteJavaScriptCode:
    def test_simple_javascript_execution(self):
        code = "console.log('Hello, world!');"
        result = execute_javascript_code(code)

        assert result["success"] is True
        assert result["output"] == "Hello, world!"
        assert result["error"] is None

    def test_javascript_execution_with_error(self):
        code = "console.log(undefinedVariable);"
        result = execute_javascript_code(code)

        assert result["success"] is False
        assert result["output"] is None
        # Our improved sandbox may return different error messages
        assert (
            "undefined" in result["error"].lower()
            or "error" in result["error"].lower()
        )

    def test_javascript_execution_with_timeout(self):
        code = "setTimeout(() => { console.log('Done'); }, 10000);"
        result = execute_javascript_code(code, timeout=1)

        assert result["success"] is False
        assert result["output"] is None
        assert (
            "timeout" in result["error"].lower()
            or "timed out" in result["error"].lower()
        )


class TestCompareOutputs:
    def test_exact_match(self):
        actual = "42"
        expected = "42"
        similarity = compare_outputs(actual, expected)

        assert similarity == 1.0

    def test_whitespace_normalization(self):
        actual = "  Hello,   world!  "
        expected = "Hello, world!"
        similarity = compare_outputs(actual, expected)

        assert similarity == 1.0

    def test_numeric_comparison(self):
        actual = "42.01"
        expected = "42.0"
        similarity = compare_outputs(actual, expected)

        assert similarity > 0.9  # Very close

        actual = "50"
        expected = "42"
        similarity = compare_outputs(actual, expected)

        assert similarity < 0.9  # More different

    def test_multiline_comparison(self):
        actual = "Line 1\nLine 2\nLine 3"
        expected = "Line 1\nLine 2\nLine 3"
        similarity = compare_outputs(actual, expected)

        assert similarity == 1.0

        actual = "Line 1\nLine 2\nLine X"  # One line different
        expected = "Line 1\nLine 2\nLine 3"
        similarity = compare_outputs(actual, expected)

        assert 0.7 < similarity < 1.0  # High but not perfect

    def test_list_comparison(self):
        actual = "[1, 2, 3]"
        expected = "[1, 2, 3]"
        similarity = compare_outputs(actual, expected)

        assert similarity == 1.0

        actual = "[1, 2, 3, 4]"  # Extra item
        expected = "[1, 2, 3]"
        similarity = compare_outputs(actual, expected)

        assert 0.7 < similarity < 1.0  # High but not perfect


class TestLocalCodeExecutionReward:
    def test_python_success_match(self):
        messages = [
            {"role": "user", "content": "Write a function to add two numbers"},
            {
                "role": "assistant",
                "content": """Here's a function to add two numbers:

```python
def add(a, b):
    return a + b

print(add(2, 3))
```

This will output `5`.
""",
            },
        ]

        result = local_code_execution_reward(
            messages=messages, expected_output="5", language="python"
        )

        assert isinstance(result, RewardOutput)
        assert result.score == 1.0
        assert (
            "executed successfully" in result.metrics["execution_result"].reason
        )

    def test_python_success_mismatch(self):
        messages = [
            {"role": "user", "content": "Write a function to add two numbers"},
            {
                "role": "assistant",
                "content": """Here's a function to add two numbers:

```python
def add(a, b):
    return a + b

print(add(1, 3))
```

This will output `4`.
""",
            },
        ]

        result = local_code_execution_reward(
            messages=messages, expected_output="5", language="python"
        )

        assert isinstance(result, RewardOutput)
        assert result.score < 1.0
        assert "Output similarity:" in result.metrics["output_match"].reason

    def test_code_execution_failure(self):
        messages = [
            {"role": "user", "content": "Write a function to add two numbers"},
            {
                "role": "assistant",
                "content": """Here's a function to add two numbers:

```python
def add(a, b):
    return a + b

print(add(undeclared_variable, 3))
```

This will output `5`.
""",
            },
        ]

        result = local_code_execution_reward(
            messages=messages, expected_output="5", language="python"
        )

        assert isinstance(result, RewardOutput)
        assert result.score == 0.0
        assert "failed with error" in result.metrics["execution_result"].reason

    def test_extract_expected_output_from_message(self):
        messages = [
            {
                "role": "user",
                "content": "Write a function to add two numbers. Expected output: 5",
            },
            {
                "role": "assistant",
                "content": """Here's a function to add two numbers:

```python
def add(a, b):
    return a + b

print(add(2, 3))
```

This will output `5`.
""",
            },
        ]

        result = local_code_execution_reward(
            messages=messages,
            original_messages=messages,  # Using the same messages
            language="python",
        )

        assert isinstance(result, RewardOutput)
        assert result.score == 1.0
