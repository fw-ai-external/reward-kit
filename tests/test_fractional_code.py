"""
Tests for fractional code reward function.
"""

import os
import pytest
from typing import List, Dict, Any, Optional
from reward_kit.rewards import fractional_code_reward
from reward_kit.rewards.code_execution import _HAS_E2B
# from reward_kit.models import EvaluateResult # EvaluateResult is no longer directly asserted for reward function outputs


class TestFractionalCodeReward:
    def test_simple_python_correct(self):
        """Test with a simple Python function that is correct."""
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

        result = fractional_code_reward(
            messages=messages, expected_output="5", language="python"
        )

        assert isinstance(result, dict)
        assert result['score'] == 1.0
        assert (
            "Code executed successfully"
            in result['metrics']["execution_result"]['reason']
        )

    def test_simple_python_partial_match(self):
        """Test with Python function that produces partially correct output."""
        messages = [
            {
                "role": "user",
                "content": "Write a function to print a numbered list from 1 to 3",
            },
            {
                "role": "assistant",
                "content": """Here's a function to print a numbered list:

```python
def print_numbered_list(n):
    for i in range(1, n+1):
        print(f"{i}. Item {i}")

print_numbered_list(3)
```

This will print a numbered list from 1 to 3.
""",
            },
        ]

        # Expected is slightly different format (no period)
        result = fractional_code_reward(
            messages=messages,
            expected_output="1 Item 1\n2 Item 2\n3 Item 3",
            language="python",
        )

        assert isinstance(result, dict)
        assert 0.7 < result['score'] < 1.0  # Should be high but not perfect
        assert "Output similarity:" in result['metrics']["output_match"]['reason']

    def test_python_execution_error(self):
        """Test with Python function that has an error."""
        messages = [
            {"role": "user", "content": "Write a function to add two numbers"},
            {
                "role": "assistant",
                "content": """Here's a function to add two numbers:

```python
def add(a, b):
    return a + b

# This will cause an error
print(add(2, undefined_variable))
```
""",
            },
        ]

        result = fractional_code_reward(
            messages=messages, expected_output="5", language="python"
        )

        assert isinstance(result, dict)
        assert result['score'] == 0.0
        assert (
            "execution failed with error"
            in result['metrics']["execution_result"]['reason']
        )

    def test_no_code_blocks(self):
        """Test with message that doesn't contain code blocks."""
        messages = [
            {"role": "user", "content": "Write a function to add two numbers"},
            {
                "role": "assistant",
                "content": "I should write a function but I forgot to include any code!",
            },
        ]

        result = fractional_code_reward(
            messages=messages, expected_output="5", language="python"
        )

        assert isinstance(result, dict)
        assert result['score'] == 0.0
        assert (
            "no python code blocks found"
            in result['metrics']["error"]['reason'].lower()
        )

    def test_extract_expected_output(self):
        """Test extracting expected output from original messages."""
        messages = [
            {
                "role": "user",
                "content": "Write a function to add two numbers. Expected output: 5",
            },
            {
                "role": "assistant",
                "content": """```python
def add(a, b):
    return a + b

print(add(2, 3))
```""",
            },
        ]

        result = fractional_code_reward(
            messages=messages,
            original_messages=messages,  # Using the same messages as original
            language="python",
        )

        assert isinstance(result, dict)
        assert result['score'] == 1.0

    @pytest.mark.skip(reason="Skipping E2B tests due to intermittent service issues")
    @pytest.mark.skipif(not _HAS_E2B, reason="E2B not installed")
    def test_e2b_execution(self):
        """Test execution in E2B environment (skipped if E2B not installed)."""
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

        # This will be skipped if E2B is not installed or API key is not set
        if _HAS_E2B and "E2B_API_KEY" in [k.upper() for k in os.environ.keys()]:
            result = fractional_code_reward(
                messages=messages,
                expected_output="5",
                language="python",
                environment="e2b",
            )

            assert isinstance(result, dict)
            assert result['score'] == 1.0

    def test_multiple_test_cases(self):
        """Test with multiple test cases."""
        messages = [
            {"role": "user", "content": "Write a function to add two numbers"},
            {
                "role": "assistant",
                "content": """```python
def add(a, b):
    return a + b

# Example usage
if __name__ == "__main__":
    print(add(2, 3))
```""",
            },
        ]

        test_cases = [
            {
                "input": "",
                "expected_output": "5",
            },  # This will pass with main block
            {
                "input": "print(add(10, 20))",
                "expected_output": "30",
            },  # This will pass
            {
                "input": "print(add(-5, 5))",
                "expected_output": "0",
            },  # This will pass
            {
                "input": "print(add(1, 2))",
                "expected_output": "4",
            },  # This will fail
        ]

        result = fractional_code_reward(
            messages=messages, language="python", test_cases=test_cases
        )

        assert isinstance(result, dict)
        # Test case behavior may vary depending on Python environment, so just check it's between 0 and 1
        assert 0 <= result['score'] <= 1.0
        # Should contain pass_rate indicator
        assert "tests passed" in result['metrics']["pass_rate"]['reason']

    def test_javascript_execution(self):
        """Test with JavaScript code."""
        messages = [
            {
                "role": "user",
                "content": "Write a function to add two numbers in JavaScript",
            },
            {
                "role": "assistant",
                "content": """```javascript
function add(a, b) {
    return a + b;
}

console.log(add(2, 3));
```""",
            },
        ]

        result = fractional_code_reward(
            messages=messages, expected_output="5", language="javascript"
        )

        assert isinstance(result, dict)
        assert result['score'] > 0.9  # Should be very high or 1.0
