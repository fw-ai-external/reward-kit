"""
Tests for C/C++ code execution reward functions.
"""

import os
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any
from reward_kit.models import MetricRewardOutput

from reward_kit.rewards.cpp_code import (
    extract_code_blocks,
    add_cpp_includes,
    add_c_includes,
    compare_outputs,
    string_similarity,
    PistonClient,
    execute_cpp_code,
    ioi_cpp_code_reward,
    binary_cpp_code_reward
)
from reward_kit.models import RewardOutput

# Example C++ code for testing
SAMPLE_CPP_CODE = """
#include <iostream>
using namespace std;

int main() {
    int a, b;
    cin >> a >> b;
    cout << a + b << endl;
    return 0;
}
"""

SAMPLE_C_CODE = """
#include <stdio.h>

int main() {
    int a, b;
    scanf("%d %d", &a, &b);
    printf("%d\\n", a + b);
    return 0;
}
"""


class TestExtractCodeBlocks:
    def test_extract_cpp_code(self):
        text = """Here's a simple C++ function:
        
```cpp
#include <iostream>
using namespace std;

int main() {
    cout << "Hello, World!" << endl;
    return 0;
}
```

This will output `Hello, World!`."""

        code_blocks = extract_code_blocks(text)
        
        assert len(code_blocks) == 1
        assert code_blocks[0]["language"] == "cpp"
        assert "#include <iostream>" in code_blocks[0]["code"]
        assert "cout << \"Hello, World!\" << endl;" in code_blocks[0]["code"]
    
    def test_extract_c_code(self):
        text = """Here's a simple C function:
        
```c
#include <stdio.h>

int main() {
    printf("Hello, World!\\n");
    return 0;
}
```

This will output `Hello, World!`."""

        code_blocks = extract_code_blocks(text, language="c")
        
        assert len(code_blocks) == 1
        assert code_blocks[0]["language"] == "c"
        assert "#include <stdio.h>" in code_blocks[0]["code"]
        assert "printf(\"Hello, World!\\n\");" in code_blocks[0]["code"]
    
    def test_extract_with_language_filter(self):
        text = """Here are multiple code blocks:
        
```python
print("Hello, Python!")
```

```cpp
#include <iostream>
using namespace std;

int main() {
    cout << "Hello, C++!" << endl;
    return 0;
}
```

```c
#include <stdio.h>

int main() {
    printf("Hello, C!\\n");
    return 0;
}
```
"""

        cpp_blocks = extract_code_blocks(text, language="cpp")
        assert len(cpp_blocks) == 1
        assert cpp_blocks[0]["language"] == "cpp"
        assert "Hello, C++" in cpp_blocks[0]["code"]
        
        c_blocks = extract_code_blocks(text, language="c")
        assert len(c_blocks) == 1
        assert c_blocks[0]["language"] == "c"
        assert "Hello, C!" in c_blocks[0]["code"]


class TestIncludes:
    def test_add_cpp_includes(self):
        code = "int main() { return 0; }"
        
        enhanced_code = add_cpp_includes(code)
        
        assert "#include <iostream>" in enhanced_code
        assert "#include <vector>" in enhanced_code
        assert "#include <string>" in enhanced_code
        assert "#include <bits/stdc++.h>" in enhanced_code
        assert "using namespace std;" in enhanced_code
        assert code in enhanced_code
    
    def test_add_c_includes(self):
        code = "int main() { return 0; }"
        
        enhanced_code = add_c_includes(code)
        
        assert "#include <stdio.h>" in enhanced_code
        assert "#include <stdlib.h>" in enhanced_code
        assert "#include <string.h>" in enhanced_code
        assert code in enhanced_code
    
    def test_dont_duplicate_includes(self):
        code = """
#include <iostream>
using namespace std;

int main() { return 0; }
"""
        
        enhanced_code = add_cpp_includes(code)
        
        # Count occurrences of iostream include
        assert enhanced_code.count("#include <iostream>") == 1
        assert enhanced_code.count("using namespace std;") == 1


class TestOutputComparison:
    def test_exact_match(self):
        actual = "42"
        expected = "42"
        similarity = compare_outputs(actual, expected)
        
        assert similarity == 1.0
    
    def test_whitespace_normalization(self):
        actual = "  42  \n"
        expected = "42"
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
    
    def test_string_similarity(self):
        assert string_similarity("hello", "hello") == 1.0
        assert string_similarity("hello", "hallo") < 1.0
        assert string_similarity("", "") == 1.0
        assert string_similarity("hello", "") == 0.0
        assert string_similarity("", "hello") == 0.0


# Mock response for Piston API
MOCK_PISTON_RUNTIME_RESPONSE = [
    {
        "language": "cpp",
        "version": "11.4.0",
        "aliases": ["c++"],
        "runtime": "gcc"
    },
    {
        "language": "c",
        "version": "11.3.0",
        "aliases": [],
        "runtime": "gcc"
    }
]

MOCK_PISTON_EXECUTE_SUCCESS = {
    "language": "cpp",
    "version": "11.4.0",
    "compile": {
        "stdout": "",
        "stderr": "",
        "code": 0,
        "signal": None
    },
    "run": {
        "stdout": "42",
        "stderr": "",
        "code": 0,
        "signal": None
    }
}

MOCK_PISTON_EXECUTE_COMPILE_ERROR = {
    "language": "cpp",
    "version": "11.4.0",
    "compile": {
        "stdout": "",
        "stderr": "main.cpp:5:10: error: 'cout' was not declared in this scope",
        "code": 1,
        "signal": None
    }
}

MOCK_PISTON_EXECUTE_RUNTIME_ERROR = {
    "language": "cpp",
    "version": "11.4.0",
    "compile": {
        "stdout": "",
        "stderr": "",
        "code": 0,
        "signal": None
    },
    "run": {
        "stdout": "",
        "stderr": "Segmentation fault",
        "code": 139,
        "signal": "SIGSEGV"
    }
}


@pytest.mark.asyncio
class TestPistonClient:
    @patch("aiohttp.ClientSession.get")
    async def test_get_runtimes(self, mock_get):
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = MagicMock(return_value=MOCK_PISTON_RUNTIME_RESPONSE)
        mock_get.return_value.__aenter__.return_value = mock_response
        
        # Create client and call method
        client = PistonClient()
        runtimes = await client.get_runtimes()
        
        # Check result
        assert runtimes == MOCK_PISTON_RUNTIME_RESPONSE
        assert len(runtimes) == 2
        assert runtimes[0]["language"] == "cpp"
        assert runtimes[1]["language"] == "c"
    
    @patch("aiohttp.ClientSession.post")
    async def test_execute_success(self, mock_post):
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = MagicMock(return_value=MOCK_PISTON_EXECUTE_SUCCESS)
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Create client and call method
        client = PistonClient()
        result = await client.execute(
            language="cpp",
            version="11.4.0",
            files=[{"name": "main.cpp", "content": SAMPLE_CPP_CODE}],
            stdin="10 15"
        )
        
        # Check result
        assert result == MOCK_PISTON_EXECUTE_SUCCESS
        assert result["run"]["stdout"] == "42"
        assert result["run"]["code"] == 0
    
    @patch("aiohttp.ClientSession.post")
    async def test_execute_compile_error(self, mock_post):
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = MagicMock(return_value=MOCK_PISTON_EXECUTE_COMPILE_ERROR)
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Create client and call method
        client = PistonClient()
        result = await client.execute(
            language="cpp",
            version="11.4.0",
            files=[{"name": "main.cpp", "content": "int main() { cout << 42; }"}],
            stdin=""
        )
        
        # Check result
        assert result == MOCK_PISTON_EXECUTE_COMPILE_ERROR
        assert result["compile"]["code"] == 1
        assert "error:" in result["compile"]["stderr"]
    
    @patch("aiohttp.ClientSession.post")
    async def test_execute_runtime_error(self, mock_post):
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = MagicMock(return_value=MOCK_PISTON_EXECUTE_RUNTIME_ERROR)
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Create client and call method
        client = PistonClient()
        result = await client.execute(
            language="cpp",
            version="11.4.0",
            files=[{"name": "main.cpp", "content": "int main() { int* p = nullptr; *p = 42; }"}],
            stdin=""
        )
        
        # Check result
        assert result == MOCK_PISTON_EXECUTE_RUNTIME_ERROR
        assert result["run"]["code"] != 0
        assert result["run"]["stderr"] == "Segmentation fault"


@pytest.mark.asyncio
class TestExecuteCppCode:
    @patch("reward_kit.rewards.cpp_code.PistonClient.execute")
    async def test_execute_cpp_success(self, mock_execute):
        # Set up mock response
        mock_execute.return_value = MOCK_PISTON_EXECUTE_SUCCESS
        
        # Call function
        result = await execute_cpp_code(SAMPLE_CPP_CODE, stdin="10 15")
        
        # Check result
        assert result["success"] is True
        assert result["output"] == "42"
        assert result["error"] is None
    
    @patch("reward_kit.rewards.cpp_code.PistonClient.execute")
    async def test_execute_cpp_compile_error(self, mock_execute):
        # Set up mock response
        mock_execute.return_value = MOCK_PISTON_EXECUTE_COMPILE_ERROR
        
        # Call function
        result = await execute_cpp_code("int main() { cout << 42; }")
        
        # Check result
        assert result["success"] is False
        assert result["output"] is None
        assert "Compilation error" in result["error"]
    
    @patch("reward_kit.rewards.cpp_code.PistonClient.execute")
    async def test_execute_c_code(self, mock_execute):
        # Set up mock response
        mock_execute.return_value = MOCK_PISTON_EXECUTE_SUCCESS
        
        # Call function
        result = await execute_cpp_code(SAMPLE_C_CODE, language="c", stdin="10 15")
        
        # Check result
        assert result["success"] is True
        assert result["output"] == "42"


@pytest.mark.asyncio
class TestIOICppCodeReward:
    @patch("reward_kit.rewards.cpp_code.execute_cpp_code")
    async def test_success_match(self, mock_execute):
        # Set up mock response
        mock_execute.return_value = {"success": True, "output": "42", "error": None}
        
        messages = [
            {"role": "user", "content": "Write a C++ program to add two numbers"},
            {"role": "assistant", "content": f"""Here's a program to add two numbers:

```cpp
{SAMPLE_CPP_CODE}
```

This program reads two integers and outputs their sum.
"""}
        ]
        
        # Call function
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(ioi_cpp_code_reward(
            messages=messages,
            expected_output="42",
            language="cpp"
        ))
        
        # Check result
        assert isinstance(result, RewardOutput)
        assert result.score == 1.0
        assert "executed successfully" in result.metrics["execution_result"].reason
    
    @patch("reward_kit.rewards.cpp_code.execute_cpp_code")
    async def test_success_mismatch(self, mock_execute):
        # Set up mock response
        mock_execute.return_value = {"success": True, "output": "25", "error": None}
        
        messages = [
            {"role": "user", "content": "Write a C++ program to add two numbers"},
            {"role": "assistant", "content": f"""Here's a program to add two numbers:

```cpp
{SAMPLE_CPP_CODE}
```

This program reads two integers and outputs their sum.
"""}
        ]
        
        # Call function
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(ioi_cpp_code_reward(
            messages=messages,
            expected_output="42",
            language="cpp"
        ))
        
        # Check result
        assert isinstance(result, RewardOutput)
        assert result.score < 1.0
        assert "Output similarity:" in result.metrics["output_match"].reason
    
    @patch("reward_kit.rewards.cpp_code.execute_cpp_code")
    async def test_execution_failure(self, mock_execute):
        # Set up mock response
        mock_execute.return_value = {"success": False, "output": None, "error": "Compilation error"}
        
        messages = [
            {"role": "user", "content": "Write a C++ program to add two numbers"},
            {"role": "assistant", "content": """Here's a program to add two numbers:

```cpp
int main() {
    cout << "Hello, World!" << endl;  // Missing iostream include
    return 0;
}
```
"""}
        ]
        
        # Call function
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(ioi_cpp_code_reward(
            messages=messages,
            expected_output="42",
            language="cpp"
        ))
        
        # Check result
        assert isinstance(result, RewardOutput)
        assert result.score == 0.0
        assert "failed with error" in result.metrics["execution_result"].reason
    
    @patch("reward_kit.rewards.cpp_code.run_cpp_test_cases")
    async def test_multiple_test_cases(self, mock_run_tests):
        # Set up mock response
        from reward_kit.rewards.cpp_code import TestResult
        mock_run_tests.return_value = [
            TestResult(test_name="Test 1", score=1.0, status="AC", feedback="Perfect"),
            TestResult(test_name="Test 2", score=1.0, status="AC", feedback="Perfect"),
            TestResult(test_name="Test 3", score=0.5, status="PA", feedback="Partial")
        ]
        
        messages = [
            {"role": "user", "content": "Write a C++ program to add two numbers"},
            {"role": "assistant", "content": f"""Here's a program to add two numbers:

```cpp
{SAMPLE_CPP_CODE}
```

This program reads two integers and outputs their sum.
"""}
        ]
        
        test_cases = [
            {"name": "Test 1", "input": "10 15", "expected_output": "25"},
            {"name": "Test 2", "input": "0 0", "expected_output": "0"},
            {"name": "Test 3", "input": "-5 5", "expected_output": "0"}
        ]
        
        # Call function
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(ioi_cpp_code_reward(
            messages=messages,
            test_cases=test_cases,
            language="cpp"
        ))
        
        # Check result
        assert isinstance(result, RewardOutput)
        assert result.score == (1.0 + 1.0 + 0.5) / 3  # Average of test scores
        assert "tests passed" in result.metrics["pass_rate"].reason


class TestBinaryCppCodeReward:
    @patch("reward_kit.rewards.cpp_code.ioi_cpp_code_reward")
    def test_binary_pass(self, mock_reward):
        # Set up mock response
        mock_metrics = {
            "execution_result": MetricRewardOutput(
                score=1.0, reason="Code executed successfully"
            ),
            "output_match": MetricRewardOutput(
                score=1.0, reason="Perfect match"
            )
        }
        mock_reward.return_value = asyncio.Future()
        mock_reward.return_value.set_result(RewardOutput(score=1.0, metrics=mock_metrics))
        
        messages = [
            {"role": "user", "content": "Write a C++ program to add two numbers"},
            {"role": "assistant", "content": f"""Here's a program to add two numbers:

```cpp
{SAMPLE_CPP_CODE}
```
"""}
        ]
        
        # Call function
        result = binary_cpp_code_reward(
            messages=messages,
            expected_output="25",
            language="cpp"
        )
        
        # Check result
        assert isinstance(result, RewardOutput)
        assert result.score == 1.0
        assert "Passed" in result.metrics["binary_result"].reason
    
    @patch("reward_kit.rewards.cpp_code.ioi_cpp_code_reward")
    def test_binary_fail(self, mock_reward):
        # Set up mock response
        mock_metrics = {
            "execution_result": MetricRewardOutput(
                score=1.0, reason="Code executed successfully"
            ),
            "output_match": MetricRewardOutput(
                score=0.8, reason="Close match"
            )
        }
        mock_reward.return_value = asyncio.Future()
        mock_reward.return_value.set_result(RewardOutput(score=0.8, metrics=mock_metrics))
        
        messages = [
            {"role": "user", "content": "Write a C++ program to add two numbers"},
            {"role": "assistant", "content": f"""Here's a program to add two numbers:

```cpp
{SAMPLE_CPP_CODE}
```
"""}
        ]
        
        # Call function
        result = binary_cpp_code_reward(
            messages=messages,
            expected_output="25",
            language="cpp",
            pass_threshold=0.9  # Set threshold higher than the actual score
        )
        
        # Check result
        assert isinstance(result, RewardOutput)
        assert result.score == 0.0
        assert "Failed" in result.metrics["binary_result"].reason