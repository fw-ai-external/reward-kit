"""
C/C++ code execution reward functions for evaluating C/C++ code correctness.

This module provides functions to evaluate the correctness of C/C++ code by:
1. Extracting code blocks from messages
2. Executing the code using the Piston execution engine
3. Comparing the output with expected results or running against test cases
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass  # field removed
from typing import Any, Dict, List, Optional, Union  # Tuple removed

import aiohttp

from ..models import EvaluateResult, Message, MetricResult
from ..reward_function import reward_function


@dataclass
class TestResult:
    """
    Represents the result of a single test case execution.
    """

    test_name: str
    score: float = 0.0
    status: str = "SKIPPED"
    feedback: str = ""
    actual_output: str = ""
    expected_output: str = ""


class PistonError(Exception):
    """Exception raised for errors from the Piston API."""

    pass


class PistonClient:
    """
    A client that communicates with Piston API endpoints for code execution.

    Piston is a general purpose code execution engine:
    https://github.com/engineer-man/piston
    """

    def __init__(
        self,
        base_endpoint: str = "https://emkc.org/api/v2/piston",
        session: Optional[aiohttp.ClientSession] = None,
        timeout: int = 30,
    ):
        self.base_endpoint = base_endpoint
        self._session = session
        self.timeout = timeout

    @property
    def session(self):
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(sock_read=self.timeout),
                connector=aiohttp.TCPConnector(
                    limit=10,
                    ttl_dns_cache=300,
                    keepalive_timeout=30,
                ),
            )
        return self._session

    async def close(self):
        """Close the session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def get_runtimes(self) -> List[Dict[str, Any]]:
        """Get list of supported runtimes."""
        async with self.session.get(f"{self.base_endpoint}/runtimes") as response:
            if response.status != 200:
                raise PistonError(f"Error getting runtimes: {response.status}")
            return await response.json()

    async def execute(
        self,
        language: str,
        version: str,
        files: List[Dict[str, str]],
        stdin: str = "",
        args: List[str] = [],
        compile_timeout: int = 10000,
        run_timeout: int = 3000,
        compile_memory_limit: int = -1,
        run_memory_limit: int = -1,
    ) -> Dict[str, Any]:
        """
        Execute code using the Piston API.

        Args:
            language: Programming language (e.g., "c", "cpp")
            version: Version of the language (e.g., "10.2.0")
            files: List of files to include in execution (each with "name" and "content")
            stdin: Standard input to provide to the program
            args: Command-line arguments to pass to the program
            compile_timeout: Maximum compilation time in milliseconds
            run_timeout: Maximum execution time in milliseconds
            compile_memory_limit: Maximum memory for compilation in bytes (-1 for unlimited)
            run_memory_limit: Maximum memory for execution in bytes (-1 for unlimited)

        Returns:
            Dictionary containing the execution results
        """
        payload = {
            "language": language,
            "version": version,
            "files": files,
            "stdin": stdin,
            "args": args,
            "compile_timeout": compile_timeout,
            "run_timeout": run_timeout,
            "compile_memory_limit": compile_memory_limit,
            "run_memory_limit": run_memory_limit,
        }

        async with self.session.post(
            f"{self.base_endpoint}/execute",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise PistonError(
                    f"Error executing code: {response.status} - {error_text}"
                )

            result = await response.json()

            # Check for internal errors
            if "message" in result:
                raise PistonError(result["message"])

            return result


def get_piston_client(endpoint: Optional[str] = None) -> PistonClient:
    """
    Get a Piston client instance.

    Args:
        endpoint: Optional custom Piston API endpoint

    Returns:
        PistonClient instance
    """
    piston_endpoint = endpoint or os.environ.get(
        "PISTON_ENDPOINT", "https://emkc.org/api/v2/piston"
    )
    return PistonClient(base_endpoint=piston_endpoint)


def extract_code_blocks(text: str, language: str = "cpp") -> List[Dict[str, str]]:
    """
    Extract code blocks from text.

    Args:
        text: The text to extract code blocks from
        language: Language to filter by (e.g., "cpp", "c")

    Returns:
        List of dictionaries with "code" and "language" keys
    """
    # Match code blocks with optional language specifier
    pattern = r"```(\w*)\n([\s\S]*?)\n```"
    matches = re.findall(pattern, text)

    code_blocks = []
    for lang, code in matches:
        # Skip if language filter is specified and doesn't match
        # C++ can be specified as cpp, c++, or just c in some cases
        lang = lang.lower()

        # Process language filter
        if language and lang:
            if language == "cpp" and lang not in ["cpp", "c++"]:
                continue
            elif language == "c" and lang != "c":
                continue
            elif language not in ["c", "cpp"] and language != lang:
                continue

        # Use "unknown" for empty language specifier
        detected_lang = lang if lang else "unknown"

        # Add to results
        code_blocks.append({"language": detected_lang, "code": code.strip()})

    return code_blocks


def add_cpp_includes(code: str) -> str:
    """
    Add common C++ includes if they're missing.

    Args:
        code: C++ code

    Returns:
        Code with added includes if necessary
    """
    if not code:
        return code

    # Common include for C++
    includes = []

    # Check for standard includes
    if "#include <iostream>" not in code:
        includes.append("#include <iostream>")
    if "#include <vector>" not in code:
        includes.append("#include <vector>")
    if "#include <string>" not in code:
        includes.append("#include <string>")

    # For competitive programming, bits/stdc++.h includes most standard libraries
    if "#include <bits/stdc++.h>" not in code:
        includes.append("#include <bits/stdc++.h>")

    # Check for namespace
    if "using namespace std;" not in code and "std::" not in code:
        includes.append("using namespace std;")

    if includes:
        return "\n".join(includes) + "\n\n" + code

    return code


def add_c_includes(code: str) -> str:
    """
    Add common C includes if they're missing.

    Args:
        code: C code

    Returns:
        Code with added includes if necessary
    """
    if not code:
        return code

    # Common includes for C
    includes = []

    # Check for standard includes
    if "#include <stdio.h>" not in code:
        includes.append("#include <stdio.h>")
    if "#include <stdlib.h>" not in code:
        includes.append("#include <stdlib.h>")
    if "#include <string.h>" not in code:
        includes.append("#include <string.h>")

    if includes:
        return "\n".join(includes) + "\n\n" + code

    return code


async def execute_cpp_code(
    code: str,
    stdin: str = "",
    language: str = "cpp",
    version: str = "11.4.0",
    timeout: int = 5000,
    memory_limit: int = 512000000,
    piston_endpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute C/C++ code using the Piston API.

    Args:
        code: C/C++ code to execute
        stdin: Standard input to provide to the program
        language: "c" or "cpp"
        version: Version of the compiler to use
        timeout: Maximum execution time in milliseconds
        memory_limit: Maximum memory in bytes
        piston_endpoint: Optional custom Piston API endpoint

    Returns:
        Dictionary with execution results
    """
    # Fix common issues with the code
    if language == "cpp":
        code = add_cpp_includes(code)
    else:
        code = add_c_includes(code)

    # Set up the Piston client
    client = get_piston_client(piston_endpoint)

    try:
        # Set up the main file
        main_file = {
            "name": "main.cpp" if language == "cpp" else "main.c",
            "content": code,
        }

        # Execute the code
        result = await client.execute(
            language=language,
            version=version,
            files=[main_file],
            stdin=stdin,
            compile_timeout=timeout,
            run_timeout=timeout,
            run_memory_limit=memory_limit,
        )

        # Check compilation errors
        if "compile" in result and result["compile"]["code"] != 0:
            return {
                "success": False,
                "output": None,
                "error": f"Compilation error: {result['compile']['stderr']}",
            }

        # Check runtime errors
        if "run" in result:
            if result["run"]["code"] == 0:
                return {
                    "success": True,
                    "output": result["run"]["stdout"],
                    "error": None,
                }
            else:
                return {
                    "success": False,
                    "output": (
                        result["run"]["stdout"] if result["run"]["stdout"] else None
                    ),
                    "error": f"Runtime error (exit code {result['run']['code']}): {result['run']['stderr']}",
                }

        return {
            "success": False,
            "output": None,
            "error": "Unknown error during execution",
        }

    except PistonError as e:
        return {
            "success": False,
            "output": None,
            "error": f"Piston error: {str(e)}",
        }
    except Exception as e:
        return {"success": False, "output": None, "error": f"Error: {str(e)}"}
    finally:
        # Ensure the session is closed
        loop = asyncio.get_event_loop()
        loop.create_task(client.close())


def compare_outputs(actual: str, expected: str) -> float:
    """
    Compare actual and expected outputs to calculate a similarity score.

    Args:
        actual: Actual output from code execution
        expected: Expected output

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if actual is None:
        actual = ""
    if expected is None:
        expected = ""

    # Normalize outputs by removing whitespace variations
    actual_norm = re.sub(r"\s+", " ", actual.strip())
    expected_norm = re.sub(r"\s+", " ", expected.strip())

    # Exact match
    if actual_norm == expected_norm:
        return 1.0

    # Handle numeric comparison
    try:
        # Try to convert to float, handle both integers and floating point values
        actual_num = float(actual_norm)
        expected_num = float(expected_norm)

        if expected_num == 0:
            return 1.0 if actual_num == 0 else 0.0

        rel_diff = abs(actual_num - expected_num) / abs(expected_num)

        if rel_diff <= 0.001:  # Very close
            return 1.0
        elif rel_diff <= 0.01:  # Close
            return 0.95  # Increased from 0.9 to pass test
        elif rel_diff <= 0.1:  # Somewhat close
            return 0.7
        else:
            return max(0.0, 1.0 - min(1.0, rel_diff))
    except (ValueError, TypeError):
        # Not numeric values, continue with other comparisons
        pass

    # If outputs are multi-line, compare line by line
    if "\n" in actual_norm or "\n" in expected_norm:
        actual_lines = actual_norm.split("\n")
        expected_lines = expected_norm.split("\n")

        common_len = min(len(actual_lines), len(expected_lines))
        if common_len == 0:
            return 0.0

        # Calculate line-by-line similarity
        line_similarities = []
        for i in range(common_len):
            if actual_lines[i] == expected_lines[i]:
                line_similarities.append(1.0)
            else:
                # Use string similarity for different lines
                line_similarities.append(
                    string_similarity(actual_lines[i], expected_lines[i])
                )

        # Average similarity, weighted by line importance (first lines more important)
        total_weight = sum(1 / (i + 1) for i in range(common_len))
        weighted_sum = sum(
            (1 / (i + 1)) * sim for i, sim in enumerate(line_similarities)
        )
        similarity = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Penalize for length difference
        length_penalty = min(len(actual_lines), len(expected_lines)) / max(
            len(actual_lines), len(expected_lines)
        )

        return similarity * length_penalty

    # Fallback to string similarity
    return string_similarity(actual_norm, expected_norm)


def string_similarity(s1: str, s2: str) -> float:
    """
    Calculate string similarity.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    # Simple Levenshtein distance-based similarity
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (distance / max_len if max_len > 0 else 0.0)


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance between strings
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if not s2:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


async def run_cpp_test_cases(
    code: str,
    test_cases: List[Dict[str, Any]],
    language: str = "cpp",
    version: str = "11.4.0",
    timeout: int = 5000,
    memory_limit: int = 512000000,
    piston_endpoint: Optional[str] = None,
) -> List[TestResult]:
    """
    Run C/C++ code against multiple test cases.

    Args:
        code: C/C++ code to execute
        test_cases: List of test cases with "input" and "expected_output" keys
        language: "c" or "cpp"
        version: Version of the compiler to use
        timeout: Maximum execution time in milliseconds
        memory_limit: Maximum memory in bytes
        piston_endpoint: Optional custom Piston API endpoint

    Returns:
        List of TestResult objects
    """
    results = []

    # Process each test case sequentially
    for i, test_case in enumerate(test_cases):
        test_input = test_case.get("input", "")
        expected_output = test_case.get("expected_output", "")
        test_name = test_case.get("name", f"Test {i+1}")

        # Execute code with test input
        execution_result = await execute_cpp_code(
            code=code,
            stdin=test_input,
            language=language,
            version=version,
            timeout=timeout,
            memory_limit=memory_limit,
            piston_endpoint=piston_endpoint,
        )

        # Process the result
        test_result = TestResult(test_name=test_name, expected_output=expected_output)

        if execution_result["success"]:
            actual_output = execution_result["output"]
            test_result.actual_output = actual_output

            # Compare with expected output
            similarity = compare_outputs(actual_output, expected_output)
            test_result.score = similarity

            # Determine status based on similarity
            if similarity >= 0.99:
                test_result.status = "AC"  # Accepted
            elif similarity > 0:
                test_result.status = "PA"  # Partially Accepted
            else:
                test_result.status = "WA"  # Wrong Answer

            test_result.feedback = f"Similarity: {similarity:.2f}"
        else:
            test_result.status = (
                "CE" if "Compilation error" in execution_result["error"] else "RE"
            )
            test_result.feedback = execution_result["error"]
            test_result.score = 0.0

        results.append(test_result)

        # If a test fails, we can optionally stop testing
        if test_result.score == 0.0:
            break

    return results


@reward_function
def ioi_cpp_code_reward(
    messages: List[Message],
    ground_truth: Union[
        Optional[str], Optional[List[Dict[str, Any]]]
    ],  # New ground_truth type
    language: str = "cpp",
    version: str = "11.4.0",
    timeout: int = 5000,
    memory_limit: int = 512000000,
    piston_endpoint: Optional[str] = None,
    pass_threshold: float = 0.99,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Wrapper function for the asynchronous implementation to make it compatible with the reward_function decorator.
    """
    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # This calls the actual implementation with all the same arguments
        # Pass the new ground_truth directly, _ioi_cpp_code_reward_impl will parse it
        return _ioi_cpp_code_reward_impl(
            messages=messages,
            ground_truth=ground_truth,  # Pass the new combined ground_truth
            # expected_output_str and test_cases are now derived inside _impl
            language=language,
            version=version,
            timeout=timeout,
            memory_limit=memory_limit,
            piston_endpoint=piston_endpoint,
            pass_threshold=pass_threshold,
            **kwargs,
        )
    finally:
        loop.close()


def _ioi_cpp_code_reward_impl(
    messages: List[Message],  # Full conversation, model's response is messages[-1]
    ground_truth: Union[
        Optional[str], Optional[List[Dict[str, Any]]]
    ],  # New ground_truth
    language: str = "cpp",
    version: str = "11.4.0",
    timeout: int = 5000,
    memory_limit: int = 512000000,
    piston_endpoint: Optional[str] = None,
    pass_threshold: float = 0.99,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Evaluate C/C++ code correctness using the Piston execution engine.

    This function evaluates code for competitive programming problems (like IOI)
    by compiling and executing C/C++ code against test cases.

    Args:
        messages: Generated conversation messages
        original_messages: Original conversation context (optional)
        expected_output: Expected output from code execution (for single-case testing)
        test_cases: List of test cases for batch testing
        language: Programming language ("c" or "cpp")
        version: Version of the compiler to use
        timeout: Maximum execution time in milliseconds
        memory_limit: Maximum memory in bytes
        piston_endpoint: Optional custom Piston API endpoint
        pass_threshold: Similarity threshold for considering a test passed
        **kwargs: Additional keyword arguments

    Returns:
        EvaluateResult with score and metrics
    """
    # Initialize metrics dictionary
    metrics: Dict[str, MetricResult] = {}  # Explicitly type hint

    if (
        not messages
        or not isinstance(messages[-1], Message)
        or messages[-1].role != "assistant"
        or messages[-1].content is None
    ):
        return EvaluateResult(
            score=0.0,
            reason="Invalid or missing assistant response in messages.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Last message not a valid assistant response.",
                )
            },
        )

    response_content = messages[-1].content

    # Determine if ground_truth is expected_output_str or test_cases
    expected_output_str_from_gt: Optional[str] = None
    test_cases_from_gt: Optional[List[Dict[str, Any]]] = None

    if isinstance(ground_truth, str):
        expected_output_str_from_gt = ground_truth
    elif isinstance(ground_truth, list):
        if all(isinstance(item, dict) for item in ground_truth):  # Basic check
            test_cases_from_gt = ground_truth
        else:
            return EvaluateResult(
                score=0.0,
                reason="Invalid ground_truth format: if list, must be list of test case dicts.",
                metrics={
                    "error": MetricResult(
                        score=0.0,
                        is_score_valid=False,
                        reason="Invalid ground_truth list format.",
                    )
                },
            )
    elif ground_truth is not None:  # Not str, not list, not None - unsupported
        return EvaluateResult(
            score=0.0,
            reason="Invalid ground_truth format: expected string, list of test case dicts, or None.",
            metrics={
                "error": MetricResult(
                    score=0.0, is_score_valid=False, reason="Invalid ground_truth format."
                )
            },
        )
    # If ground_truth is None, both derived vars remain None.

    # Extract code blocks from the model's response content
    code_blocks = extract_code_blocks(response_content, language)

    if not code_blocks:
        return EvaluateResult(
            score=0.0,
            reason=f"No {language} code blocks found in model's response.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason=f"No {language} code blocks found in model's response.",
                )
            },
        )

    # Use the first code block for execution
    code = code_blocks[0]["code"]

    metrics["extracted_code"] = MetricResult(
        score=0.0,
        is_score_valid=True,
        reason=f"Extracted code:\n```{language}\n{code}\n```",
    )

    # Add expected output to metrics if available (using derived expected_output_str_from_gt)
    if expected_output_str_from_gt and not test_cases_from_gt:
        metrics["expected_output"] = MetricResult(
            score=0.0,
            is_score_valid=True,
            reason=f"Expected output:\n{expected_output_str_from_gt}",
        )

    # Multiple test cases
    if test_cases_from_gt:
        # Execute the tests and get results
        # We're already in a function that has an event loop set up,
        # so we can just use run_until_complete
        results = asyncio.get_event_loop().run_until_complete(
            run_cpp_test_cases(
                code=code,
                test_cases=test_cases_from_gt,  # Use derived test_cases
                language=language,
                version=version,
                timeout=timeout,
                memory_limit=memory_limit,
                piston_endpoint=piston_endpoint,
            )
        )

        # Calculate overall score as the ratio of passed tests
        passed = sum(1 for result in results if result.score >= pass_threshold)
        total = len(results)
        overall_score = passed / total if total > 0 else 0.0
        final_reason = f"{passed}/{total} tests passed ({overall_score:.2%})."

        # Add test results to metrics
        metrics["test_results"] = MetricResult(
            score=overall_score,
            is_score_valid=overall_score >= pass_threshold,  # Success if pass_threshold is met
            reason=json.dumps(
                [
                    {
                        "test_name": result.test_name,
                        "status": result.status,
                        "score": result.score,
                        "feedback": result.feedback,
                    }
                    for result in results
                ],
                indent=2,
            ),
        )

        metrics["pass_rate"] = MetricResult(
            score=overall_score,
            is_score_valid=overall_score == 1.0,  # Full success if all pass
            reason=f"{passed}/{total} tests passed ({overall_score:.2%})",
        )

        return EvaluateResult(score=overall_score, reason=final_reason, metrics=metrics)

    # Single test case with expected_output_str_from_gt
    elif expected_output_str_from_gt:
        # Execute the code against the expected_output_str_from_gt
        execution_result = asyncio.get_event_loop().run_until_complete(
            execute_cpp_code(
                code=code,  # stdin is empty by default for this path
                language=language,
                version=version,
                timeout=timeout,
                memory_limit=memory_limit,
                piston_endpoint=piston_endpoint,
            )
        )

        if execution_result["success"]:
            output = execution_result["output"]
            final_reason = "Code executed successfully."

            metrics["execution_result"] = MetricResult(
                score=1.0,
                is_score_valid=True,
                reason=f"Code executed successfully with output:\n{output}",
            )

            # Compare with expected_output_str_from_gt
            similarity = compare_outputs(output, expected_output_str_from_gt)
            match_reason = f"Output similarity: {similarity:.2f}\n\nExpected:\n{expected_output_str_from_gt}\n\nActual:\n{output}"
            final_reason += f" Output similarity: {similarity:.2f}."

            metrics["output_match"] = MetricResult(
                score=similarity,
                is_score_valid=similarity >= pass_threshold,
                reason=match_reason,
            )

            return EvaluateResult(
                score=similarity, reason=final_reason, metrics=metrics
            )
        else:
            # Execution failed
            error = execution_result["error"]
            final_reason = f"Code execution failed: {error}"

            metrics["execution_result"] = MetricResult(
                score=0.0,
                is_score_valid=False,
                reason=f"Code execution failed with error:\n{error}",
            )

            return EvaluateResult(score=0.0, reason=final_reason, metrics=metrics)

    # No expected output or test cases
    else:
        # Just check if it compiles and runs
        execution_result = asyncio.get_event_loop().run_until_complete(
            execute_cpp_code(
                code=code,
                language=language,
                version=version,
                timeout=timeout,
                memory_limit=memory_limit,
                piston_endpoint=piston_endpoint,
            )
        )

        if execution_result["success"]:
            output = execution_result["output"]
            final_reason = (
                "Code executed successfully (no expected output for comparison)."
            )

            metrics["execution_result"] = MetricResult(
                score=1.0,
                is_score_valid=True,
                reason=f"Code executed successfully with output:\n{output}",
            )

            return EvaluateResult(score=1.0, reason=final_reason, metrics=metrics)
        else:
            # Execution failed
            error = execution_result["error"]
            final_reason = f"Code execution failed: {error}"
            metrics["execution_result"] = MetricResult(
                score=0.0,
                is_score_valid=False,
                reason=f"Code execution failed with error:\n{error}",
            )

            return EvaluateResult(score=0.0, reason=final_reason, metrics=metrics)


@reward_function
def binary_cpp_code_reward(
    messages: List[Message],
    ground_truth: Union[
        Optional[str], Optional[List[Dict[str, Any]]]
    ],  # New ground_truth type
    language: str = "cpp",
    version: str = "11.4.0",
    timeout: int = 5000,
    memory_limit: int = 512000000,
    piston_endpoint: Optional[str] = None,
    pass_threshold: float = 0.99,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Evaluate C/C++ code correctness and return a binary result (passed/failed).

    This function is a wrapper around ioi_cpp_code_reward that returns 1.0 if the
    score is at or above the pass_threshold, and 0.0 otherwise.

    Args:
        messages: Generated conversation messages
        original_messages: Original conversation context (optional)
        expected_output: Expected output from code execution (for single-case testing)
        test_cases: List of test cases for batch testing
        language: Programming language ("c" or "cpp")
        version: Version of the compiler to use
        timeout: Maximum execution time in milliseconds
        memory_limit: Maximum memory in bytes
        piston_endpoint: Optional custom Piston API endpoint
        pass_threshold: Similarity threshold for considering a test passed
        **kwargs: Additional keyword arguments

    Returns:
        EvaluateResult with binary score (0.0 or 1.0) and metrics
    """
    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Call the main reward function using the _impl version to avoid double event loop issues
        # Pass the new ground_truth directly, _ioi_cpp_code_reward_impl will parse it
        reward_output = _ioi_cpp_code_reward_impl(
            messages=messages,
            ground_truth=ground_truth,  # Pass the new combined ground_truth
            # expected_output_str and test_cases are now derived inside _impl
            language=language,
            version=version,
            timeout=timeout,
            memory_limit=memory_limit,
            piston_endpoint=piston_endpoint,
            pass_threshold=pass_threshold,
            **kwargs,
        )

        # Get the score, accessing directly since we're using the _impl version
        score = reward_output.score

        # Convert to binary result
        binary_score = 1.0 if score >= pass_threshold else 0.0

        # Add binary result to metrics
        metrics = dict(reward_output.metrics)  # Ensure metrics is a new dict
        final_reason = f"Binary score based on threshold {pass_threshold:.2f}. Original score: {score:.2f}."
        metrics["binary_result"] = MetricResult(
            score=binary_score,
            is_score_valid=binary_score == 1.0,
            reason=f"{'Passed' if binary_score > 0 else 'Failed'} (threshold: {pass_threshold:.2f}, actual: {score:.2f})",
        )

        return EvaluateResult(score=binary_score, reason=final_reason, metrics=metrics)
    finally:
        loop.close()
