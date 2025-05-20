"""
Code execution reward functions for evaluating code correctness.

This module provides functions to evaluate the correctness of code by:
1. Extracting code blocks from messages
2. Executing the code in a secure environment (local or E2B sandbox)
3. Comparing the output with expected results

Available reward functions:
- local_code_execution_reward: Execute code locally and evaluate correctness
- e2b_code_execution_reward: Execute code in E2B sandbox and evaluate correctness
- fractional_code_reward: Execute code and return exact pass rate
"""

import faulthandler
import json
import multiprocessing
import os
import platform
import re
import resource
import shlex  # Added for robust splitting of arguments
import signal
import subprocess
import sys
import tempfile
import traceback
from io import StringIO
from multiprocessing.managers import DictProxy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Try to import from e2b_code_interpreter first (preferred)
try:
    from e2b_code_interpreter.sync import Sandbox  # type: ignore # Use SyncSandbox

    _HAS_E2B = True
    _E2B_SOURCE = "e2b_code_interpreter"
except ImportError:
    # Fallback to e2b
    try:
        # Assuming 'e2b' package's default Sandbox is synchronous.
        # If 'e2b' also defaults to async, this part might need adjustment too.
        from e2b import Sandbox  # type: ignore

        _HAS_E2B = True
        _E2B_SOURCE = "e2b"
    except ImportError:
        _HAS_E2B = False
        _E2B_SOURCE = ""  # Use empty string instead of None

from ..models import EvaluateResult, Message, MetricResult
from ..reward_function import reward_function


def _target_func_for_execution(result_container, execute_func, args):
    try:
        result = execute_func(*args)
        result_container.update(result)
    except Exception as e:
        error_traceback = traceback.format_exc()
        result_container.update(
            {
                "success": False,
                "output": None,
                "error": f"Execution error: {str(e)}\n{error_traceback}",
            }
        )


def extract_code_blocks(
    text: str, language: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Extract code blocks from text.

    Args:
        text: The text to extract code blocks from
        language: Optional language to filter by (e.g., "python", "javascript")

    Returns:
        List of dictionaries with "code" and "language" keys
    """
    # Match code blocks with optional language specifier
    pattern = r"```(\w*)\n([\s\S]*?)\n```"
    matches = re.findall(pattern, text)

    code_blocks = []
    verbose_patterns_removed = []

    # Define patterns for verbose text that might appear inside code blocks
    # These patterns will be removed.
    # Using re.DOTALL to make '.' match newlines.
    verbose_regex_patterns = [
        re.compile(r"<think>.*?</think>", re.DOTALL),
        re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL),
        re.compile(
            r"Thinking:\s*.*?(?=\n\S)", re.DOTALL
        ),  # Matches "Thinking: ..." until a new non-whitespace line
        re.compile(r"^\s*Here's the Python code.*?\n", re.MULTILINE | re.IGNORECASE),
        re.compile(r"^\s*Okay, here is the code:.*?\n", re.MULTILINE | re.IGNORECASE),
    ]

    for lang, code_content in matches:
        # Skip if language filter is specified and doesn't match
        if language and lang and language.lower() != lang.lower():
            continue

        # Use "unknown" for empty language specifier
        detected_lang = lang.lower() if lang else "unknown"

        original_code_content = code_content  # Keep a copy for comparison
        cleaned_code_content = code_content

        for verbose_pattern in verbose_regex_patterns:
            cleaned_code_content = verbose_pattern.sub("", cleaned_code_content)

        if cleaned_code_content != original_code_content:
            verbose_patterns_removed.append(
                f"Verbose content removed from '{detected_lang}' block."
            )
            # Potentially log the actual removed content here if needed for debugging,
            # e.g., by diffing original_code_content and cleaned_code_content.

        # Add to results
        # Store both cleaned code and information about removed verbosity
        block_info = {
            "language": detected_lang,
            "code": cleaned_code_content.strip(),
        }
        if (
            verbose_patterns_removed
        ):  # Add only if something was actually removed for this block
            block_info["verbosity_cleaned_reason"] = "; ".join(verbose_patterns_removed)
            verbose_patterns_removed = []  # Reset for next block

        code_blocks.append(block_info)

    # If verbose_patterns_removed has items here, it means some patterns were found
    # outside of any specific block, or it's a general flag.
    # For now, the verbosity_cleaned_reason is per-block.

    return code_blocks


@reward_function
def local_code_execution_reward(
    messages: List[Message],  # Full conversation, last message is model's response
    ground_truth: Optional[str] = None,  # This is the new expected_output_str
    language: str = "python",
    timeout: int = 5,
    max_memory_mb: int = 100,  # Specific to local execution
    **kwargs,
) -> EvaluateResult:
    """
    Evaluate code correctness by executing it locally and comparing the output.

    This function executes code in a secure sandbox with memory limits, CPU limits,
    and timeouts to prevent malicious code from harming the system.

    Args:
        messages: List of conversation messages. The last message is assumed to be the
                  assistant's response containing the code.
        ground_truth: Expected output string from code execution. This corresponds to
                      the `expected_output_str` in the previous signature.
        language: Programming language of the code ("python", "javascript", etc.)
        timeout: Maximum execution time in seconds.
        max_memory_mb: Maximum memory usage in megabytes (default: 100).
        **kwargs: Additional keyword arguments.

    Returns:
        EvaluateResult with score and metrics.
    """
    # Initialize metrics dictionary for tracking various aspects of the execution
    metrics: Dict[str, MetricResult] = {}

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
    expected_output_str = (
        ground_truth  # Use the new ground_truth parameter as expected_output_str
    )

    # Extract code blocks from the model's response content
    code_blocks = extract_code_blocks(response_content, language)

    if not code_blocks:
        return EvaluateResult(
            score=0.0,
            reason=f"No {language} code blocks found in model's response.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    reason=f"No {language} code blocks found in model's response.",
                    is_score_valid=False,
                )
            },
        )

    # Use the first code block for execution
    code = code_blocks[0]["code"]

    metrics["extracted_code"] = MetricResult(
        score=0.0,
        reason=f"Extracted code:\n```{language}\n{code}\n```",
        is_score_valid=True,
    )

    # Add expected output to metrics if available
    if expected_output_str:
        metrics["expected_output"] = MetricResult(
            score=0.0,
            reason=f"Expected output:\n{expected_output_str}",
            is_score_valid=True,
        )

    # Execute the code based on language
    if language.lower() == "python":
        execution_result = execute_python_code(
            code, timeout
        )  # max_memory_mb is handled inside _execute_python_in_subprocess
    elif language.lower() in ["javascript", "js"]:
        execution_result = execute_javascript_code(code, timeout)
    else:
        metrics["error"] = MetricResult(
            score=0.0, reason=f"Unsupported language: {language}", is_score_valid=False
        )
        return EvaluateResult(
            score=0.0, reason=f"Unsupported language: {language}", metrics=metrics
        )

    # Check execution result
    if execution_result["success"]:
        output = execution_result["output"]

        metrics["execution_result"] = MetricResult(
            score=1.0,
            reason=f"Code executed successfully with output:\n{output}",
            is_score_valid=True,
        )

        # Compare with expected output if provided
        if expected_output_str:
            similarity = compare_outputs(output, expected_output_str)
            match_reason = f"Output similarity: {similarity:.2f}\n\nExpected:\n{expected_output_str}\n\nActual:\n{output}"

            metrics["output_match"] = MetricResult(
                score=similarity, reason=match_reason, is_score_valid=similarity == 1.0
            )
            final_reason = f"Execution successful. Output similarity: {similarity:.2f}."
            return EvaluateResult(
                score=similarity, reason=final_reason, metrics=metrics
            )

        # No expected output provided, score based on successful execution
        final_reason = "Execution successful. No expected output to compare."
        return EvaluateResult(score=1.0, reason=final_reason, metrics=metrics)
    else:
        # Execution failed
        error = execution_result["error"]

        metrics["execution_result"] = MetricResult(
            score=0.0,
            reason=f"Code execution failed with error:\n{error}",
            is_score_valid=False,
        )
        final_reason = f"Code execution failed: {error}"
        return EvaluateResult(score=0.0, reason=final_reason, metrics=metrics)


# New top-level function to be called by multiprocessing.Process
def _process_target_wrapper(
    execute_func: Callable, args: Tuple, result_container: DictProxy
):
    try:
        # Execute the code with the provided function
        result = execute_func(*args)
        result_container.update(result)
    except Exception as e:
        # traceback is imported at the top of the file
        error_traceback = traceback.format_exc()
        result_container.update(
            {
                "success": False,
                "output": None,
                "error": f"Execution error: {str(e)}\n{error_traceback}",
            }
        )


def _execute_code_in_process(
    execute_func: Callable, args: Tuple, timeout: int = 5
) -> Dict[str, Any]:
    """
    Execute code in a separate process with timeout and resource limits.

    Args:
        execute_func: Function to execute the code
        args: Arguments to pass to the execute function
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with execution results
    """
    import multiprocessing

    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    # Create and start the process using the top-level wrapper
    process = multiprocessing.Process(
        target=_process_target_wrapper, args=(execute_func, args, result_dict)
    )
    process.start()
    process.join(timeout=timeout + 0.5)

    if process.is_alive():
        process.terminate()
        process.join(0.5)
        if process.is_alive():
            process.kill()
        return {
            "success": False,
            "output": None,
            "error": f"Timeout: execution timed out after {timeout} seconds",
        }

    if not result_dict:
        return {
            "success": False,
            "output": None,
            "error": "Execution failed without producing any output",
        }

    return dict(result_dict)


def _execute_python_in_subprocess(code: str, timeout: int) -> Dict[str, Any]:
    """
    Inner function to execute Python code in a subprocess.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with execution results
    """
    try:
        # Create temporary file for the code
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file_path = temp_file.name

            # Add imports and reliability guard
            safe_code = (
                "import sys\n"
                "import os\n"
                "import signal\n"
                "import resource\n"
                "import platform\n\n"
                # Add the reliability guard code here
                "def _reliability_guard():\n"
                "    # Set memory limits\n"
                "    memory_limit = 100 * 1024 * 1024  # 100 MB\n"
                "    if platform.uname().system != 'Darwin':\n"
                "        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))\n"
                "        resource.setrlimit(resource.RLIMIT_DATA, (memory_limit, memory_limit))\n"
                "        resource.setrlimit(resource.RLIMIT_STACK, (memory_limit, memory_limit))\n"
                "    \n"
                "    # Disable harmful builtins\n"
                "    import builtins\n"
                "    builtins.exit = None\n"
                "    builtins.quit = None\n"
                "    os.environ['OMP_NUM_THREADS'] = '1'\n"
                "    # Restrict file access\n"
                "    os.system = None\n"
                "    os.popen = None\n"
                "    os.execl = None\n"
                "    os.execve = None\n"
                "    os.fork = None\n"
                "    os.remove = None\n"
                "    os.removedirs = None\n"
                "    os.rmdir = None\n"
                "    os.unlink = None\n"
                "    os.access = None\n"
                "\n"
                "_reliability_guard()\n\n"
                # User's code
                + code
            )

            temp_file.write(safe_code.encode("utf-8"))

        # Set up signal handler for timeout
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution timed out after {timeout} seconds")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            # Execute in a separate process
            process = subprocess.Popen(
                [sys.executable, temp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                # Limit resource usage
                preexec_fn=lambda: resource.setrlimit(
                    resource.RLIMIT_CPU, (timeout, timeout + 1)
                ),
            )

            stdout, stderr = process.communicate()

            # Cancel the alarm
            signal.alarm(0)

            if process.returncode == 0:
                return {
                    "success": True,
                    "output": stdout.strip(),
                    "error": None,
                }
            else:
                return {
                    "success": False,
                    "output": None,
                    "error": stderr.strip(),
                }
        except TimeoutError as e:
            # Handle timeout
            return {"success": False, "output": None, "error": str(e)}
        finally:
            # Clean up
            signal.alarm(0)
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    except Exception as e:
        error_traceback = traceback.format_exc()
        return {
            "success": False,
            "output": None,
            "error": f"Setup error: {str(e)}\n{error_traceback}",
        }


def execute_python_code(code: str, timeout: int = 5) -> Dict[str, Any]:
    """
    Execute Python code in a secure sandbox.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with execution results
    """
    # Execute the code in a separate process with timeouts and resource limits
    return _execute_code_in_process(
        _execute_python_in_subprocess, args=(code, timeout), timeout=timeout
    )


def _execute_javascript_in_subprocess(code: str, timeout: int) -> Dict[str, Any]:
    """
    Inner function to execute JavaScript code in a subprocess.

    Args:
        code: JavaScript code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with execution results
    """
    try:
        # Check if Node.js is installed
        try:
            subprocess.run(["node", "--version"], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            return {
                "success": False,
                "output": None,
                "error": "Node.js is not installed or not found in PATH",
            }

        # Create temporary file for the code
        with tempfile.NamedTemporaryFile(suffix=".js", delete=False) as temp_file:
            temp_file_path = temp_file.name

            # Add safety wrapper around the code to prevent dangerous operations
            safe_code = (
                "// Safety wrapper to prevent dangerous operations\n"
                "process.on('uncaughtException', function(err) {\n"
                "  console.error('Uncaught exception:', err.message);\n"
                "  process.exit(1);\n"
                "});\n\n"
                "// Disable dangerous functions\n"
                "process.exit = function() { console.error('exit() is disabled'); };\n"
                "process.kill = function() { console.error('kill() is disabled'); };\n"
                "const fs = require('fs');\n"
                "const originalFsReadFile = fs.readFileSync;\n"
                "const originalFsWriteFile = fs.writeFileSync;\n"
                "fs.readFileSync = function() { console.error('fs.readFileSync() is disabled'); return ''; };\n"
                "fs.writeFileSync = function() { console.error('fs.writeFileSync() is disabled'); };\n"
                "// Allow only safe require functions\n"
                "const originalRequire = require;\n"
                "global.require = function(module) {\n"
                "  const safeModules = ['assert', 'buffer', 'crypto', 'events', 'path', 'querystring',\n"
                "                      'string_decoder', 'stream', 'timers', 'url', 'util', 'zlib'];\n"
                "  if (safeModules.includes(module)) {\n"
                "    return originalRequire(module);\n"
                "  } else {\n"
                "    console.error(`Requiring module '${module}' is not allowed for security reasons`);\n"
                "    return {};\n"
                "  }\n"
                "};\n\n"
                "// User code begins here\n"
                "try {\n"
                "  " + code.replace("\n", "\n  ") + "\n"
                "} catch (error) {\n"
                "  console.error('Code execution error:', error.message);\n"
                "  process.exitCode = 1; // Set non-zero exit code to indicate failure\n"
                "}\n"
            )

            temp_file.write(safe_code.encode("utf-8"))

        # Set up signal handler for timeout
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution timed out after {timeout} seconds")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)  # Keep signal.alarm as a secondary guard

        try:
            # Execute in a separate process
            process = subprocess.Popen(
                [
                    "node",
                    "--no-warnings",
                    "--max-old-space-size=100",
                    temp_file_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()  # Ensure the process is killed
                stdout, stderr = process.communicate()  # Drain pipes
                signal.alarm(0)  # Cancel alarm if communicate timed out
                return {
                    "success": False,
                    "output": None,
                    "error": f"JavaScript execution timed out after {timeout} seconds (subprocess.TimeoutExpired). Output: {stdout.strip()}, Error: {stderr.strip()}",
                }

            # Cancel the alarm
            signal.alarm(0)

            if process.returncode == 0:
                return {
                    "success": True,
                    "output": stdout.strip(),
                    "error": None,
                }
            else:
                return {
                    "success": False,
                    "output": None,
                    "error": stderr.strip()
                    or f"JavaScript process exited with code {process.returncode}",  # Provide exit code if stderr is empty
                }
        except TimeoutError as e:  # This would be from signal.alarm
            process.kill()  # Ensure the process is killed
            _, _ = process.communicate()  # Drain pipes
            # Handle timeout from signal.alarm
            return {
                "success": False,
                "output": None,
                "error": f"JavaScript execution timed out after {timeout} seconds (signal.alarm): {str(e)}",
            }
        finally:
            # Clean up
            signal.alarm(0)  # Ensure alarm is cancelled
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        error_traceback = traceback.format_exc()
        return {
            "success": False,
            "output": None,
            "error": f"Setup error: {str(e)}\n{error_traceback}",
        }


def execute_javascript_code(code: str, timeout: int = 5) -> Dict[str, Any]:
    """
    Execute JavaScript code in a secure sandbox.

    Args:
        code: JavaScript code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with execution results
    """
    # Execute the code in a separate process with timeouts and resource limits
    return _execute_code_in_process(
        _execute_javascript_in_subprocess, args=(code, timeout), timeout=timeout
    )


def compare_outputs(actual: str, expected: str) -> float:
    """
    Compare actual and expected outputs to calculate a similarity score.

    Args:
        actual: Actual output from code execution
        expected: Expected output

    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Normalize outputs for comparison
    actual_norm = normalize_output(actual)
    expected_norm = normalize_output(expected)

    # Check for exact match after normalization
    if actual_norm == expected_norm:
        return 1.0

    # For numeric outputs, calculate relative difference
    if is_numeric(actual_norm) and is_numeric(expected_norm):
        try:
            actual_num = float(actual_norm)
            expected_num = float(expected_norm)

            if expected_num == 0:
                return 1.0 if actual_num == 0 else 0.0

            rel_diff = abs(actual_num - expected_num) / abs(expected_num)
            if rel_diff <= 0.001:  # Very close
                return 1.0
            elif rel_diff <= 0.01:  # Close
                return 0.9
            elif rel_diff <= 0.1:  # Somewhat close
                return 0.7
            else:
                return max(0.0, 1.0 - min(1.0, rel_diff))
        except (ValueError, TypeError):
            pass

    # For list/array outputs, try to parse and compare
    if (
        actual_norm.startswith("[")
        and actual_norm.endswith("]")
        and expected_norm.startswith("[")
        and expected_norm.endswith("]")
    ):
        try:
            actual_list = json.loads(actual_norm)
            expected_list = json.loads(expected_norm)

            if not actual_list and not expected_list:
                return 1.0

            if not isinstance(actual_list, list) or not isinstance(expected_list, list):
                raise ValueError("Not a list")

            # Check length similarity
            len_similarity = 1.0 - min(
                1.0,
                abs(len(actual_list) - len(expected_list))
                / max(1, max(len(actual_list), len(expected_list))),
            )

            # Check items similarity
            items_similarity = 0.0
            if len(actual_list) > 0 and len(expected_list) > 0:
                # For each item in expected, find best match in actual
                total_similarity = 0.0
                for exp_item in expected_list:
                    best_match = 0.0
                    for act_item in actual_list:
                        # Recursively compare items
                        item_similarity = compare_outputs(str(act_item), str(exp_item))
                        best_match = max(best_match, item_similarity)
                    total_similarity += best_match

                items_similarity = total_similarity / len(expected_list)

            # Combine length and items similarity
            return 0.3 * len_similarity + 0.7 * items_similarity

        except (ValueError, json.JSONDecodeError):
            pass

    # For multiline text, compare line by line
    if "\n" in actual_norm or "\n" in expected_norm:
        actual_lines = actual_norm.strip().split("\n")
        expected_lines = expected_norm.strip().split("\n")

        if not actual_lines and not expected_lines:
            return 1.0

        # Compare line count
        len_similarity = 1.0 - min(
            1.0,
            abs(len(actual_lines) - len(expected_lines))
            / max(1, max(len(actual_lines), len(expected_lines))),
        )

        # Compare line content
        lines_similarity = 0.0
        common_len = min(len(actual_lines), len(expected_lines))
        if common_len > 0:
            total_similarity = 0.0
            for i in range(common_len):
                # Use string similarity for each line
                line_similarity = string_similarity(actual_lines[i], expected_lines[i])
                total_similarity += line_similarity

            lines_similarity = total_similarity / common_len

        # Combine length and content similarity
        return 0.3 * len_similarity + 0.7 * lines_similarity

    # Fallback to string similarity
    return string_similarity(actual_norm, expected_norm)


def string_similarity(s1: str, s2: str) -> float:
    """
    Calculate string similarity using character-level comparison.

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

    # Simple scoring based on longest common subsequence
    m, n = len(s1), len(s2)
    lcs_length = longest_common_subsequence_length(s1, s2)

    return lcs_length / max(m, n)


def longest_common_subsequence_length(s1: str, s2: str) -> int:
    """
    Calculate the length of the longest common subsequence.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Length of longest common subsequence
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def normalize_output(output: str) -> str:
    """
    Normalize output for comparison.

    Args:
        output: Output string to normalize

    Returns:
        Normalized output string
    """
    # Remove leading/trailing whitespace
    normalized = output.strip()

    # Standardize line endings
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # Remove duplicate whitespace
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized


def is_numeric(value: str) -> bool:
    """
    Check if a string value represents a numeric value.

    Args:
        value: String value to check

    Returns:
        True if the value is numeric, False otherwise
    """
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def noop(*args: Any, **kwargs: Any) -> Any:
    """A no-operation function that returns None."""
    return None


def execute_code_with_e2b(
    code: str,
    language: str = "python",
    timeout: int = 30,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute code within an E2B sandbox.

    Args:
        code: Code to execute
        language: Programming language of the code ("python", "javascript", etc.)
        timeout: Maximum execution time in seconds
        api_key: Optional E2B API key (if not provided, will use E2B_API_KEY env var)

    Returns:
        Dictionary with execution results
    """
    if not _HAS_E2B:
        return {
            "success": False,
            "output": None,
            "error": "E2B package not installed. Install with: pip install e2b",
        }

    try:
        # Check for API key
        if api_key is None and os.environ.get("E2B_API_KEY") is None:
            return {
                "success": False,
                "output": None,
                "error": "API key is required for E2B execution. Set it using the api_key parameter or E2B_API_KEY environment variable.",
            }

        # Use sandbox as a context manager
        with Sandbox(api_key=api_key) as sandbox:
            # Capture stdout and stderr
            stdout = []
            stderr = []

            def capture_stdout(output):
                if hasattr(output, "line"):
                    stdout.append(output.line)
                else:
                    stdout.append(str(output))

            def capture_stderr(output):
                if hasattr(output, "line"):
                    stderr.append(output.line)
                else:
                    stderr.append(str(output))

            # Create file based on language
            if language.lower() in ["python", "py"]:
                file_path = "/code/script.py"
                cmd = "python3 /code/script.py"
            elif language.lower() in ["javascript", "js"]:
                file_path = "/code/script.js"
                cmd = "node /code/script.js"
            else:
                return {
                    "success": False,
                    "output": None,
                    "error": f"Unsupported language for E2B: {language}",
                }

            # Write code to file in sandbox
            try:
                fs_handler = None
                if _E2B_SOURCE == "e2b_code_interpreter":
                    if hasattr(sandbox, "filesystem"):
                        fs_handler = sandbox.filesystem
                elif _E2B_SOURCE == "e2b":  # Fallback for 'e2b'
                    if hasattr(
                        sandbox, "_filesystem"
                    ):  # Older 'e2b' might use _filesystem
                        fs_handler = sandbox._filesystem
                    elif hasattr(
                        sandbox, "filesystem"
                    ):  # Or it might have been updated
                        fs_handler = sandbox.filesystem

                if not fs_handler:
                    return {
                        "success": False,
                        "output": None,
                        "error": "Could not access E2B sandbox filesystem handler.",
                    }

                # Create directory if it doesn't exist
                try:
                    fs_handler.make_dir("/code")
                except Exception:
                    # Directory might already exist, or other error
                    pass  # Continue to attempt writing the file

                # Write code to file
                fs_handler.write(file_path, code)
            except Exception as e:
                return {
                    "success": False,
                    "output": None,
                    "error": f"Failed to write code to sandbox: {str(e)}",
                }

            # Execute code
            try:
                # Use the commands interface to run the code
                result = sandbox.commands.run(
                    cmd,
                    on_stdout=capture_stdout,
                    on_stderr=capture_stderr,
                    timeout=timeout,
                )

                # Combine captured output
                output = "\n".join(stdout)
                error_output = "\n".join(stderr)

                # Sandbox is automatically closed by the 'with' statement

                if result.exit_code == 0:
                    return {"success": True, "output": output, "error": None}
                else:
                    return {
                        "success": False,
                        "output": None,
                        "error": f"Process exited with code {result.exit_code}: {error_output}",
                    }

            except Exception as e:
                # Sandbox is automatically closed by the 'with' statement even on exception

                return {
                    "success": False,
                    "output": None,
                    "error": f"Execution error: {str(e)}",
                }

    except Exception as e:
        error_traceback = traceback.format_exc()
        return {
            "success": False,
            "output": None,
            "error": f"E2B setup error: {str(e)}\n{error_traceback}",
        }


@reward_function
def e2b_code_execution_reward(
    messages: List[Message],  # Full conversation, last message is model's response
    ground_truth: Optional[str] = None,  # This is the new expected_output_str
    language: str = "python",
    timeout: int = 30,
    api_key: Optional[str] = None,
    **kwargs,
) -> EvaluateResult:
    """
    Evaluate code correctness by executing it in E2B sandbox and comparing the output.

    E2B provides a secure, cloud-based sandbox for executing code safely.

    Args:
        messages: List of conversation messages. The last message is assumed to be the
                  assistant's response containing the code.
        ground_truth: Expected output string from code execution. This corresponds to
                      the `expected_output_str` in the previous signature.
        language: Programming language of the code ("python", "javascript", etc.)
        timeout: Maximum execution time in seconds.
        api_key: Optional E2B API key (if not provided, will use E2B_API_KEY env var).
        **kwargs: Additional keyword arguments.

    Returns:
        EvaluateResult with score and metrics.
    """
    if not _HAS_E2B:
        return EvaluateResult(
            score=0.0,
            reason="E2B package not installed.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    reason="E2B package not installed. Install with: pip install e2b",
                    is_score_valid=False,
                )
            },
        )

    # Check for E2B API key in environment variables if not provided
    if api_key is None and os.environ.get("E2B_API_KEY") is None:
        return EvaluateResult(
            score=0.0,
            reason="E2B API key is required.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    reason="E2B API key is required. Set the E2B_API_KEY environment variable or provide api_key parameter.",
                    is_score_valid=False,
                )
            },
        )

    # Initialize metrics dictionary for tracking various aspects of the execution
    metrics: Dict[str, MetricResult] = {}

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
    expected_output_str = (
        ground_truth  # Use the new ground_truth parameter as expected_output_str
    )

    # Extract code blocks from the model's response content
    code_blocks = extract_code_blocks(response_content, language)

    if not code_blocks:
        return EvaluateResult(
            score=0.0,
            reason=f"No {language} code blocks found in model's response.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    reason=f"No {language} code blocks found in model's response.",
                    is_score_valid=False,
                )
            },
        )

    # Use the first code block for execution
    code = code_blocks[0]["code"]

    metrics["extracted_code"] = MetricResult(
        score=0.0,
        reason=f"Extracted code:\n```{language}\n{code}\n```",
        is_score_valid=True,
    )

    # Add expected output to metrics if available
    if expected_output_str:
        metrics["expected_output"] = MetricResult(
            score=0.0,
            reason=f"Expected output:\n{expected_output_str}",
            is_score_valid=True,
        )

    # Execute the code in E2B sandbox
    execution_result = execute_code_with_e2b(
        code=code, language=language, timeout=timeout, api_key=api_key
    )

    # Check execution result
    if execution_result["success"]:
        output = execution_result["output"]

        metrics["execution_result"] = MetricResult(
            score=1.0,
            reason=f"Code executed successfully in E2B sandbox with output:\n{output}",
            is_score_valid=True,
        )

        # Compare with expected output if provided
        if expected_output_str:
            similarity = compare_outputs(output, expected_output_str)
            match_reason = f"Output similarity: {similarity:.2f}\n\nExpected:\n{expected_output_str}\n\nActual:\n{output}"

            metrics["output_match"] = MetricResult(
                score=similarity, reason=match_reason, is_score_valid=similarity == 1.0
            )
            final_reason = (
                f"E2B execution successful. Output similarity: {similarity:.2f}."
            )
            return EvaluateResult(
                score=similarity, reason=final_reason, metrics=metrics
            )

        # No expected output provided, score based on successful execution
        final_reason = "E2B execution successful. No expected output to compare."
        return EvaluateResult(score=1.0, reason=final_reason, metrics=metrics)
    else:
        # Execution failed
        error = execution_result["error"]

        metrics["execution_result"] = MetricResult(
            score=0.0,
            reason=f"Code execution failed in E2B sandbox with error:\n{error}",
            is_score_valid=False,
        )
        final_reason = f"E2B code execution failed: {error}"
        return EvaluateResult(score=0.0, reason=final_reason, metrics=metrics)


@reward_function
def fractional_code_reward(
    messages: List[Message],  # Full conversation, last message is model's response
    ground_truth: Union[
        Optional[str], Optional[List[Dict[str, Any]]]
    ],  # Expected output string OR list of test_cases
    language: str = "python",
    timeout: int = 30,
    environment: str = "local",
    api_key: Optional[str] = None,
    # function_to_call can be passed via kwargs to _run_test_cases
    **kwargs: Any,
) -> EvaluateResult:
    """
    Execute code and return the exact pass rate as a score between 0 and 1.

    Unlike the binary code reward, this function returns the actual score representing
    how closely the code output matches the expected output or how many test cases pass.

    Args:
        messages: List of conversation messages. The last message is assumed to be the
                  assistant's response containing the code.
        ground_truth: Expected output string from code execution, OR a list of test cases.
                      If a string, it's direct output comparison.
                      If a list of dicts, each dict is a test case with "input" and "expected_output".
        language: Programming language of the code ("python", "javascript", etc.).
        timeout: Maximum execution time in seconds.
        environment: Environment to run the code in ("local" or "e2b").
        api_key: Optional E2B API key (if using e2b environment).
        **kwargs: Additional keyword arguments (e.g., function_to_call for _run_test_cases).

    Returns:
        EvaluateResult with score between 0 and 1 representing the exact pass rate.
    """
    # Initialize metrics dictionary
    metrics_strings: Dict[str, str] = {}  # Store string reasons first

    if (
        not messages
        or not isinstance(messages[-1], Message)
        or messages[-1].role != "assistant"
        or messages[-1].content is None
    ):
        return EvaluateResult(
            score=0.0,
            reason="Invalid or missing assistant response in messages for fractional code reward.",
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
        # Basic check to see if it looks like a list of test cases
        if all(isinstance(item, dict) for item in ground_truth):
            test_cases_from_gt = ground_truth
        else:
            # It's a list, but not of dicts, treat as an error or unsupported ground_truth format for now
            return EvaluateResult(
                score=0.0,
                reason="Invalid ground_truth format: expected string or list of test case dicts.",
                metrics={
                    "error": MetricResult(
                        score=0.0,
                        is_score_valid=False,
                        reason="Invalid ground_truth format.",
                    )
                },
            )
    elif ground_truth is not None:  # Not str, not list, not None - unsupported
        return EvaluateResult(
            score=0.0,
            reason="Invalid ground_truth format: expected string, list of test case dicts, or None.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Invalid ground_truth format.",
                )
            },
        )
    # If ground_truth is None, both expected_output_str_from_gt and test_cases_from_gt will remain None.

    # Extract code blocks from the model's response content
    code_blocks = extract_code_blocks(response_content, language)

    if not code_blocks:
        return EvaluateResult(
            score=0.0,
            reason=f"No {language} code blocks found in model's response for fractional code reward.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    reason=f"No {language} code blocks found in model's response.",
                    is_score_valid=False,
                )
            },
        )

    # Use the first code block for execution
    code = code_blocks[0]["code"]

    metrics_strings["extracted_code"] = f"Extracted code:\n```{language}\n{code}\n```"

    # Add expected output to metrics if available and not using test cases
    if expected_output_str_from_gt and not test_cases_from_gt:
        metrics_strings["expected_output"] = (
            f"Expected output:\n{expected_output_str_from_gt}"
        )

    # Handle multiple test cases if provided
    if test_cases_from_gt:
        # Pass kwargs through to _run_test_cases, which might include function_to_call
        return _run_test_cases(
            code=code,
            language=language,
            test_cases=test_cases_from_gt,  # Use derived test_cases
            timeout=timeout,
            environment=environment,
            api_key=api_key,
            **kwargs,
        )

    # This part handles single expected_output_str if test_cases_from_gt are not provided
    # (i.e., ground_truth was a string or None)
    execution_result: Dict[str, Any]
    if environment.lower() == "e2b":
        if not _HAS_E2B:
            return EvaluateResult(
                score=0.0,
                reason="E2B package not installed for fractional code reward.",
                metrics={
                    "error": MetricResult(
                        score=0.0,
                        reason="E2B package not installed. Install with: pip install e2b",
                        is_score_valid=False,
                    )
                },
            )
        execution_result = execute_code_with_e2b(
            code=code, language=language, timeout=timeout, api_key=api_key
        )
    else:  # local execution
        if language.lower() == "python":
            execution_result = execute_python_code(code, timeout)
        elif language.lower() in ["javascript", "js"]:
            execution_result = execute_javascript_code(code, timeout)
        else:
            # Convert string metrics to MetricResult objects before returning
            final_metrics_on_error: Dict[str, MetricResult] = {
                k: MetricResult(
                    score=0.0, reason=v, is_score_valid=(k == "extracted_code")
                )
                for k, v in metrics_strings.items()
            }
            final_metrics_on_error["error"] = MetricResult(
                score=0.0,
                reason=f"Unsupported language: {language}",
                is_score_valid=False,
            )
            return EvaluateResult(
                score=0.0,
                reason=f"Unsupported language for fractional code reward: {language}",
                metrics=final_metrics_on_error,
            )

    # Convert initial string metrics to MetricResult objects
    metric_results: Dict[str, MetricResult] = {
        k: MetricResult(
            score=0.0,
            reason=v,
            is_score_valid=(
                k == "extracted_code"
                or (k == "expected_output" and expected_output_str_from_gt is not None)
            ),
        )
        for k, v in metrics_strings.items()
    }

    if execution_result["success"]:
        output = execution_result["output"]
        metric_results["execution_result"] = MetricResult(
            score=1.0,
            reason=f"Code executed successfully with output:\n{output}",
            is_score_valid=True,
        )

        if (
            expected_output_str_from_gt
        ):  # Only compare if expected_output_str_from_gt is available
            similarity = compare_outputs(output, expected_output_str_from_gt)
            match_reason = f"Output similarity: {similarity:.2f}\n\nExpected:\n{expected_output_str_from_gt}\n\nActual:\n{output}"
            metric_results["output_match"] = MetricResult(
                score=similarity, reason=match_reason, is_score_valid=similarity == 1.0
            )
            final_reason = f"Fractional code execution successful. Output similarity: {similarity:.2f}."
            return EvaluateResult(
                score=similarity, reason=final_reason, metrics=metric_results
            )
        else:  # Successful execution, but no expected_output_str_from_gt to compare against
            final_reason = "Fractional code execution successful. No expected output string to compare."
            return EvaluateResult(
                score=1.0, reason=final_reason, metrics=metric_results
            )  # Score 1.0 for successful execution if no expected output
    else:
        # Execution failed
        error = execution_result["error"]
        metric_results["execution_result"] = MetricResult(
            score=0.0,
            reason=f"Code execution failed with error:\n{error}",
            is_score_valid=False,
        )
        final_reason = f"Fractional code execution failed: {error}"
        return EvaluateResult(score=0.0, reason=final_reason, metrics=metric_results)


def _run_test_cases(
    code: str,
    language: str,
    test_cases: List[Dict[str, Any]],
    timeout: int,
    environment: str,
    api_key: Optional[str] = None,
    function_to_call: Optional[str] = None,
    prompt_for_name_extraction: Optional[
        str
    ] = None,  # Not used yet, but for future use
    **kwargs: Any,  # Keep kwargs for flexibility, though function_to_call is now explicit
) -> EvaluateResult:  # Changed return type hint to match actual returns
    """
    Run code against multiple test cases and return the fraction of passing tests.
    Can optionally call a specific function if `function_to_call` is provided.

    Args:
        code: The code to execute
        language: Programming language of the code
        test_cases: List of test cases with input and expected output
        timeout: Maximum execution time in seconds
        environment: Environment to run the code in ("local" or "e2b")
        api_key: Optional E2B API key (if using e2b environment)

    Returns:
        EvaluateResult with score representing the fraction of passing tests
    """
    # --- Start of Reverted Code ---
    metrics = {}
    results = []
    passed = 0
    total = len(test_cases)

    if total == 0:
        return EvaluateResult(
            score=0.0,
            reason="No test cases provided",
            metrics={
                "error": MetricResult(  # Changed
                    score=0.0, reason="No test cases provided", is_score_valid=False
                )
            },
        )

    # Prepare the code wrapper based on language and whether a function name is provided
    if language.lower() in ["python", "py"]:
        if function_to_call:
            # Mode 1: Function Call Harness (Python)
            def prepare_test_code(
                user_code: str, test_input_str: str, func_name: Optional[str]
            ) -> str:
                import ast  # For ast.literal_eval
                import json  # For json.loads and json.JSONDecodeError

                # Helper function to refine evaluated values
                def refine_evaluated_value(val: Any) -> Any:
                    # If val is a string, try to parse it further if it looks like list/dict or number
                    if isinstance(val, str):
                        stripped_val = val.strip()
                        if stripped_val.startswith(("[", "{")):  # Looks like list/dict
                            try:
                                return json.loads(stripped_val)
                            except json.JSONDecodeError:
                                return val  # Keep original string if json.loads fails
                        else:  # Not list-like or dict-like string
                            try:  # Try to convert to number
                                if (
                                    "." in stripped_val
                                    or "e" in stripped_val.lower()
                                    or "E" in stripped_val
                                ):
                                    return float(stripped_val)
                                else:
                                    return int(stripped_val)
                            except ValueError:
                                return val  # Keep original string if not a number
                    return val  # Not a string, or already refined (e.g. actual list/int from initial parse)

                # Argument parsing logic:
                # The goal is to convert the test_input_str into a list of Python objects
                # that will be used as arguments to the target function.
                # This involves a multi-stage approach to handle various input string formats:
                # 1. Try to parse the whole input string as a single JSON entity.
                # 2. If that fails, try to parse the whole input string as a single Python literal.
                # 3. If both fail, split the string by spaces and parse each part as a Python literal.
                # The `refine_evaluated_value` helper is used to further process parsed strings
                # into more specific types (list, dict, int, float) if applicable.
                parsed_args = []
                args_str_stripped = test_input_str.strip()

                if not args_str_stripped:
                    # No arguments if the input string is empty.
                    pass
                else:
                    parsed_as_single_arg = False
                    # Attempt 1: Parse the entire string as a single JSON entity.
                    # Handles valid JSON values like "[1,2,3]", "5", "\"a string\"", "true".
                    try:
                        val_from_json = json.loads(args_str_stripped)
                        parsed_args.append(refine_evaluated_value(val_from_json))
                        parsed_as_single_arg = True
                    except json.JSONDecodeError:
                        # Attempt 2: If JSON parsing fails, try parsing as a single Python literal.
                        # Handles Python literals like "['a','b']", "{'k':'v'}", "None".
                        try:
                            val_from_ast = ast.literal_eval(args_str_stripped)
                            parsed_args.append(refine_evaluated_value(val_from_ast))
                            parsed_as_single_arg = True
                        except (ValueError, SyntaxError):
                            # Both single-argument parse attempts failed.
                            # Proceed to split by space for multiple arguments.
                            pass  # Fall through to space-splitting logic

                    if not parsed_as_single_arg:
                        # Attempt 3: Fallback - treat as space-separated arguments, respecting quotes.
                        # Each part is treated as a Python literal.
                        # Handles inputs like "1 'foo' \"[1, 2]\"" or "item1 item2".
                        try:
                            arg_parts = shlex.split(args_str_stripped)
                        except ValueError:  # Handle shlex errors e.g. unmatched quotes
                            arg_parts = [
                                args_str_stripped
                            ]  # Fallback to treating as a single, possibly problematic, part

                        for part_str in arg_parts:
                            try:
                                # Try to evaluate the part as a Python literal
                                val_from_part_ast = ast.literal_eval(part_str)
                                parsed_args.append(
                                    refine_evaluated_value(val_from_part_ast)
                                )
                            except (ValueError, SyntaxError):
                                # If ast.literal_eval fails on a part, it's likely an unquoted string
                                # or a malformed literal. Treat it as a string, but still refine.
                                # (e.g. "item" or even "[1,2,3" if it ended up here due to earlier parse failures)
                                parsed_args.append(refine_evaluated_value(part_str))

                # Create the final argument string for the function call template.
                args_repr = ", ".join(map(repr, parsed_args))

                # Use triple quotes for the multi-line string
                return f"""import sys
import json
import traceback

# User code (contains function definition)
{user_code}

# Call the function with parsed arguments and print repr(result)
try:
    result = {func_name}({args_repr})
    print(repr(result))
except Exception as e:
    import traceback # Ensure traceback is imported
    print(f'Error calling function {func_name}: {{traceback.format_exc()}}', file=sys.stderr)
    import sys # Ensure sys is imported here
    sys.exit(1) # Exit non-zero
"""

        else:
            # Mode 2: Stdin/Stdout Harness (Python) - Fallback
            # Make sure the signature is identical to the function defined in if branch
            def prepare_test_code(
                user_code: str, test_input_str: str, func_name: Optional[str]
            ) -> str:
                escaped_test_input = json.dumps(test_input_str)[1:-1].replace(
                    "'''", "'\\''\\''\\''"
                )
                # Use triple quotes here too for consistency
                return f"""import sys
from io import StringIO

original_stdout = sys.stdout
sys.stdout = captured_stdout = StringIO()
sys.stdin = StringIO('''{escaped_test_input}''')

try:
    exec({repr(user_code)}) # Execute user code as a script
except Exception as e:
    import traceback
    print(f'Error executing script: {{traceback.format_exc()}}', file=sys.stderr)
    import sys # Ensure sys is imported here
    sys.exit(1) # Exit non-zero

sys.stdout = original_stdout
print(captured_stdout.getvalue(), end='')
"""

    elif language.lower() in ["javascript", "js"]:
        if function_to_call:
            # Mode 1: Function Call Harness (JavaScript) - Basic implementation
            def prepare_test_code(
                user_code: str, test_input_str: str, func_name: Optional[str]
            ) -> str:
                # Basic input parsing for JS (similar to Python, less robust)
                args_str = test_input_str.strip()
                parsed_args_js = []
                if args_str:
                    for arg in args_str.split():
                        if arg.isdigit() or (arg.startswith("-") and arg[1:].isdigit()):
                            parsed_args_js.append(arg)  # Keep as string number
                        elif "." in arg and all(
                            c.isdigit() or c == "." or (i == 0 and c == "-")
                            for i, c in enumerate(arg)
                        ):
                            try:
                                float(arg)  # Check if it's a float
                                parsed_args_js.append(arg)  # Keep as string number
                            except ValueError:
                                parsed_args_js.append(json.dumps(arg))  # String literal
                        else:
                            parsed_args_js.append(json.dumps(arg))  # String literal

                args_js_repr = ", ".join(parsed_args_js)
                # Use triple quotes for JS harness string
                return f"""// User code (contains function definition)
{user_code}

// Call the function and print result
try {{
    const result = {func_name}({args_js_repr});
    // Use JSON.stringify for more robust output representation
    console.log(JSON.stringify(result));
}} catch (error) {{
    console.error(`Error calling function {func_name}:`, error);
    process.exitCode = 1; // Indicate error
}}
"""

        else:
            # Mode 2: Stdin/Stdout Harness (JavaScript) - Fallback
            def prepare_test_code(
                user_code: str, test_input_str: str, func_name: Optional[str]
            ) -> str:
                input_lines = test_input_str.strip().split("\n")
                input_setup = "const inputs = " + json.dumps(input_lines) + ";\n"
                input_setup += "let inputIndex = 0;\n"
                input_setup += "const readline = () => inputs[inputIndex++];\n"
                # Use triple quotes for JS harness string
                return f"""// Capture console.log output
const originalLog = console.log;
let output = '';
console.log = function(...args) {{ // Capture multiple args
  output += args.map(String).join(' ') + '\\n';
}};

{input_setup}

// User code
try {{
    {user_code} // Execute user code as script
}} catch (error) {{
    console.error('Error executing script:', error);
    process.exitCode = 1; // Indicate error
}}

// Print captured output
console.log = originalLog;
process.stdout.write(output); // Write directly to avoid extra newline
"""

    else:
        # This case should be caught by the prepare_test_code logic or earlier language checks
        # However, to be safe, return an error EvaluateResult
        return EvaluateResult(
            score=0.0,
            reason=f"Unsupported language for test cases: {language}",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    reason=f"Unsupported language for test cases: {language}",
                    is_score_valid=False,
                )
            },
        )

    # Process each test case
    for i, test_case in enumerate(test_cases):
        test_input = test_case.get("input", "")
        expected = test_case.get("expected_output", "")

        # Prepare code with test input using the appropriate harness
        test_code_prepared = prepare_test_code(code, test_input, function_to_call)

        # Execute code in the specified environment
        if environment.lower() == "e2b":
            if not _HAS_E2B:
                return EvaluateResult(  # Changed
                    score=0.0,
                    reason="E2B package not installed for test cases.",  # Added reason
                    metrics={
                        "error": MetricResult(  # Changed
                            score=0.0,
                            reason="E2B package not installed. Install with: pip install e2b",
                            is_score_valid=False,
                        )
                    },
                )

            # Always execute the prepared code harness
            execution_result = execute_code_with_e2b(
                code=test_code_prepared,
                language=language,
                timeout=timeout,
                api_key=api_key,
            )
        else:  # local execution
            if language.lower() in ["python", "py"]:
                execution_result = execute_python_code(test_code_prepared, timeout)
            elif language.lower() in ["javascript", "js"]:
                execution_result = execute_javascript_code(test_code_prepared, timeout)
            # Need to handle the case where language is not supported here too
            else:
                return EvaluateResult(
                    score=0.0,
                    reason=f"Unsupported language for local execution: {language}",
                    metrics={
                        "error": MetricResult(
                            score=0.0,
                            reason=f"Unsupported language for local execution: {language}",
                            is_score_valid=False,
                        )
                    },
                )  # Changed

        # Process the result
        test_result = {
            "test_number": i + 1,
            "input": test_input,
            "expected_output": expected,
            "passed": False,
            "details": "",
        }

        if execution_result["success"]:
            output = execution_result["output"]
            # Use exact match after normalization for DeepCoder style pass/fail
            # Normalize both actual and expected outputs
            normalized_output = normalize_output(output)
            normalized_expected = normalize_output(expected)

            # For function call mode, expected output might be repr() format
            # Try to match repr() format if function_to_call was used
            expected_repr = (
                repr(expected)
                if function_to_call and language.lower() in ["python", "py"]
                else None
            )
            normalized_expected_repr = (
                normalize_output(expected_repr) if expected_repr else None
            )

            # Check for exact match against normalized expected or its repr() form
            is_pass = normalized_output == normalized_expected
            if not is_pass and normalized_expected_repr:
                is_pass = normalized_output == normalized_expected_repr

            test_result["passed"] = is_pass
            test_result["actual_output"] = output  # Store raw output
            test_result["normalized_actual"] = normalized_output
            test_result["normalized_expected"] = normalized_expected
            test_result["details"] = f"Passed: {is_pass}"

            if test_result["passed"]:
                passed += 1
        else:
            test_result["error"] = execution_result["error"]
            test_result["details"] = f"Error: {execution_result['error']}"

        results.append(test_result)

    # Calculate the final score as the fraction of passing tests
    score = passed / total if total > 0 else 0.0

    # Ensure results is treated as a list of dicts
    if isinstance(results, list):
        metrics["test_results"] = results  # results is List[Dict], not MetricResult
    else:
        # If somehow results is a string, convert it to a list with one dict
        metrics["test_results"] = [{"error": "Invalid results format"}]
    metrics["pass_rate"] = f"{passed}/{total} tests passed ({score:.2%})"

    # Convert metrics to MetricResult objects
    final_metrics: Dict[str, MetricResult] = {}
    for key, value in metrics.items():
        if key == "test_results":
            final_metrics[key] = MetricResult(
                score=score,  # Use overall score for this summary metric
                reason=json.dumps(value, indent=2),  # Serialize list of dicts
                is_score_valid=score == 1.0,
            )
        elif key == "pass_rate":
            final_metrics[key] = MetricResult(
                score=score,  # Use overall score
                reason=str(value),
                is_score_valid=score == 1.0,
            )
        elif isinstance(
            value, MetricResult
        ):  # Should not happen here as metrics are strings or lists
            final_metrics[key] = value
        elif isinstance(value, str):  # Should not happen here anymore
            final_metrics[key] = MetricResult(
                score=0.0, reason=value, is_score_valid=False
            )

    return EvaluateResult(
        score=score, reason=f"{passed}/{total} tests passed.", metrics=final_metrics
    )


def reliability_guard(maximum_memory_bytes: Optional[int] = None) -> None:
    """
    Disable various destructive functions and prevent the generated code
    from interfering with the test system.

    This sets resource limits and disables various system calls that could
    be used to interfere with the testing environment.

    Args:
        maximum_memory_bytes: Maximum memory allocation allowed in bytes (optional)

    Warning:
        This function is NOT a security sandbox. Untrusted code should not be
        blindly executed outside of a proper sandbox environment.
    """
    # Set memory limits if specified
    if maximum_memory_bytes is not None:
        if platform.uname().system != "Darwin":  # not MacOS
            resource.setrlimit(
                resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
            )
            resource.setrlimit(
                resource.RLIMIT_DATA,
                (maximum_memory_bytes, maximum_memory_bytes),
            )
            resource.setrlimit(
                resource.RLIMIT_STACK,
                (maximum_memory_bytes, maximum_memory_bytes),
            )

    # Disable faulthandler to avoid unwanted crash dumps
    faulthandler.disable()

    # Type ignores are needed because we're deliberately breaking type safety
    # to prevent dangerous operations in child processes

    # Disable destructive functions in builtins
    import builtins

    builtins.exit = noop  # type: ignore
    builtins.quit = noop  # type: ignore

    # Disable threading/parallelism for resource control
    os.environ["OMP_NUM_THREADS"] = "1"

    # Instead of completely nullifying functions, we'll replace them with noop
    # This preserves the callable interface while making them do nothing
    os.kill = noop  # type: ignore
    os.system = noop  # type: ignore
    os.putenv = noop  # type: ignore
    os.remove = noop  # type: ignore
    os.removedirs = noop  # type: ignore
    os.rmdir = noop  # type: ignore
    os.fchdir = noop  # type: ignore
    os.setuid = noop  # type: ignore
    os.fork = noop  # type: ignore
    os.forkpty = noop  # type: ignore
    os.killpg = noop  # type: ignore
    os.rename = noop  # type: ignore
    os.renames = noop  # type: ignore
    os.truncate = noop  # type: ignore
    os.replace = noop  # type: ignore
    os.unlink = noop  # type: ignore
    os.fchmod = noop  # type: ignore
    os.fchown = noop  # type: ignore
    os.chmod = noop  # type: ignore
    os.chown = noop  # type: ignore
    os.chroot = noop  # type: ignore

    # Only disable these if they exist
    if hasattr(os, "lchflags"):
        os.lchflags = noop  # type: ignore
    if hasattr(os, "lchmod"):
        os.lchmod = noop  # type: ignore
    if hasattr(os, "lchown"):
        os.lchown = noop  # type: ignore

    # These are read-only functions that we'll keep as is
    # os.getcwd = noop  # type: ignore
    # os.chdir = noop  # type: ignore

    # Disable shutil functions
    import shutil

    shutil.rmtree = noop  # type: ignore
    shutil.move = noop  # type: ignore
    shutil.chown = noop  # type: ignore

    # We don't disable subprocess completely because we need it for our own code
    # but we could disable it in the sandboxed environment

    # Create empty modules for potentially dangerous imports
    class EmptyModule:
        def __getattr__(self, name: str) -> Any:
            return noop

    # Disable dangerous modules
    for mod_name in ["ipdb", "joblib", "psutil", "tkinter"]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = EmptyModule()  # type: ignore


class Capturing(list):
    """
    Context manager for capturing stdout output.

    This class captures all output to stdout and stores it in a list,
    allowing for the examination of output from executed code.
    """

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: None
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout
