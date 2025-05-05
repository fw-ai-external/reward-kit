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

import os
import re
import sys
import json
import signal
import platform
import tempfile
import resource
import subprocess
import faulthandler
import multiprocessing
import traceback
from io import StringIO
from typing import Dict, List, Any, Optional, Tuple, Callable, Union

# Try to import from e2b_code_interpreter first (preferred)
try:
    from e2b_code_interpreter import Sandbox  # type: ignore
    _HAS_E2B = True
    _E2B_SOURCE = "e2b_code_interpreter"
except ImportError:
    # Fallback to e2b
    try:
        from e2b import Sandbox  # type: ignore
        _HAS_E2B = True
        _E2B_SOURCE = "e2b"
    except ImportError:
        _HAS_E2B = False
        _E2B_SOURCE = None

from ..models import RewardOutput, MetricRewardOutput, Message, EvaluateResult, MetricResult
from ..reward_function import reward_function


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
    for lang, code in matches:
        # Skip if language filter is specified and doesn't match
        if language and lang and language.lower() != lang.lower():
            continue

        # Use "unknown" for empty language specifier
        detected_lang = lang.lower() if lang else "unknown"

        # Add to results
        code_blocks.append({"language": detected_lang, "code": code.strip()})

    return code_blocks


def local_code_execution_reward(
    messages: List[Dict[str, str]],
    original_messages: Optional[List[Dict[str, str]]] = None,
    expected_output: Optional[str] = None,
    language: str = "python",
    timeout: int = 5,
    max_memory_mb: int = 100,
    **kwargs,
) -> RewardOutput:
    """
    Evaluate code correctness by executing it locally and comparing the output.

    This function executes code in a secure sandbox with memory limits, CPU limits,
    and timeouts to prevent malicious code from harming the system.

    Args:
        messages: Generated conversation messages
        original_messages: Original conversation context (optional)
        expected_output: Expected output from code execution
        language: Programming language of the code ("python", "javascript", etc.)
        timeout: Maximum execution time in seconds
        max_memory_mb: Maximum memory usage in megabytes (default: 100)
        **kwargs: Additional keyword arguments

    Returns:
        RewardOutput with score and metrics
    """
    # Initialize metrics dictionary for tracking various aspects of the execution
    metrics = {}

    # Note: We don't set the reliability guard in the main process
    # as it would interfere with the test runner and other system processes.
    # Instead, the guard is applied in each subprocess during code execution.

    # Extract the last assistant message
    if not messages:
        return RewardOutput(
            score=0.0,
            metrics={
                "error": MetricRewardOutput(score=0.0, reason="No messages provided")
            },
        )

    last_message = messages[-1]

    # Check role of the last message
    if last_message.get("role") != "assistant":
        return RewardOutput(
            score=0.0,
            metrics={
                "error": MetricRewardOutput(
                    score=0.0, reason="Last message is not from assistant"
                )
            },
        )

    # Extract code blocks from the message
    code_blocks = extract_code_blocks(last_message.get("content", ""), language)

    if not code_blocks:
        return RewardOutput(
            score=0.0,
            metrics={
                "error": MetricRewardOutput(
                    score=0.0, reason=f"No {language} code blocks found in message"
                )
            },
        )

    # Extract expected output if not provided directly
    if expected_output is None and original_messages:
        # Try to find expected output in the original messages
        for msg in original_messages:
            if msg.get("role") == "user":
                # Look for expected output patterns like "Expected output:" or "Output:"
                content = msg.get("content", "")
                output_patterns = [
                    r"Expected output:?\s*([\s\S]+)",
                    r"Output:?\s*([\s\S]+)",
                    r"Result:?\s*([\s\S]+)",
                    r"Should (output|return|print):?\s*([\s\S]+)",
                ]

                for pattern in output_patterns:
                    match = re.search(pattern, content)
                    if match:
                        # Use group 1 or 2 depending on the pattern
                        expected_output = (
                            match.group(2)
                            if len(match.groups()) > 1 and match.group(2)
                            else match.group(1)
                        )
                        expected_output = expected_output.strip()
                        break

                if expected_output:
                    break

    # Use the first code block for execution
    code = code_blocks[0]["code"]

    # Log the extracted code
    metrics["extracted_code"] = MetricRewardOutput(
        score=0.0,  # Not a real score
        reason=f"Extracted code:\n```{language}\n{code}\n```",
    )

    # Add expected output to metrics if available
    if expected_output:
        metrics["expected_output"] = MetricRewardOutput(
            score=0.0,  # Not a real score
            reason=f"Expected output:\n{expected_output}",
        )

    # Execute the code based on language
    if language.lower() == "python":
        execution_result = execute_python_code(code, timeout)
    elif language.lower() in ["javascript", "js"]:
        execution_result = execute_javascript_code(code, timeout)
    else:
        return RewardOutput(
            score=0.0,
            metrics={
                **metrics,
                "error": MetricRewardOutput(
                    score=0.0, reason=f"Unsupported language: {language}"
                ),
            },
        )

    # Check execution result
    if execution_result["success"]:
        output = execution_result["output"]

        metrics["execution_result"] = MetricRewardOutput(
            score=1.0, reason=f"Code executed successfully with output:\n{output}"
        )

        # Compare with expected output if provided
        if expected_output:
            similarity = compare_outputs(output, expected_output)
            match_reason = f"Output similarity: {similarity:.2f}\n\nExpected:\n{expected_output}\n\nActual:\n{output}"

            metrics["output_match"] = MetricRewardOutput(
                score=similarity, reason=match_reason
            )

            return RewardOutput(score=similarity, metrics=metrics)

        # No expected output provided, score based on successful execution
        return RewardOutput(score=1.0, metrics=metrics)
    else:
        # Execution failed
        error = execution_result["error"]

        metrics["execution_result"] = MetricRewardOutput(
            score=0.0, reason=f"Code execution failed with error:\n{error}"
        )

        return RewardOutput(score=0.0, metrics=metrics)


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
    # Use multiprocessing to isolate the execution
    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    def target_func(result_container):
        try:
            # Execute the code with the provided function
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

    # Create and start the process
    process = multiprocessing.Process(target=target_func, args=(result_dict,))
    process.start()
    process.join(timeout=timeout + 0.5)  # Add a small buffer to the timeout

    # If the process is still running, terminate it
    if process.is_alive():
        process.terminate()
        process.join(0.5)  # Give it a chance to terminate gracefully
        if process.is_alive():
            process.kill()  # Force kill if still running
        return {
            "success": False,
            "output": None,
            "error": f"Timeout: execution timed out after {timeout} seconds",
        }

    # If process died without updating result_dict
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
                return {"success": True, "output": stdout.strip(), "error": None}
            else:
                return {"success": False, "output": None, "error": stderr.strip()}
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
        signal.alarm(timeout)

        try:
            # Execute in a separate process
            process = subprocess.Popen(
                ["node", "--no-warnings", "--max-old-space-size=100", temp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            stdout, stderr = process.communicate()

            # Cancel the alarm
            signal.alarm(0)

            if process.returncode == 0:
                return {"success": True, "output": stdout.strip(), "error": None}
            else:
                return {"success": False, "output": None, "error": stderr.strip()}
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

        # Initialize sandbox with the provided API key or use environment variable
        sandbox = Sandbox(api_key=api_key)

        # Capture stdout and stderr
        stdout = []
        stderr = []

        def capture_stdout(output):
            if hasattr(output, 'line'):
                stdout.append(output.line)
            else:
                stdout.append(str(output))

        def capture_stderr(output):
            if hasattr(output, 'line'):
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
            # Create directory if it doesn't exist
            try:
                sandbox.files.mkdir("/code")
            except Exception:
                # Directory might already exist, ignore error
                pass
            
            # Write code to file
            sandbox.files.write(file_path, code)
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
                timeout=timeout
            )

            # Combine captured output
            output = "\n".join(stdout)
            error_output = "\n".join(stderr)

            # Clean up sandbox
            try:
                sandbox.close()
            except Exception:
                pass

            if result.exit_code == 0:
                return {"success": True, "output": output, "error": None}
            else:
                return {
                    "success": False,
                    "output": None,
                    "error": f"Process exited with code {result.exit_code}: {error_output}",
                }

        except Exception as e:
            # Ensure sandbox is closed even if execution fails
            try:
                sandbox.close()
            except Exception:
                pass

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


def e2b_code_execution_reward(
    messages: List[Dict[str, str]],
    original_messages: Optional[List[Dict[str, str]]] = None,
    expected_output: Optional[str] = None,
    language: str = "python",
    timeout: int = 30,
    api_key: Optional[str] = None,
    **kwargs,
) -> RewardOutput:
    """
    Evaluate code correctness by executing it in E2B sandbox and comparing the output.

    E2B provides a secure, cloud-based sandbox for executing code safely.

    Args:
        messages: Generated conversation messages
        original_messages: Original conversation context (optional)
        expected_output: Expected output from code execution
        language: Programming language of the code ("python", "javascript", etc.)
        timeout: Maximum execution time in seconds
        api_key: Optional E2B API key (if not provided, will use E2B_API_KEY env var)
        **kwargs: Additional keyword arguments

    Returns:
        RewardOutput with score and metrics
    """
    if not _HAS_E2B:
        return RewardOutput(
            score=0.0,
            metrics={
                "error": MetricRewardOutput(
                    score=0.0,
                    reason="E2B package not installed. Install with: pip install e2b",
                )
            },
        )

    # Check for E2B API key in environment variables if not provided
    if api_key is None and os.environ.get("E2B_API_KEY") is None:
        return RewardOutput(
            score=0.0,
            metrics={
                "error": MetricRewardOutput(
                    score=0.0,
                    reason="E2B API key is required. Set the E2B_API_KEY environment variable or provide api_key parameter.",
                )
            },
        )

    # Initialize metrics dictionary for tracking various aspects of the execution
    metrics = {}

    # Extract the last assistant message
    if not messages:
        return RewardOutput(
            score=0.0,
            metrics={
                "error": MetricRewardOutput(score=0.0, reason="No messages provided")
            },
        )

    last_message = messages[-1]

    # Check role of the last message
    if last_message.get("role") != "assistant":
        return RewardOutput(
            score=0.0,
            metrics={
                "error": MetricRewardOutput(
                    score=0.0, reason="Last message is not from assistant"
                )
            },
        )

    # Extract code blocks from the message
    code_blocks = extract_code_blocks(last_message.get("content", ""), language)

    if not code_blocks:
        return RewardOutput(
            score=0.0,
            metrics={
                "error": MetricRewardOutput(
                    score=0.0, reason=f"No {language} code blocks found in message"
                )
            },
        )

    # Extract expected output if not provided directly
    if expected_output is None and original_messages:
        # Try to find expected output in the original messages
        for msg in original_messages:
            if msg.get("role") == "user":
                # Look for expected output patterns like "Expected output:" or "Output:"
                content = msg.get("content", "")
                output_patterns = [
                    r"Expected output:?\s*([\s\S]+)",
                    r"Output:?\s*([\s\S]+)",
                    r"Result:?\s*([\s\S]+)",
                    r"Should (output|return|print):?\s*([\s\S]+)",
                ]

                for pattern in output_patterns:
                    match = re.search(pattern, content)
                    if match:
                        # Use group 1 or 2 depending on the pattern
                        expected_output = (
                            match.group(2)
                            if len(match.groups()) > 1 and match.group(2)
                            else match.group(1)
                        )
                        expected_output = expected_output.strip()
                        break

                if expected_output:
                    break

    # Use the first code block for execution
    code = code_blocks[0]["code"]

    # Log the extracted code
    metrics["extracted_code"] = MetricRewardOutput(
        score=0.0,  # Not a real score
        reason=f"Extracted code:\n```{language}\n{code}\n```",
    )

    # Add expected output to metrics if available
    if expected_output:
        metrics["expected_output"] = MetricRewardOutput(
            score=0.0,  # Not a real score
            reason=f"Expected output:\n{expected_output}",
        )

    # Execute the code in E2B sandbox
    execution_result = execute_code_with_e2b(
        code=code, language=language, timeout=timeout, api_key=api_key
    )

    # Check execution result
    if execution_result["success"]:
        output = execution_result["output"]

        metrics["execution_result"] = MetricRewardOutput(
            score=1.0,
            reason=f"Code executed successfully in E2B sandbox with output:\n{output}",
        )

        # Compare with expected output if provided
        if expected_output:
            similarity = compare_outputs(output, expected_output)
            match_reason = f"Output similarity: {similarity:.2f}\n\nExpected:\n{expected_output}\n\nActual:\n{output}"

            metrics["output_match"] = MetricRewardOutput(
                score=similarity, reason=match_reason
            )

            return RewardOutput(score=similarity, metrics=metrics)

        # No expected output provided, score based on successful execution
        return RewardOutput(score=1.0, metrics=metrics)
    else:
        # Execution failed
        error = execution_result["error"]

        metrics["execution_result"] = MetricRewardOutput(
            score=0.0,
            reason=f"Code execution failed in E2B sandbox with error:\n{error}",
        )

        return RewardOutput(score=0.0, metrics=metrics)


@reward_function
def fractional_code_reward(
    messages: Union[List[Dict[str, Any]], List[Message]],
    original_messages: Optional[Union[List[Dict[str, Any]], List[Message]]] = None,
    expected_output: Optional[str] = None,
    language: str = "python", 
    timeout: int = 30,
    environment: str = "local",
    api_key: Optional[str] = None,
    test_cases: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any
) -> RewardOutput:
    """
    Execute code and return the exact pass rate as a score between 0 and 1.
    
    Unlike the binary code reward, this function returns the actual score representing
    how closely the code output matches the expected output or how many test cases pass.
    
    Args:
        messages: Generated conversation messages
        original_messages: Original conversation context (optional)
        expected_output: Expected output from code execution
        language: Programming language of the code ("python", "javascript", etc.)
        timeout: Maximum execution time in seconds
        environment: Environment to run the code in ("local" or "e2b")
        api_key: Optional E2B API key (if using e2b environment)
        test_cases: List of test cases, each with "input" and "expected_output" keys
        **kwargs: Additional keyword arguments
    
    Returns:
        EvaluateResult with score between 0 and 1 representing the exact pass rate
    """
    # Initialize metrics dictionary
    metrics = {}
    
    # Extract the last assistant message
    if not messages:
        return RewardOutput(
            score=0.0,
            metrics={
                "error": MetricRewardOutput(score=0.0, reason="No messages provided")
            },
        )

    last_message = messages[-1]
    
    # Check role of the last message
    if getattr(last_message, "role", last_message.get("role")) != "assistant":
        return RewardOutput(
            score=0.0,
            metrics={
                "error": MetricRewardOutput(score=0.0, reason="Last message is not from assistant")
            },
        )
    
    # Get content from the message
    content = getattr(last_message, "content", last_message.get("content", ""))
    
    # Extract code blocks from the message
    code_blocks = extract_code_blocks(content, language)
    
    if not code_blocks:
        return RewardOutput(
            score=0.0,
            metrics={
                "error": MetricRewardOutput(score=0.0, reason=f"No {language} code blocks found in message")
            },
        )
    
    # Extract expected output if not provided directly
    if expected_output is None and original_messages and not test_cases:
        # Convert original_messages to standard dict format if needed
        orig_msgs = []
        for msg in original_messages:
            if hasattr(msg, "role"):
                orig_msgs.append({
                    "role": msg.role,
                    "content": msg.content
                })
            else:
                orig_msgs.append(msg)
        
        # Try to find expected output in the original messages
        for msg in orig_msgs:
            if msg.get("role") == "user":
                # Look for expected output patterns like "Expected output:" or "Output:"
                content = msg.get("content", "")
                output_patterns = [
                    r"Expected output:?\s*([\s\S]+)",
                    r"Output:?\s*([\s\S]+)",
                    r"Result:?\s*([\s\S]+)",
                    r"Should (output|return|print):?\s*([\s\S]+)",
                ]

                for pattern in output_patterns:
                    match = re.search(pattern, content)
                    if match:
                        # Use group 1 or 2 depending on the pattern
                        expected_output = (
                            match.group(2)
                            if len(match.groups()) > 1 and match.group(2)
                            else match.group(1)
                        )
                        expected_output = expected_output.strip()
                        break

                if expected_output:
                    break
    
    # Use the first code block for execution
    code = code_blocks[0]["code"]
    
    # Log the extracted code
    metrics["extracted_code"] = f"Extracted code:\n```{language}\n{code}\n```"
    
    # Add expected output to metrics if available and not using test cases
    if expected_output and not test_cases:
        metrics["expected_output"] = f"Expected output:\n{expected_output}"
    
    # Handle multiple test cases if provided
    if test_cases:
        return _run_test_cases(code, language, test_cases, timeout, environment, api_key)
    
    # Execute code in specified environment
    if environment.lower() == "e2b":
        if not _HAS_E2B:
            return RewardOutput(
                score=0.0,
                metrics={
                    "error": MetricRewardOutput(score=0.0, reason="E2B package not installed. Install with: pip install e2b")
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
            return RewardOutput(
                score=0.0,
                metrics={
                    "error": MetricRewardOutput(score=0.0, reason=f"Unsupported language: {language}")
                },
            )
    
    # Check execution result
    if execution_result["success"]:
        output = execution_result["output"]
        
        metrics["execution_result"] = f"Code executed successfully with output:\n{output}"
        
        # Compare with expected output if provided
        if expected_output:
            similarity = compare_outputs(output, expected_output)
            match_reason = f"Output similarity: {similarity:.2f}\n\nExpected:\n{expected_output}\n\nActual:\n{output}"
            
            # Convert metrics dict to MetricResult objects
            metric_results = {}
            for key, value in metrics.items():
                if isinstance(value, str):
                    metric_results[key] = MetricResult(score=1.0, reason=value)
                elif isinstance(value, dict) and "score" in value and "reason" in value:
                    metric_results[key] = MetricResult(score=value.get("score", 1.0), reason=value.get("reason", ""))
            
            # Add output match metric
            metric_results["output_match"] = MetricResult(score=similarity, reason=match_reason)
            
            # Convert MetricResult objects to MetricRewardOutput for RewardOutput
            reward_metrics = {}
            for key, value in metric_results.items():
                if isinstance(value, MetricResult):
                    reward_metrics[key] = MetricRewardOutput(score=value.score, reason=value.reason)
                else:
                    reward_metrics[key] = value
                    
            return RewardOutput(score=similarity, metrics=reward_metrics)
        
        # No expected output provided, score based on successful execution
        # Convert metrics dict to MetricResult objects
        metric_results = {}
        for key, value in metrics.items():
            if isinstance(value, str):
                metric_results[key] = MetricResult(score=1.0, reason=value)
            elif isinstance(value, dict) and "score" in value and "reason" in value:
                metric_results[key] = MetricResult(score=value.get("score", 1.0), reason=value.get("reason", ""))
                
        # Convert MetricResult objects to MetricRewardOutput for RewardOutput
        reward_metrics = {}
        for key, value in metric_results.items():
            if isinstance(value, MetricResult):
                reward_metrics[key] = MetricRewardOutput(score=value.score, reason=value.reason)
            else:
                reward_metrics[key] = value
                
        return RewardOutput(score=1.0, metrics=reward_metrics)
    else:
        # Execution failed
        error = execution_result["error"]
        
        metrics["execution_result"] = f"Code execution failed with error:\n{error}"
        
        metric_results = {}
        for key, value in metrics.items():
            if isinstance(value, str):
                metric_results[key] = MetricResult(score=0.0, reason=value)
            elif isinstance(value, dict) and "score" in value and "reason" in value:
                metric_results[key] = MetricResult(score=value.get("score", 0.0), reason=value.get("reason", ""))
        
        # Convert MetricResult objects to MetricRewardOutput for RewardOutput
        reward_metrics = {}
        for key, value in metric_results.items():
            if isinstance(value, MetricResult):
                reward_metrics[key] = MetricRewardOutput(score=value.score, reason=value.reason)
            else:
                reward_metrics[key] = value
                
        return RewardOutput(score=0.0, metrics=reward_metrics)


def _run_test_cases(
    code: str,
    language: str,
    test_cases: List[Dict[str, Any]],
    timeout: int,
    environment: str,
    api_key: Optional[str] = None
) -> EvaluateResult:
    """
    Run code against multiple test cases and return the fraction of passing tests.
    
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
    metrics = {}
    results = []
    passed = 0
    total = len(test_cases)
    
    if total == 0:
        return RewardOutput(
            score=0.0,
            metrics={"error": MetricRewardOutput(score=0.0, reason="No test cases provided")}
        )
    
    # Prepare the code wrapper based on language
    if language.lower() in ["python", "py"]:
        # Python code wrapper that captures input/output and provides test input
        def prepare_test_code(test_code: str, test_input: str) -> str:
            return (
                "import sys\n"
                "from io import StringIO\n\n"
                "# Capture stdout\n"
                "original_stdout = sys.stdout\n"
                "sys.stdout = StringIO()\n\n"
                "# Set up test input\n"
                f"sys.stdin = StringIO('''{test_input}''')\n\n"
                "# User code\n"
                f"{test_code}\n\n"
                "# Get output\n"
                "output = sys.stdout.getvalue()\n"
                "sys.stdout = original_stdout\n"
                "print(output, end='')"
            )
    elif language.lower() in ["javascript", "js"]:
        # JavaScript code wrapper that captures console output and provides input
        def prepare_test_code(test_code: str, test_input: str) -> str:
            # Create a simple input simulation for JavaScript
            input_lines = test_input.strip().split("\n")
            input_setup = "const inputs = " + json.dumps(input_lines) + ";\n"
            input_setup += "let inputIndex = 0;\n"
            input_setup += "const readline = () => inputs[inputIndex++];\n"
            
            return (
                "// Capture console.log output\n"
                "const originalLog = console.log;\n"
                "let output = '';\n"
                "console.log = function() {\n"
                "  output += Array.from(arguments).join(' ') + '\\n';\n"
                "};\n\n"
                f"{input_setup}\n\n"
                "// User code\n"
                f"{test_code}\n\n"
                "// Print captured output\n"
                "console.log = originalLog;\n"
                "console.log(output);"
            )
    else:
        return RewardOutput(
            score=0.0,
            metrics={"error": MetricRewardOutput(score=0.0, reason=f"Unsupported language for test cases: {language}")}
        )
    
    # Process each test case
    for i, test_case in enumerate(test_cases):
        test_input = test_case.get("input", "")
        expected = test_case.get("expected_output", "")
        
        # Prepare code with test input
        test_code = prepare_test_code(code, test_input)
        
        # Execute code in the specified environment
        if environment.lower() == "e2b":
            if not _HAS_E2B:
                return RewardOutput(
                    score=0.0,
                    metrics={"error": MetricRewardOutput(score=0.0, reason="E2B package not installed. Install with: pip install e2b")}
                )
            
            execution_result = execute_code_with_e2b(
                code=test_code, language=language, timeout=timeout, api_key=api_key
            )
        else:  # local execution
            if language.lower() in ["python", "py"]:
                execution_result = execute_python_code(test_code, timeout)
            elif language.lower() in ["javascript", "js"]:
                execution_result = execute_javascript_code(test_code, timeout)
        
        # Process the result
        test_result = {
            "test_number": i + 1,
            "input": test_input,
            "expected_output": expected,
            "passed": False,
            "details": ""
        }
        
        if execution_result["success"]:
            output = execution_result["output"]
            similarity = compare_outputs(output, expected)
            
            # Consider the test passed if similarity is high (above 0.9)
            test_result["passed"] = similarity > 0.9
            test_result["similarity"] = similarity
            test_result["actual_output"] = output
            test_result["details"] = f"Similarity: {similarity:.2f}"
            
            if test_result["passed"]:
                passed += 1
        else:
            test_result["error"] = execution_result["error"]
            test_result["details"] = f"Error: {execution_result['error']}"
        
        results.append(test_result)
    
    # Calculate the final score as the fraction of passing tests
    score = passed / total if total > 0 else 0.0
    
    metrics["test_results"] = results
    metrics["pass_rate"] = f"{passed}/{total} tests passed ({score:.2%})"
    
    # Convert metrics to MetricResult objects
    metric_results = {}
    for key, value in metrics.items():
        if key == "test_results":  # Special handling for test results array
            metric_results[key] = MetricResult(
                score=score,
                reason=str(value)  # Convert test results to string for display
            )
        elif isinstance(value, str):
            metric_results[key] = MetricResult(score=score, reason=value)
        elif isinstance(value, dict) and "score" in value and "reason" in value:
            metric_results[key] = MetricResult(score=value.get("score", score), reason=value.get("reason", ""))
    
    # Convert MetricResult objects to MetricRewardOutput for RewardOutput
    reward_metrics = {}
    for key, value in metric_results.items():
        if isinstance(value, MetricResult):
            reward_metrics[key] = MetricRewardOutput(score=value.score, reason=value.reason)
        else:
            reward_metrics[key] = value
            
    return RewardOutput(score=score, metrics=reward_metrics)


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
                resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
            )
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
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
