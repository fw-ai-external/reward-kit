"""
Simplified MCP Agent Filesystem RL Reward Function

This reward function works with the standard reward-kit pipeline by managing
MCP sessions internally. It creates a new filesystem instance for each evaluation,
simulates the agent's actions by parsing the LLM response, and then evaluates
the final filesystem state.
"""

import asyncio
import json
import logging
import re
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Union

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from reward_kit.models import EvaluateResult, Message, MetricResult
from reward_kit.typed_interface import reward_function

logger = logging.getLogger(__name__)


@reward_function
def mcp_filesystem_move_reward(
    messages: Union[List[Dict[str, Any]], List[Message]],
    ground_truth: Optional[Union[str, Dict[str, Any]]] = None,
    backend_name_ref: str = "filesystem_rl_example",
    mcp_server_url: str = "http://localhost:8001/mcp",
    **kwargs: Any,
) -> EvaluateResult:
    """
    Reward function that evaluates filesystem move operations via MCP.

    This function:
    1. Creates a new MCP session with a filesystem instance
    2. Parses the LLM's response to extract intended file operations
    3. Simulates executing those operations via MCP tools
    4. Evaluates whether the file was successfully moved
    5. Cleans up the session

    Args:
        messages: List of conversation messages (user query + assistant response)
        ground_truth: Expected ground truth (optional)
        backend_name_ref: MCP backend reference name
        mcp_server_url: URL of the MCP intermediary server
        **kwargs: Additional arguments

    Returns:
        EvaluateResult with evaluation score and metrics
    """

    try:
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in a loop, use a thread pool
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    _evaluate_with_mcp_async(
                        messages, ground_truth, backend_name_ref, mcp_server_url
                    ),
                )
                result = future.result(timeout=60)
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            result = asyncio.run(
                _evaluate_with_mcp_async(
                    messages, ground_truth, backend_name_ref, mcp_server_url
                )
            )

        return result
    except Exception as e:
        logger.error(f"Error in MCP filesystem reward evaluation: {e}", exc_info=True)
        return EvaluateResult(
            score=0.0,
            reason=f"Error during MCP evaluation: {str(e)}",
            is_score_valid=False,
            metrics={},
        )


async def _evaluate_with_mcp_async(
    messages: Union[List[Dict[str, Any]], List[Message]],
    ground_truth: Optional[Union[str, Dict[str, Any]]],
    backend_name_ref: str,
    mcp_server_url: str,
) -> EvaluateResult:
    """
    Async helper to evaluate the LLM's response using MCP tools.
    """

    # Extract the assistant's response from messages
    assistant_response = None
    for msg in messages:
        if isinstance(msg, dict):
            if msg.get("role") == "assistant":
                assistant_response = msg.get("content", "")
                break
        elif hasattr(msg, "role") and msg.role == "assistant":
            assistant_response = msg.content
            break

    if not assistant_response:
        return EvaluateResult(
            score=0.0,
            reason="No assistant response found in messages",
            is_score_valid=False,
            metrics={},
        )

    async with AsyncExitStack() as stack:
        try:
            # Connect to MCP intermediary server
            transport_tuple = await stack.enter_async_context(
                streamablehttp_client(mcp_server_url)
            )
            read_stream, write_stream, _ = transport_tuple

            mcp_client_session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await mcp_client_session.initialize()

            # Initialize a new session with filesystem instance
            init_payload = {
                "args": {
                    "backends": [
                        {"backend_name_ref": backend_name_ref, "num_instances": 1}
                    ]
                }
            }

            init_result = await mcp_client_session.call_tool(
                "initialize_session", init_payload
            )
            if init_result.isError:
                raise ValueError(f"Failed to initialize MCP session: {init_result}")

            init_data = json.loads(init_result.content[0].text)
            rk_session_id = init_data["rk_session_id"]
            instance_id = init_data["initialized_backends"][0]["instances"][0][
                "instance_id"
            ]

            # Check initial state
            initial_source_files = await _list_directory_files(
                mcp_client_session,
                rk_session_id,
                backend_name_ref,
                instance_id,
                "/data/source_files",
            )
            initial_archive_files = await _list_directory_files(
                mcp_client_session,
                rk_session_id,
                backend_name_ref,
                instance_id,
                "/data/archive",
            )

            target_file = "important_document.txt"

            # Check if the file is initially in the right place
            if target_file not in initial_source_files:
                return EvaluateResult(
                    score=0.0,
                    reason=f"Target file '{target_file}' not found in initial source directory",
                    is_score_valid=False,
                    metrics={},
                )

            # Parse the assistant response to extract file operations
            move_operations = _parse_move_operations(assistant_response)

            # Execute the operations mentioned in the response
            operations_executed = 0
            successful_moves = 0

            for operation in move_operations:
                try:
                    if operation["type"] == "move_file":
                        await _call_mcp_tool(
                            mcp_client_session,
                            rk_session_id,
                            backend_name_ref,
                            instance_id,
                            "move_file",
                            {
                                "source": operation["source"],
                                "destination": operation["destination"],
                            },
                        )
                        operations_executed += 1
                        successful_moves += 1
                    # Add other operation types as needed (list_directory, read_file, etc.)
                except Exception as e:
                    logger.warning(f"Failed to execute operation {operation}: {e}")
                    operations_executed += 1

            # Check final state
            final_source_files = await _list_directory_files(
                mcp_client_session,
                rk_session_id,
                backend_name_ref,
                instance_id,
                "/data/source_files",
            )
            final_archive_files = await _list_directory_files(
                mcp_client_session,
                rk_session_id,
                backend_name_ref,
                instance_id,
                "/data/archive",
            )

            # Parse ground truth for expected state
            expected_state = {}
            success_criteria = {}

            if isinstance(ground_truth, dict):
                expected_state = ground_truth.get("expected_final_state", {})
                success_criteria = ground_truth.get("success_criteria", {})

            # Evaluate the outcome
            file_in_archive = target_file in final_archive_files
            file_removed_from_source = target_file not in final_source_files

            # Compare against expected final state if provided
            state_match_score = 1.0
            state_match_details = []

            if expected_state:
                for directory, expected_files in expected_state.items():
                    if directory == "/data/source_files":
                        actual_files = final_source_files
                    elif directory == "/data/archive":
                        actual_files = final_archive_files
                    else:
                        continue

                    # Filter out .gitkeep for comparison
                    expected_clean = [f for f in expected_files if f != ".gitkeep"]
                    actual_clean = [f for f in actual_files if f != ".gitkeep"]

                    if set(expected_clean) == set(actual_clean):
                        state_match_details.append(f"{directory}: ✓ matches expected")
                    else:
                        state_match_details.append(
                            f"{directory}: ✗ expected {expected_clean}, got {actual_clean}"
                        )
                        state_match_score = 0.0

            # Check success criteria if provided
            criteria_met = True
            criteria_details = []

            if success_criteria:
                if success_criteria.get("file_in_target", True):
                    if file_in_archive:
                        criteria_details.append("✓ file in target directory")
                    else:
                        criteria_details.append("✗ file not in target directory")
                        criteria_met = False

                if success_criteria.get("file_removed_from_source", True):
                    if file_removed_from_source:
                        criteria_details.append("✓ file removed from source")
                    else:
                        criteria_details.append("✗ file still in source")
                        criteria_met = False

                if success_criteria.get("file_moved", True):
                    if file_in_archive and file_removed_from_source:
                        criteria_details.append("✓ file successfully moved")
                    else:
                        criteria_details.append("✗ file not properly moved")
                        criteria_met = False

            # Calculate final score
            if expected_state and success_criteria:
                # Use both state matching and criteria
                score = (state_match_score + (1.0 if criteria_met else 0.0)) / 2.0
                reason = (
                    f"State match: {state_match_score:.1f}, Criteria met: {criteria_met}. "
                    + f"State: {'; '.join(state_match_details)}. Criteria: {'; '.join(criteria_details)}"
                )
            elif expected_state:
                # Use state matching only
                score = state_match_score
                reason = f"State matching: {'; '.join(state_match_details)}"
            elif success_criteria:
                # Use criteria only
                score = 1.0 if criteria_met else 0.0
                reason = f"Success criteria: {'; '.join(criteria_details)}"
            else:
                # Fallback to simple file location check
                if file_in_archive and file_removed_from_source:
                    score = 1.0
                    reason = (
                        f"Successfully moved '{target_file}' from source to archive"
                    )
                elif file_in_archive and not file_removed_from_source:
                    score = 0.5
                    reason = f"File '{target_file}' copied to archive but still in source (should be moved)"
                elif not file_in_archive and not file_removed_from_source:
                    score = 0.0
                    reason = (
                        f"File '{target_file}' still in source, not moved to archive"
                    )
                else:  # not in archive, removed from source
                    score = 0.0
                    reason = f"File '{target_file}' disappeared from source but not found in archive"

            # Cleanup session
            try:
                cleanup_payload = {"args": {"rk_session_id": rk_session_id}}
                await mcp_client_session.call_tool("cleanup_session", cleanup_payload)
            except Exception as e:
                logger.warning(f"Failed to cleanup MCP session: {e}")

            return EvaluateResult(
                score=score,
                reason=reason,
                is_score_valid=True,
                metrics={
                    "file_move_success": MetricResult(
                        score=(
                            1.0
                            if (file_in_archive and file_removed_from_source)
                            else 0.0
                        ),
                        is_score_valid=True,
                        reason=f"File move {'successful' if (file_in_archive and file_removed_from_source) else 'failed'}",
                    ),
                    "state_match_score": MetricResult(
                        score=state_match_score,
                        is_score_valid=True,
                        reason=f"Filesystem state matching score: {state_match_score:.2f}",
                    ),
                    "success_criteria_met": MetricResult(
                        score=1.0 if criteria_met else 0.0,
                        is_score_valid=True,
                        reason=f"Success criteria {'met' if criteria_met else 'not met'}",
                    ),
                    "operations_detected": MetricResult(
                        score=min(
                            len(move_operations), 1.0
                        ),  # Cap at 1.0 for MetricResult validation
                        is_score_valid=True,
                        reason=f"Detected {len(move_operations)} file operations in response",
                    ),
                    "operations_executed": MetricResult(
                        score=min(
                            successful_moves, 1.0
                        ),  # Cap at 1.0 for MetricResult validation
                        is_score_valid=True,
                        reason=f"Successfully executed {successful_moves} file operations",
                    ),
                    "file_in_target": MetricResult(
                        score=1.0 if file_in_archive else 0.0,
                        is_score_valid=True,
                        reason=f"File {'found' if file_in_archive else 'not found'} in target directory",
                    ),
                    "file_removed_from_source": MetricResult(
                        score=1.0 if file_removed_from_source else 0.0,
                        is_score_valid=True,
                        reason=f"File {'removed' if file_removed_from_source else 'still present'} in source directory",
                    ),
                },
            )

        except Exception as e:
            logger.error(f"Error during async MCP evaluation: {e}", exc_info=True)
    return EvaluateResult(
        score=0.0,
        reason=f"Error connecting to MCP server or evaluating",
        is_score_valid=False,
        metrics={},
    )


async def _call_mcp_tool(
    session: ClientSession,
    rk_session_id: str,
    backend_name_ref: str,
    instance_id: str,
    tool_name: str,
    tool_args: Dict[str, Any],
) -> Dict[str, Any]:
    """Helper to call backend tools via MCP intermediary."""

    payload = {
        "args": {
            "rk_session_id": rk_session_id,
            "backend_name_ref": backend_name_ref,
            "instance_id": instance_id,
            "tool_name": tool_name,
            "tool_args": tool_args,
        }
    }

    result = await session.call_tool("call_backend_tool", payload)

    if result.isError or not result.content or not hasattr(result.content[0], "text"):
        raise ValueError(
            f"MCP tool call failed: {getattr(result.content[0], 'text', 'Unknown error') if result.content else 'No content'}"
        )

    return json.loads(result.content[0].text)


async def _list_directory_files(
    session: ClientSession,
    rk_session_id: str,
    backend_name_ref: str,
    instance_id: str,
    directory_path: str,
) -> List[str]:
    """Get list of files in a directory via MCP."""

    try:
        result = await _call_mcp_tool(
            session,
            rk_session_id,
            backend_name_ref,
            instance_id,
            "list_directory",
            {"path": directory_path},
        )

        if result.get("isError"):
            return []

        content = result.get("content", [])
        if not content or not isinstance(content[0], dict):
            return []

        listing_text = content[0].get("text", "").strip()

        # Parse the directory listing format: "[FILE] filename"
        files = []
        for line in listing_text.split("\n"):
            line = line.strip()
            if line.startswith("[FILE]"):
                filename = line.replace("[FILE]", "").strip()
                if filename and filename != ".gitkeep":
                    files.append(filename)

        return files
    except Exception as e:
        logger.warning(f"Failed to list directory {directory_path}: {e}")
        return []


def _parse_move_operations(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse the LLM response to extract intended file operations.
    This is a simple heuristic-based parser.
    """
    operations = []

    # Look for move_file operations in the response
    # This regex looks for patterns like: move_file("/data/source/file.txt", "/data/target/file.txt")
    move_patterns = [
        r'move_files?\s*\(\s*source\s*=\s*["\']([^"\']+)["\']\s*,\s*destination\s*=\s*["\']([^"\']+)["\']\s*\)',
        r'move_files?\s*\(\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\s*\)',
        r'move.*["\']([^"\']*important_document\.txt[^"\']*)["\'].*to.*["\']([^"\']*archive[^"\']*)["\']',
        r'move.*from.*["\']([^"\']*source_files[^"\']*)["\'].*to.*["\']([^"\']*archive[^"\']*)["\']',
    ]

    for pattern in move_patterns:
        matches = re.finditer(pattern, response_text, re.IGNORECASE)
        for match in matches:
            source = match.group(1)
            destination = match.group(2)

            # If the destination doesn't include the filename, add it
            if destination.endswith("/") or "important_document.txt" not in destination:
                if destination.endswith("/"):
                    destination += "important_document.txt"
                else:
                    destination += "/important_document.txt"

            operations.append(
                {"type": "move_file", "source": source, "destination": destination}
            )

    # If no explicit operations found, but the response mentions moving the file,
    # add a default operation
    if not operations and re.search(
        r"move.*important_document\.txt", response_text, re.IGNORECASE
    ):
        operations.append(
            {
                "type": "move_file",
                "source": "/data/source_files/important_document.txt",
                "destination": "/data/archive/important_document.txt",
            }
        )

    return operations
