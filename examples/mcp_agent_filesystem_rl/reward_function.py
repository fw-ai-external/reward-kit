"""
MCP Agent Filesystem RL Reward Function

This reward function evaluates whether an LLM agent successfully moved
a file from one directory to another using MCP filesystem tools.
"""

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Union

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from reward_kit.models import EvaluateResult, Message, MetricResult
from reward_kit.typed_interface import reward_function

logger = logging.getLogger(__name__)

# MCP Intermediary Server URL
INTERMEDIARY_SERVER_URL = "http://localhost:8001/mcp"


@reward_function
def mcp_filesystem_move_reward(
    messages: Union[List[Dict[str, Any]], List[Message]],
    rk_session_id: Optional[str] = None,
    instance_id: Optional[str] = None,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Reward function that evaluates whether the agent successfully moved a file
    from /data/source_files/ to /data/archive/ using MCP filesystem tools.

    Args:
        messages: List of conversation messages
        rk_session_id: The RewardKit session ID for MCP agent interaction
        instance_id: The specific filesystem instance ID to check
        **kwargs: Additional arguments

    Returns:
        EvaluateResult with evaluation score and metrics
    """

    if not rk_session_id:
        return EvaluateResult(
            score=0.0,
            reason="No rk_session_id provided for MCP agent interaction",
            is_score_valid=False,
            metrics={},
        )

    if not instance_id:
        return EvaluateResult(
            score=0.0,
            reason="No instance_id provided for MCP filesystem instance",
            is_score_valid=False,
            metrics={},
        )

    try:
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in a loop, we need to create a task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, _verify_file_move_async(rk_session_id, instance_id)
                )
                result = future.result(timeout=30)
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            result = asyncio.run(_verify_file_move_async(rk_session_id, instance_id))

        return result
    except Exception as e:
        logger.error(f"Error in MCP filesystem reward evaluation: {e}", exc_info=True)
        return EvaluateResult(
            score=0.0,
            reason=f"Error during MCP evaluation: {str(e)}",
            is_score_valid=False,
            metrics={},
        )


async def _verify_file_move_async(
    rk_session_id: str, instance_id: str
) -> EvaluateResult:
    """
    Async helper to verify the file was moved correctly using MCP tools.
    """

    async with AsyncExitStack() as stack:
        try:
            # Connect to MCP intermediary server
            transport_tuple = await stack.enter_async_context(
                streamablehttp_client(INTERMEDIARY_SERVER_URL)
            )
            read_stream, write_stream, _ = transport_tuple

            mcp_client_session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await mcp_client_session.initialize()

            # Check if file is in target directory (/data/archive/)
            target_check = await _call_mcp_tool(
                mcp_client_session,
                rk_session_id,
                instance_id,
                "list_directory",
                {"path": "/data/archive"},
            )

            target_files = _extract_directory_listing(target_check)
            file_in_target = "important_document.txt" in target_files

            # Check if file is still in source directory (/data/source_files/)
            source_check = await _call_mcp_tool(
                mcp_client_session,
                rk_session_id,
                instance_id,
                "list_directory",
                {"path": "/data/source_files"},
            )

            source_files = _extract_directory_listing(source_check)
            file_in_source = "important_document.txt" in source_files

            # Calculate score and reason
            if file_in_target and not file_in_source:
                score = 1.0
                reason = (
                    "File successfully moved from /data/source_files/ to /data/archive/"
                )
                move_success = True
            elif file_in_target and file_in_source:
                score = 0.5
                reason = "File copied to /data/archive/ but still exists in /data/source_files/ (should be moved, not copied)"
                move_success = False
            elif not file_in_target and not file_in_source:
                score = 0.0
                reason = "File not found in either directory - may have been deleted or moved elsewhere"
                move_success = False
            else:  # not file_in_target and file_in_source
                score = 0.0
                reason = "File still in /data/source_files/ and not in /data/archive/ - move operation not performed"
                move_success = False

            return EvaluateResult(
                score=score,
                reason=reason,
                is_score_valid=True,
                metrics={
                    "file_move_success": MetricResult(
                        score=1.0 if move_success else 0.0,
                        is_score_valid=True,
                        reason=f"File move {'successful' if move_success else 'failed'}",
                    ),
                    "target_directory_check": MetricResult(
                        score=1.0 if file_in_target else 0.0,
                        is_score_valid=True,
                        reason=f"File {'found' if file_in_target else 'not found'} in target directory",
                    ),
                    "source_directory_check": MetricResult(
                        score=0.0 if file_in_source else 1.0,
                        is_score_valid=True,
                        reason=f"File {'still present' if file_in_source else 'removed'} from source directory",
                    ),
                },
            )

        except Exception as e:
            logger.error(f"Error during async MCP verification: {e}", exc_info=True)
    return EvaluateResult(
        score=0.0,
        reason=f"Error connecting to MCP server or checking filesystem",
        is_score_valid=False,
        metrics={},
    )


async def _call_mcp_tool(
    session: ClientSession,
    rk_session_id: str,
    instance_id: str,
    tool_name: str,
    tool_args: Dict[str, Any],
) -> Dict[str, Any]:
    """Helper to call backend tools via MCP intermediary."""

    payload = {
        "args": {
            "rk_session_id": rk_session_id,
            "backend_name_ref": "filesystem_test",
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


def _extract_directory_listing(mcp_result: Dict[str, Any]) -> List[str]:
    """Extract file names from MCP directory listing result."""

    if mcp_result.get("isError"):
        return []

    content = mcp_result.get("content", [])
    if not content or not isinstance(content[0], dict):
        return []

    listing_text = content[0].get("text", "").strip()

    # Parse the directory listing format: "[FILE] filename" or "[DIR] dirname"
    files = []
    for line in listing_text.split("\n"):
        line = line.strip()
        if line.startswith("[FILE]"):
            filename = line.replace("[FILE]", "").strip()
            if filename:
                files.append(filename)

    return files
