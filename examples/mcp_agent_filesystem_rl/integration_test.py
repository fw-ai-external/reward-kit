#!/usr/bin/env python3
"""
Integration test for MCP Agent Filesystem RL Example

This test demonstrates the complete RL evaluation flow:
1. Start MCP intermediary server
2. Initialize session with filesystem_rl_example backend
3. Simulate an agent moving the file
4. Evaluate with the reward function
5. Clean up
"""

import asyncio
import json
import subprocess
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

# Add the reward-kit package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the reward function
import reward_function
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def simulate_file_move_task():
    """Simulate the complete file move task and evaluation."""

    print("Starting MCP Agent Filesystem RL Integration Test")
    print("=" * 55)

    async with AsyncExitStack() as stack:
        print("1. Connecting to MCP intermediary server...")
        transport_tuple = await stack.enter_async_context(
            streamablehttp_client("http://localhost:8001/mcp")
        )
        read_stream, write_stream, _ = transport_tuple

        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        print("✓ Connected to MCP intermediary server")

        print("\n2. Initializing session with filesystem_rl_example backend...")
        init_payload = {
            "args": {
                "backends": [
                    {"backend_name_ref": "filesystem_rl_example", "num_instances": 1}
                ]
            }
        }

        init_result = await session.call_tool("initialize_session", init_payload)
        init_data = json.loads(init_result.content[0].text)

        rk_session_id = init_data["rk_session_id"]
        instance_id = init_data["initialized_backends"][0]["instances"][0][
            "instance_id"
        ]

        print(f"✓ Session initialized: {rk_session_id}")
        print(f"✓ Instance created: {instance_id}")

        print("\n3. Verifying initial state...")
        # Check source directory
        source_check = await call_backend_tool(
            session,
            rk_session_id,
            instance_id,
            "list_directory",
            {"path": "/data/source_files"},
        )
        source_files = extract_files_from_listing(source_check)
        print(f"✓ Source directory contents: {source_files}")
        assert (
            "important_document.txt" in source_files
        ), "important_document.txt not in source!"

        # Check archive directory
        archive_check = await call_backend_tool(
            session,
            rk_session_id,
            instance_id,
            "list_directory",
            {"path": "/data/archive"},
        )
        archive_files = extract_files_from_listing(archive_check)
        print(f"✓ Archive directory contents: {archive_files}")

        print("\n4. Simulating agent action: moving file...")
        move_result = await call_backend_tool(
            session,
            rk_session_id,
            instance_id,
            "move_file",
            {
                "source": "/data/source_files/important_document.txt",
                "destination": "/data/archive/important_document.txt",
            },
        )
        print(
            f"✓ Move operation result: {move_result.get('content', [{}])[0].get('text', 'No details')}"
        )

        print("\n5. Verifying final state...")
        # Check source directory again
        source_check = await call_backend_tool(
            session,
            rk_session_id,
            instance_id,
            "list_directory",
            {"path": "/data/source_files"},
        )
        source_files_after = extract_files_from_listing(source_check)
        print(f"✓ Source directory after move: {source_files_after}")

        # Check archive directory again
        archive_check = await call_backend_tool(
            session,
            rk_session_id,
            instance_id,
            "list_directory",
            {"path": "/data/archive"},
        )
        archive_files_after = extract_files_from_listing(archive_check)
        print(f"✓ Archive directory after move: {archive_files_after}")

        print("\n6. Evaluating with reward function...")
        # Simulate agent messages
        messages = [
            {
                "role": "user",
                "content": "You have access to a filesystem. Please move the file named 'important_document.txt' from the '/data/source_files/' directory to the '/data/archive/' directory.",
            },
            {
                "role": "assistant",
                "content": "I'll move the important_document.txt file from /data/source_files/ to /data/archive/ using the move_file tool.",
            },
        ]

        # Call reward function
        reward_result = reward_function.mcp_filesystem_move_reward(
            messages=messages, rk_session_id=rk_session_id, instance_id=instance_id
        )

        print(f"✓ Reward score: {reward_result.score}")
        print(f"✓ Reward reason: {reward_result.reason}")
        print(f"✓ Score valid: {reward_result.is_score_valid}")

        if reward_result.metrics:
            print("✓ Detailed metrics:")
            for metric_name, metric_result in reward_result.metrics.items():
                print(
                    f"  - {metric_name}: {metric_result.score} ({metric_result.reason})"
                )

        print("\n7. Cleaning up session...")
        cleanup_payload = {"args": {"rk_session_id": rk_session_id}}
        await session.call_tool("cleanup_session", cleanup_payload)
        print("✓ Session cleaned up")

        print("\n" + "=" * 55)
        print("🎉 Integration test completed successfully!")
        print(f"Final score: {reward_result.score}/1.0")

        return reward_result.score == 1.0


async def call_backend_tool(session, rk_session_id, instance_id, tool_name, tool_args):
    """Helper to call backend tools via MCP intermediary."""
    payload = {
        "args": {
            "rk_session_id": rk_session_id,
            "backend_name_ref": "filesystem_rl_example",
            "instance_id": instance_id,
            "tool_name": tool_name,
            "tool_args": tool_args,
        }
    }

    result = await session.call_tool("call_backend_tool", payload)
    return json.loads(result.content[0].text)


def extract_files_from_listing(mcp_result):
    """Extract file names from MCP directory listing result."""
    if mcp_result.get("isError"):
        return []

    content = mcp_result.get("content", [])
    if not content or not isinstance(content[0], dict):
        return []

    listing_text = content[0].get("text", "").strip()

    files = []
    for line in listing_text.split("\n"):
        line = line.strip()
        if line.startswith("[FILE]"):
            filename = line.replace("[FILE]", "").strip()
            if filename and filename != ".gitkeep":
                files.append(filename)

    return files


def main():
    """Run the integration test."""
    print("Starting MCP intermediary server...")

    # Start the MCP server
    server_process = subprocess.Popen(
        [sys.executable, "-m", "reward_kit.mcp_agent.main"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Wait for server to start
        time.sleep(8)
        print("✓ MCP server should be started")

        # Run the test
        success = asyncio.run(simulate_file_move_task())

        if success:
            print("\n🎉 All tests passed!")
            return 0
        else:
            print("\n❌ Test failed - reward score was not 1.0")
            return 1

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        print("\nShutting down MCP server...")
        server_process.terminate()
        server_process.wait()
        print("✓ MCP server shut down")


if __name__ == "__main__":
    sys.exit(main())
