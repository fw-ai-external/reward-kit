#!/usr/bin/env python3
"""
Simple integration test that avoids async event loop conflicts
"""

import asyncio
import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path

# Add the reward-kit package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client


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
    for line in listing_text.split('\n'):
        line = line.strip()
        if line.startswith('[FILE]'):
            filename = line.replace('[FILE]', '').strip()
            if filename and filename != '.gitkeep':
                files.append(filename)
    
    return files


async def test_filesystem_rl_scenario():
    """Test the complete filesystem RL scenario."""
    
    print("Simple MCP Agent Filesystem RL Integration Test")
    print("=" * 50)
    
    async with AsyncExitStack() as stack:
        print("1. Connecting to MCP intermediary server...")
        try:
            transport_tuple = await stack.enter_async_context(
                streamablehttp_client("http://localhost:8001/mcp")
            )
            read_stream, write_stream, _ = transport_tuple
            
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            print("✓ Connected to MCP intermediary server")
        except Exception as e:
            print(f"❌ Failed to connect to MCP server: {e}")
            return False
        
        print("\n2. Initializing session with filesystem_rl_example backend...")
        try:
            init_payload = {
                "args": {
                    "backends": [{"backend_name_ref": "filesystem_rl_example", "num_instances": 1}]
                }
            }
            
            init_result = await session.call_tool("initialize_session", init_payload)
            init_data = json.loads(init_result.content[0].text)
            
            rk_session_id = init_data["rk_session_id"]
            instance_id = init_data["initialized_backends"][0]["instances"][0]["instance_id"]
            
            print(f"✓ Session initialized: {rk_session_id}")
            print(f"✓ Instance created: {instance_id}")
        except Exception as e:
            print(f"❌ Failed to initialize session: {e}")
            return False
        
        print("\n3. Verifying initial state...")
        try:
            # Check source directory
            source_check = await call_backend_tool(
                session, rk_session_id, instance_id,
                "list_directory", {"path": "/data/source_files"}
            )
            source_files = extract_files_from_listing(source_check)
            print(f"✓ Source directory contents: {source_files}")
            
            if "important_document.txt" not in source_files:
                print("❌ important_document.txt not found in source directory!")
                return False
            
            # Check archive directory  
            archive_check = await call_backend_tool(
                session, rk_session_id, instance_id,
                "list_directory", {"path": "/data/archive"}
            )
            archive_files = extract_files_from_listing(archive_check)
            print(f"✓ Archive directory contents: {archive_files}")
            
            if "important_document.txt" in archive_files:
                print("❌ important_document.txt already in archive directory!")
                return False
        except Exception as e:
            print(f"❌ Failed to verify initial state: {e}")
            return False
        
        print("\n4. Simulating agent action: moving file...")
        try:
            move_result = await call_backend_tool(
                session, rk_session_id, instance_id,
                "move_file", {
                    "source": "/data/source_files/important_document.txt",
                    "destination": "/data/archive/important_document.txt"
                }
            )
            print(f"✓ Move operation: {move_result.get('content', [{}])[0].get('text', 'Success')}")
        except Exception as e:
            print(f"❌ Failed to move file: {e}")
            return False
        
        print("\n5. Verifying final state...")
        try:
            # Check source directory again
            source_check = await call_backend_tool(
                session, rk_session_id, instance_id,
                "list_directory", {"path": "/data/source_files"}
            )
            source_files_after = extract_files_from_listing(source_check)
            print(f"✓ Source directory after move: {source_files_after}")
            
            # Check archive directory again
            archive_check = await call_backend_tool(
                session, rk_session_id, instance_id,
                "list_directory", {"path": "/data/archive"}
            )
            archive_files_after = extract_files_from_listing(archive_check)
            print(f"✓ Archive directory after move: {archive_files_after}")
            
            # Verify success conditions
            file_moved_to_archive = "important_document.txt" in archive_files_after
            file_removed_from_source = "important_document.txt" not in source_files_after
            
            if file_moved_to_archive and file_removed_from_source:
                print("✓ File successfully moved from source to archive!")
                success = True
            elif file_moved_to_archive and not file_removed_from_source:
                print("⚠ File copied to archive but still in source (should be moved)")
                success = False
            else:
                print("❌ File move operation failed")
                success = False
        except Exception as e:
            print(f"❌ Failed to verify final state: {e}")
            return False
        
        print("\n6. Cleaning up session...")
        try:
            cleanup_payload = {"args": {"rk_session_id": rk_session_id}}
            await session.call_tool("cleanup_session", cleanup_payload)
            print("✓ Session cleaned up")
        except Exception as e:
            print(f"⚠ Failed to clean up session: {e}")
        
        print("\n" + "=" * 50)
        if success:
            print("🎉 Integration test PASSED!")
            print("The MCP agent filesystem RL example is working correctly.")
        else:
            print("❌ Integration test FAILED!")
        
        return success


def main():
    """Run the test."""
    try:
        success = asyncio.run(test_filesystem_rl_scenario())
        return 0 if success else 1
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())