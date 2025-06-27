#!/usr/bin/env python3
"""
Simple MCP connection test to debug issues.
"""

import asyncio
import json


async def test_simple_connection():
    """Simple test to verify local MCP connection works."""
    print("🔌 Testing basic local MCP connection...")
    print("-" * 40)

    try:
        # Import MCP client directly for basic connectivity test
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        local_url = "http://localhost:8000/mcp/"
        print(f"📡 Connecting to: {local_url}")

        async with streamablehttp_client(local_url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                print("✅ MCP session created, initializing...")
                await session.initialize()
                print("✅ Local MCP session initialized successfully")

                # List available tools
                tools_response = await session.list_tools()
                print(f"✅ Found {len(tools_response.tools)} tools on local server:")
                for tool in tools_response.tools:
                    print(f"   - {tool.name}: {tool.description}")

                # List available resources
                resources_response = await session.list_resources()
                print(
                    f"✅ Found {len(resources_response.resources)} resources on local server:"
                )
                for resource in resources_response.resources:
                    print(f"   - {resource.uri}: {resource.description}")

                # Try reading initial state resource
                if resources_response.resources:
                    initial_state_uri = "game://frozen_lake/initial_state"
                    try:
                        content = await session.read_resource(initial_state_uri)
                        print(f"✅ Read initial state from local server")
                        # Parse and display initial state - fix the content access
                        state_content = (
                            content.contents[0].text if content.contents else ""
                        )
                        state_data = json.loads(state_content)
                        print(f"   Position: {state_data.get('position')}")
                        print(f"   Grid layout:")
                        for line in state_data.get("grid_layout", "").split("\n"):
                            print(f"     {line}")
                    except Exception as e:
                        print(f"⚠️  Could not read initial state: {e}")

                # Try making a move
                if tools_response.tools:
                    try:
                        result = await session.call_tool(
                            "lake_move", {"action": "DOWN"}
                        )
                        print(f"✅ Made move on local server")
                        # Fix the result content access
                        result_content = (
                            result.content[0].text if result.content else ""
                        )
                        result_data = json.loads(result_content)
                        print(f"   New position: {result_data.get('position')}")
                        print(f"   Reward: {result_data.get('reward')}")
                        print(f"   Terminated: {result_data.get('terminated')}")
                    except Exception as e:
                        print(f"⚠️  Could not make move: {e}")

        print("🎉 Local MCP connection test passed!")
        return True

    except Exception as e:
        print(f"❌ Local connection test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main test runner for local setup."""
    print("🧪 LOCAL MCP CONNECTION TEST")
    print("=" * 40)
    print("🎯 Purpose: Test basic MCP connection to local server")
    print("📡 Server: http://localhost:8000/mcp/")
    print("=" * 40)

    success = await test_simple_connection()

    if success:
        print("\n🎉 Connection test passed! Ready for north star test.")
    else:
        print("\n💥 Connection test failed. Fix connection before proceeding.")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
