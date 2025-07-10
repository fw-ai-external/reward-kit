#!/usr/bin/env python3
"""
Test session creation using proper MCP client protocol.
"""

import asyncio
import json
from contextlib import AsyncExitStack

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Implementation


async def test_mcp_session():
    """Test MCP session creation properly."""
    print("🧪 Testing MCP session creation...")

    proxy_url = "http://localhost:8090/mcp/"

    try:
        async with AsyncExitStack() as stack:
            # Connect using proper MCP client
            read_stream, write_stream, session_info = await stack.enter_async_context(
                streamablehttp_client(proxy_url)
            )

            print(f"📡 Connected to proxy server")
            print(f"🔗 Session info: {session_info}")

            # Create MCP client session
            client_info = Implementation(name="test-client", version="1.0.0")
            client_info._extra = {"seed": 42, "config": {}}  # Add seed for testing

            client = await stack.enter_async_context(
                ClientSession(read_stream, write_stream, client_info=client_info)
            )

            # Initialize the session
            await client.initialize()
            print("✅ MCP session initialized successfully!")

            # Try to get resources
            try:
                resources = await client.list_resources()
                print(
                    f"📋 Available resources: {[r.name for r in resources.resources]}"
                )

                # Try to read initial state resource
                if resources.resources:
                    initial_state_resource = None
                    for resource in resources.resources:
                        if "initial_state" in str(resource.uri):
                            initial_state_resource = resource
                            break

                    if initial_state_resource:
                        print(
                            f"🎮 Reading initial state from: {initial_state_resource.uri}"
                        )
                        content = await client.read_resource(initial_state_resource.uri)
                        print(f"🎯 Initial state content type: {type(content)}")

                        if hasattr(content, "text"):
                            try:
                                game_state = json.loads(content.text)
                                if "error" in game_state:
                                    print(
                                        f"❌ Session creation failed: {game_state['error']}"
                                    )
                                    return False
                                else:
                                    print("✅ Session created successfully!")
                                    print(f"🎮 Game state: {game_state}")
                                    return True
                            except json.JSONDecodeError:
                                print(f"✅ Got initial state: {content.text[:100]}...")
                                return True
                        else:
                            print(f"✅ Got initial state: {content}")
                            return True
                    else:
                        print("❌ No initial_state resource found")
                        return False
                else:
                    print("❌ No resources available")
                    return False

            except Exception as e:
                print(f"❌ Resource request failed: {e}")
                return False

    except Exception as e:
        print(f"❌ MCP connection failed: {e}")
        return False


async def main():
    """Main test function."""
    print("🚀 Starting MCP session test...")
    success = await test_mcp_session()

    if success:
        print("🎉 MCP session test passed!")
    else:
        print("💥 MCP session test failed!")

    return success


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
