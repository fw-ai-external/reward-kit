#!/usr/bin/env python3
"""
Simple test to verify session creation in the multi-environment proxy.
"""

import asyncio
import json
from pathlib import Path

import aiohttp


async def test_session_creation():
    """Test that the proxy server can create sessions properly."""
    print("🧪 Testing session creation...")

    # Test resource request that should trigger session creation
    proxy_url = "http://localhost:8090/mcp"

    async with aiohttp.ClientSession() as session:
        try:
            # Try to get initial state resource
            async with session.get(
                f"{proxy_url}/resources/game://initial_state"
            ) as response:
                print(f"Response status: {response.status}")
                content = await response.text()
                print(f"Response content: {content}")

                if response.status == 200:
                    try:
                        data = json.loads(content)
                        if "error" in data:
                            print(f"❌ Session creation failed: {data['error']}")
                            return False
                        else:
                            print("✅ Session created successfully!")
                            print(f"Initial state: {data}")
                            return True
                    except json.JSONDecodeError:
                        print(
                            f"✅ Got non-JSON response (might be game state): {content[:100]}"
                        )
                        return True
                else:
                    print(f"❌ HTTP Error: {response.status}")
                    return False

        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False


async def main():
    """Main test function."""
    print("🚀 Starting session creation test...")
    success = await test_session_creation()

    if success:
        print("🎉 Session creation test passed!")
    else:
        print("💥 Session creation test failed!")

    return success


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
