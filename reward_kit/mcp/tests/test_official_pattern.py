#!/usr/bin/env python3
"""
Test Official MCP Client Pattern

This script tests the official MCP client pattern from the SDK readme:
- Using async context managers for both streamablehttp_client and ClientSession
- Testing on both our simple server and production server

Usage:
    python test_official_pattern.py --port 9000
"""

import argparse
import asyncio
import logging

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import AnyUrl


async def test_official_mcp_pattern(port: int, log_level: str = "INFO"):
    """Test the official MCP client pattern from SDK readme."""

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    url = f"http://localhost:{port}/mcp"
    logger.info(f"Testing official MCP pattern to {url}")

    try:
        # Use the official pattern: async context managers for both
        logger.info(
            "Step 1: Creating streamablehttp_client with async context manager..."
        )
        async with streamablehttp_client(url) as (read_stream, write_stream, _):
            logger.info("✅ Streamablehttp_client created successfully")

            logger.info("Step 2: Creating ClientSession with async context manager...")
            async with ClientSession(read_stream, write_stream) as session:
                logger.info("✅ ClientSession created successfully")

                logger.info("Step 3: Initializing connection...")
                await asyncio.wait_for(session.initialize(), timeout=10.0)
                logger.info("✅ Connection initialized successfully!")

                # Test resource reading
                logger.info("Step 4: Testing resource reading...")
                try:
                    resource_url = AnyUrl("game://initial_state")
                    result = await asyncio.wait_for(
                        session.read_resource(resource_url), timeout=5.0
                    )
                    logger.info(f"✅ Resource read successful: {result}")

                    if hasattr(result, "contents") and result.contents:
                        content = result.contents[0]
                        if hasattr(content, "text"):
                            logger.info(f"📄 Resource content: {content.text}")

                except Exception as e:
                    logger.error(f"❌ Resource read failed: {e}")

                # Test tool calls
                logger.info("Step 5: Testing tool calls...")
                try:
                    result = await asyncio.wait_for(
                        session.call_tool(
                            "echo_tool", {"message": "Hello from official pattern!"}
                        ),
                        timeout=5.0,
                    )
                    logger.info(f"✅ Tool call successful: {result}")

                except Exception as e:
                    logger.error(f"❌ Tool call failed: {e}")

                # Test control plane resources
                logger.info("Step 6: Testing control plane resources...")
                try:
                    for resource_name in ["control://reward", "control://status"]:
                        resource_url = AnyUrl(resource_name)
                        result = await asyncio.wait_for(
                            session.read_resource(resource_url), timeout=5.0
                        )
                        if hasattr(result, "contents") and result.contents:
                            content = result.contents[0]
                            if hasattr(content, "text"):
                                logger.info(f"✅ {resource_name}: {content.text}")

                except Exception as e:
                    logger.error(f"❌ Control plane resource failed: {e}")

        logger.info("🎉 Official MCP pattern test completed successfully!")
        return True

    except asyncio.TimeoutError:
        logger.error("❌ Timeout during official MCP pattern test")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during official MCP pattern test: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Official MCP Client Pattern")
    parser.add_argument(
        "--port", type=int, default=9000, help="Port of the MCP server to test"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    print(f"🔍 Testing official MCP pattern to localhost:{args.port}")
    print(f"📋 Log level: {args.log_level}")

    success = asyncio.run(test_official_mcp_pattern(args.port, args.log_level))

    if success:
        print("✅ Test completed successfully!")
        exit(0)
    else:
        print("❌ Test failed!")
        exit(1)


if __name__ == "__main__":
    main()
