#!/usr/bin/env python3
"""
Test MCP Client Without Initialize

This script tests if we can use the MCP session directly without calling initialize(),
since we've confirmed that protocol negotiation works but initialize() hangs.

Usage:
    python test_mcp_without_initialize.py --port 9000
"""

import argparse
import asyncio
import logging
from contextlib import AsyncExitStack

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import AnyUrl


async def test_mcp_without_initialize(port: int, log_level: str = "INFO"):
    """Test MCP client without calling initialize()."""

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    url = f"http://localhost:{port}/mcp"
    logger.info(f"Testing MCP client without initialize() to {url}")

    exit_stack = AsyncExitStack()

    try:
        # Step 1: Create streamable HTTP client
        logger.info("Step 1: Creating streamable HTTP client...")
        read_stream, write_stream, get_session_id = await asyncio.wait_for(
            exit_stack.enter_async_context(streamablehttp_client(url)), timeout=15.0
        )
        logger.info("✅ Streamable HTTP client created successfully")

        # Step 2: Create MCP client session
        logger.info("Step 2: Creating MCP client session...")
        mcp_session = ClientSession(read_stream, write_stream)
        logger.info("✅ MCP client session created successfully")

        # Step 3: SKIP initialize() and try to use session directly
        logger.info("Step 3: Skipping initialize(), testing direct usage...")

        # Test if we can read resources directly
        logger.info("Step 4: Testing direct resource reading...")
        try:
            resource_url = AnyUrl("game://initial_state")
            result = await asyncio.wait_for(
                mcp_session.read_resource(resource_url), timeout=5.0
            )
            logger.info(f"✅ Direct resource read successful: {result}")

            if hasattr(result, "contents") and result.contents:
                content = result.contents[0]
                if hasattr(content, "text"):
                    logger.info(f"📄 Resource content: {content.text}")

        except Exception as e:
            logger.error(f"❌ Direct resource read failed: {e}")

        # Test if we can call tools directly
        logger.info("Step 5: Testing direct tool calls...")
        try:
            result = await asyncio.wait_for(
                mcp_session.call_tool(
                    "echo_tool", {"message": "Hello without initialize!"}
                ),
                timeout=5.0,
            )
            logger.info(f"✅ Direct tool call successful: {result}")

        except Exception as e:
            logger.error(f"❌ Direct tool call failed: {e}")

        logger.info("🎉 MCP client test without initialize() completed!")
        return True

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False
    finally:
        await exit_stack.aclose()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test MCP Client Without Initialize")
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

    print(f"🔍 Testing MCP client without initialize() to localhost:{args.port}")
    print(f"📋 Log level: {args.log_level}")

    success = asyncio.run(test_mcp_without_initialize(args.port, args.log_level))

    if success:
        print("✅ Test completed successfully!")
        exit(0)
    else:
        print("❌ Test failed!")
        exit(1)


if __name__ == "__main__":
    main()
