#!/usr/bin/env python3
"""
Test MCP Client Connection to Simple Server

This script tests if we can establish an MCP client connection to our simple server
and perform basic operations like reading resources and calling tools.

Usage:
    python test_mcp_client_connection.py --port 9000
"""

import argparse
import asyncio
import json
import logging
from contextlib import AsyncExitStack

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import AnyUrl


async def test_mcp_client_connection(port: int, log_level: str = "INFO"):
    """Test MCP client connection to the simple server."""

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    url = f"http://localhost:{port}/mcp"
    logger.info(f"Testing MCP client connection to {url}")

    exit_stack = AsyncExitStack()

    try:
        # Step 1: Create streamable HTTP client
        logger.info("Step 1: Creating streamable HTTP client...")
        try:
            read_stream, write_stream, get_session_id = await asyncio.wait_for(
                exit_stack.enter_async_context(streamablehttp_client(url)), timeout=15.0
            )
            logger.info("✅ Streamable HTTP client created successfully")
        except asyncio.TimeoutError:
            logger.error("❌ Timeout creating streamable HTTP client")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to create streamable HTTP client: {e}")
            return False

        # Step 2: Create MCP client session
        logger.info("Step 2: Creating MCP client session...")
        try:
            mcp_session = ClientSession(read_stream, write_stream)
            logger.info("✅ MCP client session created successfully")
        except Exception as e:
            logger.error(f"❌ Failed to create MCP client session: {e}")
            return False

        # Step 3: Initialize session (this is where the hang occurs in the proxy)
        logger.info("Step 3: Initializing MCP client session...")
        try:
            await asyncio.wait_for(mcp_session.initialize(), timeout=10.0)
            logger.info("✅ MCP client session initialized successfully")
        except asyncio.TimeoutError:
            logger.error("❌ Timeout during MCP session initialization")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize MCP session: {e}")
            return False

        # Step 4: Test resource reading
        logger.info("Step 4: Testing resource reading...")
        try:
            # Test game://initial_state resource
            resource_url = AnyUrl("game://initial_state")
            result = await asyncio.wait_for(
                mcp_session.read_resource(resource_url), timeout=5.0
            )
            logger.info(f"✅ Resource read successful: {result}")

            # Extract and display content
            if hasattr(result, "contents") and result.contents:
                content = result.contents[0]
                if hasattr(content, "text"):
                    logger.info(f"📄 Resource content: {content.text}")

        except asyncio.TimeoutError:
            logger.error("❌ Timeout during resource reading")
        except Exception as e:
            logger.error(f"❌ Failed to read resource: {e}")

        # Step 5: Test tool calls
        logger.info("Step 5: Testing tool calls...")
        try:
            # Test echo_tool
            result = await asyncio.wait_for(
                mcp_session.call_tool(
                    "echo_tool", {"message": "Hello from MCP client test!"}
                ),
                timeout=5.0,
            )
            logger.info(f"✅ Tool call successful: {result}")

        except asyncio.TimeoutError:
            logger.error("❌ Timeout during tool call")
        except Exception as e:
            logger.error(f"❌ Failed to call tool: {e}")

        # Step 6: Test control plane resources
        logger.info("Step 6: Testing control plane resources...")
        try:
            for resource_name in ["control://reward", "control://status"]:
                resource_url = AnyUrl(resource_name)
                result = await asyncio.wait_for(
                    mcp_session.read_resource(resource_url), timeout=5.0
                )
                if hasattr(result, "contents") and result.contents:
                    content = result.contents[0]
                    if hasattr(content, "text"):
                        logger.info(f"✅ {resource_name}: {content.text}")

        except Exception as e:
            logger.error(f"❌ Failed to read control plane resource: {e}")

        logger.info("🎉 MCP client connection test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Unexpected error during MCP client test: {e}")
        return False
    finally:
        await exit_stack.aclose()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test MCP Client Connection")
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

    print(f"🔍 Testing MCP client connection to localhost:{args.port}")
    print(f"📋 Log level: {args.log_level}")

    success = asyncio.run(test_mcp_client_connection(args.port, args.log_level))

    if success:
        print("✅ Test completed successfully!")
        exit(0)
    else:
        print("❌ Test failed!")
        exit(1)


if __name__ == "__main__":
    main()
