#!/usr/bin/env python3
"""
Test MCP Initialize Steps

This script tests each step of the initialize() process separately to identify
exactly where the hang occurs.

Usage:
    python test_initialize_steps.py --port 9000
"""

import argparse
import asyncio
import logging
from contextlib import AsyncExitStack

import mcp.types as types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def test_initialize_steps(port: int, log_level: str = "DEBUG"):
    """Test each step of MCP initialize() separately."""

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    url = f"http://localhost:{port}/mcp"
    logger.info(f"Testing MCP initialize steps to {url}")

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

        # Step 3a: Manually send initialize request (first part of initialize())
        logger.info("Step 3a: Sending initialize request...")
        try:
            result = await asyncio.wait_for(
                mcp_session.send_request(
                    types.ClientRequest(
                        types.InitializeRequest(
                            method="initialize",
                            params=types.InitializeRequestParams(
                                protocolVersion=types.LATEST_PROTOCOL_VERSION,
                                capabilities=types.ClientCapabilities(
                                    sampling=None,
                                    elicitation=None,
                                    experimental=None,
                                    roots=None,
                                ),
                                clientInfo=types.Implementation(
                                    name="test-client", version="1.0.0"
                                ),
                            ),
                        )
                    ),
                    types.InitializeResult,
                ),
                timeout=10.0,
            )
            logger.info(f"✅ Initialize request successful: {result}")
        except asyncio.TimeoutError:
            logger.error("❌ Timeout during initialize request")
            return False
        except Exception as e:
            logger.error(f"❌ Initialize request failed: {e}")
            return False

        # Step 3b: Validate protocol version (second part of initialize())
        logger.info("Step 3b: Validating protocol version...")
        if result.protocolVersion not in types.SUPPORTED_PROTOCOL_VERSIONS:
            logger.error(f"❌ Unsupported protocol version: {result.protocolVersion}")
            return False
        logger.info(f"✅ Protocol version validated: {result.protocolVersion}")

        # Step 3c: Send initialized notification (third part of initialize())
        logger.info("Step 3c: Sending initialized notification...")
        try:
            await asyncio.wait_for(
                mcp_session.send_notification(
                    types.ClientNotification(
                        types.InitializedNotification(
                            method="notifications/initialized"
                        )
                    )
                ),
                timeout=10.0,
            )
            logger.info("✅ Initialized notification sent successfully")
        except asyncio.TimeoutError:
            logger.error("❌ Timeout during initialized notification")
            return False
        except Exception as e:
            logger.error(f"❌ Initialized notification failed: {e}")
            return False

        logger.info("🎉 All initialize steps completed successfully!")

        # Step 4: Test that session is now usable
        logger.info("Step 4: Testing session after initialization...")
        try:
            from pydantic import AnyUrl

            resource_url = AnyUrl("game://initial_state")
            result = await asyncio.wait_for(
                mcp_session.read_resource(resource_url), timeout=5.0
            )
            logger.info("✅ Session is usable after manual initialization")

        except Exception as e:
            logger.error(f"❌ Session not usable after initialization: {e}")

        return True

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False
    finally:
        await exit_stack.aclose()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test MCP Initialize Steps")
    parser.add_argument(
        "--port", type=int, default=9000, help="Port of the MCP server to test"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="DEBUG",
        help="Logging level",
    )

    args = parser.parse_args()

    print(f"🔍 Testing MCP initialize steps to localhost:{args.port}")
    print(f"📋 Log level: {args.log_level}")

    success = asyncio.run(test_initialize_steps(args.port, args.log_level))

    if success:
        print("✅ Test completed successfully!")
        exit(0)
    else:
        print("❌ Test failed!")
        exit(1)


if __name__ == "__main__":
    main()
