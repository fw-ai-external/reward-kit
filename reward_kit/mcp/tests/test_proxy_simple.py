#!/usr/bin/env python3
"""
Simple test for the fixed multi-environment proxy

This script tests the basic functionality of the updated proxy to ensure
the MCP client pattern fix works correctly.

Usage:
    python test_proxy_simple.py
"""

import asyncio
import logging
import subprocess
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import AnyUrl


async def test_simple_proxy():
    """Test basic proxy functionality."""

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Start proxy server
    base_dir = (
        Path(__file__).parent.parent.parent.parent / "examples" / "frozen_lake_mcp"
    )
    server_script = str(base_dir / "server.py")
    requirements_path = str(base_dir / "requirements.txt")

    logger.info("Starting multi-environment proxy server...")
    cmd = [
        "python",
        "-m",
        "reward_kit.mcp.multi_environment_proxy",
        "--server-script",
        server_script,
        "--requirements",
        requirements_path,
        "--port",
        "8091",
        "--max-envs",
        "1",
    ]

    proxy_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for proxy to start
    time.sleep(10)

    if proxy_process.poll() is not None:
        stdout, stderr = proxy_process.communicate()
        logger.error(
            f"Proxy server failed to start:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        )
        return False

    try:
        # Test connection to proxy
        url = "http://localhost:8091/mcp"
        logger.info(f"Testing connection to proxy at {url}")

        async with streamablehttp_client(url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                logger.info("✅ Connected to proxy successfully")

                # Test resource reading
                logger.info("Testing resource reading...")
                resource_url = AnyUrl("game://initial_state")
                result = await session.read_resource(resource_url)
                logger.info(f"✅ Resource read successful: {result}")

                # Test tool call
                logger.info("Testing tool call...")
                tool_result = await session.call_tool("lake_move", {"action": "RIGHT"})
                logger.info(f"✅ Tool call successful: {tool_result}")

        logger.info("🎉 Simple proxy test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Simple proxy test failed: {e}")
        return False

    finally:
        # Clean up proxy server
        if proxy_process.poll() is None:
            proxy_process.terminate()
            try:
                proxy_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proxy_process.kill()


def main():
    """Main entry point."""
    print("🔍 Testing simple proxy functionality...")
    success = asyncio.run(test_simple_proxy())

    if success:
        print("✅ Simple proxy test passed!")
        exit(0)
    else:
        print("❌ Simple proxy test failed!")
        exit(1)


if __name__ == "__main__":
    main()
