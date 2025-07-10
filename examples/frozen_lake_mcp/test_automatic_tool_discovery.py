#!/usr/bin/env python3
"""
Test script to verify automatic tool discovery in the multi-environment proxy

This script tests the automatic tool discovery functionality by:
1. Starting a backend server
2. Testing MCP client connection to the backend
3. Verifying tool discovery works correctly
4. Testing tool registration and proxying
"""

import asyncio
import json
import logging
import subprocess
import time
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ToolDiscoveryTester:
    """Test automatic tool discovery functionality."""

    def __init__(self, backend_port: int = 9999):
        self.backend_port = backend_port
        self.backend_process = None

    async def start_backend_server(self):
        """Start a backend server for testing."""
        logger.info(f"Starting backend server on port {self.backend_port}")

        # Start the backend server
        self.backend_process = subprocess.Popen(
            ["python", "server.py", "--port", str(self.backend_port), "--seed", "42"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for server to start
        await asyncio.sleep(5)

        # Check if server is running
        if self.backend_process.poll() is not None:
            stdout, stderr = self.backend_process.communicate()
            logger.error(
                f"Backend server failed to start:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
            )
            return False

        logger.info(
            f"✅ Backend server started successfully on port {self.backend_port}"
        )
        return True

    async def test_mcp_client_connection(self):
        """Test MCP client connection to backend server."""
        logger.info("Testing MCP client connection to backend server...")

        try:
            url = f"http://localhost:{self.backend_port}/mcp"

            async with streamablehttp_client(url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    logger.info("✅ MCP client connection successful")

                    # Test tool discovery
                    logger.info("Testing tool discovery...")
                    tools_response = await session.list_tools()

                    logger.info(
                        f"✅ Tool discovery successful: found {len(tools_response.tools)} tools"
                    )

                    # Display discovered tools
                    for tool in tools_response.tools:
                        logger.info(f"  Tool: {tool.name}")
                        logger.info(f"    Description: {tool.description}")
                        logger.info(f"    Input Schema: {tool.inputSchema}")

                    return tools_response.tools

        except Exception as e:
            logger.error(f"❌ MCP client connection failed: {e}")
            return None

    async def test_tool_call(self, tools):
        """Test making a tool call to the backend server."""
        if not tools:
            logger.error("No tools available for testing")
            return False

        logger.info("Testing tool call to backend server...")

        try:
            url = f"http://localhost:{self.backend_port}/mcp"

            async with streamablehttp_client(url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    # Find the lake_move tool
                    lake_move_tool = None
                    for tool in tools:
                        if tool.name == "lake_move":
                            lake_move_tool = tool
                            break

                    if not lake_move_tool:
                        logger.error("❌ lake_move tool not found")
                        return False

                    # Test tool call
                    logger.info("Making tool call: lake_move with action='RIGHT'")
                    result = await session.call_tool("lake_move", {"action": "RIGHT"})

                    logger.info(f"✅ Tool call successful")
                    logger.info(f"  Result: {result}")

                    # Check if result has expected structure
                    if hasattr(result, "content") and result.content:
                        content = result.content[0]
                        if hasattr(content, "text"):
                            response_data = json.loads(content.text)
                            logger.info(f"  Parsed response: {response_data}")

                            # Verify expected fields
                            if "position" in response_data and "grid" in response_data:
                                logger.info("✅ Tool call returned expected game state")
                                return True
                            else:
                                logger.error(
                                    "❌ Tool call response missing expected fields"
                                )
                                return False

                    return False

        except Exception as e:
            logger.error(f"❌ Tool call failed: {e}")
            return False

    async def test_proxy_tool_discovery(self):
        """Test the proxy server's automatic tool discovery."""
        logger.info("Testing proxy server tool discovery...")

        try:
            # Import the proxy functionality
            from reward_kit.mcp.multi_environment_proxy import MultiEnvironmentProxy

            # Create proxy instance
            proxy = MultiEnvironmentProxy(
                server_script_path="server.py",
                requirements_path="requirements.txt",
                port_range=(10000, 11000),
                max_concurrent_envs=1,
            )

            # Test tool discovery method
            logger.info("Testing _discover_and_register_tools method...")
            await proxy._discover_and_register_tools(self.backend_port)

            logger.info("✅ Proxy tool discovery completed successfully")

            # Check registered tools
            logger.info(f"Registered tools: {proxy.registered_tools}")

            if "lake_move" in proxy.registered_tools:
                logger.info("✅ lake_move tool was automatically registered")
                return True
            else:
                logger.error("❌ lake_move tool was not registered")
                return False

        except Exception as e:
            logger.error(f"❌ Proxy tool discovery failed: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

    def cleanup(self):
        """Clean up resources."""
        if self.backend_process:
            logger.info("Stopping backend server...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
                self.backend_process.wait()
            logger.info("✅ Backend server stopped")


async def main():
    """Main test function."""
    logger.info("🧪 Testing Automatic Tool Discovery")
    logger.info("=" * 50)

    tester = ToolDiscoveryTester()

    try:
        # Test 1: Start backend server
        if not await tester.start_backend_server():
            logger.error("❌ Failed to start backend server")
            return False

        # Test 2: Test MCP client connection and tool discovery
        tools = await tester.test_mcp_client_connection()
        if not tools:
            logger.error("❌ Failed to discover tools from backend")
            return False

        # Test 3: Test tool call functionality
        if not await tester.test_tool_call(tools):
            logger.error("❌ Failed to make tool call to backend")
            return False

        # Test 4: Test proxy tool discovery
        if not await tester.test_proxy_tool_discovery():
            logger.error("❌ Failed to test proxy tool discovery")
            return False

        logger.info("🎉 All tests passed!")
        logger.info("✅ Automatic tool discovery is working correctly")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

    finally:
        tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
