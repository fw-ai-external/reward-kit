#!/usr/bin/env python3
"""
Simple FastMCP Server for Testing

This is the most basic MCP server implementation to test our MCP setup.
It includes:
- Basic resources (game://initial_state, control://reward)
- Simple tools (echo_tool, math_tool)
- Minimal state management

Usage:
    python simple_mcp_server.py --port 9000
"""

import argparse
import asyncio
import json
import logging
import os
from typing import Any, Dict

from mcp.server.fastmcp import Context, FastMCP


class SimpleMCPServer:
    """Minimal MCP server for testing basic functionality."""

    def __init__(self, name: str = "simple-mcp-test"):
        self.mcp = FastMCP(name)
        self.state = {"counter": 0, "game_state": "initial", "reward": 0.0}
        self.logger = logging.getLogger(__name__)
        self._register_resources()
        self._register_tools()

    def _register_resources(self):
        """Register basic MCP resources."""

        @self.mcp.resource("game://initial_state")
        async def get_initial_state() -> str:
            """Return the initial game state."""
            self.logger.info("Resource request: game://initial_state")
            state_data = {
                "state": self.state["game_state"],
                "counter": self.state["counter"],
                "timestamp": "2025-01-01T00:00:00Z",
            }
            return json.dumps(state_data)

        @self.mcp.resource("control://reward")
        async def get_reward() -> str:
            """Return current reward value."""
            self.logger.info("Resource request: control://reward")
            reward_data = {
                "reward": self.state["reward"],
                "counter": self.state["counter"],
            }
            return json.dumps(reward_data)

        @self.mcp.resource("control://status")
        async def get_status() -> str:
            """Return current status."""
            self.logger.info("Resource request: control://status")
            status_data = {
                "terminated": self.state["counter"] >= 10,
                "truncated": False,
                "step_count": self.state["counter"],
            }
            return json.dumps(status_data)

    def _register_tools(self):
        """Register basic MCP tools."""

        @self.mcp.tool(
            name="echo_tool", description="Echo back the input message with a counter."
        )
        async def echo_tool(ctx: Context, message: str) -> Dict[str, Any]:
            """Simple echo tool for testing."""
            self.logger.info(f"Tool call: echo_tool with message: {message}")
            self.state["counter"] += 1

            return {
                "echoed_message": message,
                "counter": self.state["counter"],
                "timestamp": "2025-01-01T00:00:00Z",
            }

        @self.mcp.tool(name="math_tool", description="Perform simple math operations.")
        async def math_tool(
            ctx: Context, operation: str, a: float, b: float
        ) -> Dict[str, Any]:
            """Simple math tool for testing."""
            self.logger.info(
                f"Tool call: math_tool with {operation}: {a} {operation} {b}"
            )
            self.state["counter"] += 1

            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                result = a / b if b != 0 else float("inf")
            else:
                result = 0.0

            # Update reward based on result
            self.state["reward"] += result * 0.1

            return {
                "operation": operation,
                "operands": [a, b],
                "result": result,
                "counter": self.state["counter"],
                "new_reward": self.state["reward"],
            }

    def run(self, transport: str = "streamable-http"):
        """Run the MCP server."""
        self.logger.info(f"Starting Simple MCP Server with transport: {transport}")
        self.mcp.run(transport=transport)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple MCP Test Server")
    parser.add_argument(
        "--transport",
        choices=["streamable-http", "stdio"],
        default="streamable-http",
        help="Transport protocol to use",
    )
    parser.add_argument(
        "--port", type=int, default=9000, help="Port for HTTP transport"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Set environment variable for HTTP port (required by FastMCP)
    if args.transport == "streamable-http":
        os.environ["FASTMCP_PORT"] = str(args.port)

    # Create and run server
    server = SimpleMCPServer()

    print(f"🚀 Starting Simple MCP Test Server on port {args.port}")
    print(f"📡 Transport: {args.transport}")
    print(f"📋 Log level: {args.log_level}")
    print(
        f"🔗 Available resources: game://initial_state, control://reward, control://status"
    )
    print(f"🛠️ Available tools: echo_tool, math_tool")

    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
