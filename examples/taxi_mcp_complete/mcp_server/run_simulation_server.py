#!/usr/bin/env python3
"""
Standalone runner for the Taxi simulation server.
This bypasses any CLI argument parsing issues and directly starts the server.
"""

import os
import sys

# Add project root to Python path
project_root = os.path.join(os.path.dirname(__file__), "../../..")
sys.path.insert(0, project_root)

# Import required modules
from examples.taxi_mcp_complete.mcp_server import taxi_mcp_server
from examples.taxi_mcp_complete.mcp_server.simulation_server import TaxiSimulation


def main():
    """Run the simulation server with FastMCP."""

    print("🚀 Starting FrozenLake Simulation Server...")
    print(f"📁 Project root: {project_root}")

    # Create the simulation server
    server = TaxiSimulation(
        "Taxi-Simulation-Local",
        production_server_app=taxi_mcp_server.app,
    )

    print("✅ Server instance created")
    print("🌐 Starting server on http://localhost:8001")
    print("📡 MCP endpoint will be available at http://localhost:8001")
    print("🛑 Press Ctrl+C to stop")

    # The simulation server already creates its own FastMCP app internally
    # We just need to run it with the correct host/port
    server.mcp_server.run(host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
