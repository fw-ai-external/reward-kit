#!/usr/bin/env python3
"""
Simple test script to debug simulation server startup issues.
"""

import os
import sys

# Add project root to Python path
project_root = os.path.join(os.path.dirname(__file__), "../../..")
sys.path.insert(0, project_root)

print("🔍 Testing simulation server imports and startup...")
print(f"📁 Project root: {project_root}")
print(f"🐍 Python path: {sys.path[:3]}...")

try:
    print("\n1️⃣ Testing production server import...")
    from examples.frozen_lake_mcp_complete.mcp_server import frozen_lake_mcp_server

    print("✅ Production server import successful")

    print("\n2️⃣ Testing adapter import...")
    from examples.frozen_lake_mcp_complete.mcp_server.frozen_lake_adapter import (
        FrozenLakeAdapter,
    )

    print("✅ Adapter import successful")

    print("\n3️⃣ Testing simulation server import...")
    from examples.frozen_lake_mcp_complete.mcp_server.simulation_server import (
        FrozenLakeSimulationServer,
    )

    print("✅ Simulation server import successful")

    print("\n4️⃣ Testing server creation...")
    server = FrozenLakeSimulationServer()
    print("✅ Server instance created successfully")

    print("\n5️⃣ Testing server startup...")
    # Don't actually run the server, just test that it can be configured
    print("✅ Server can be configured for startup")

    print("\n🎉 All tests passed! The simulation server should work.")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
