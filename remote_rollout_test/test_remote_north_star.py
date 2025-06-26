#!/usr/bin/env python3
"""
Test the exact north star interface design from mcp_north_star.md
REMOTE VERSION: Connects to deployed Google Cloud Run MCP server

This tests a purely remote rollout setup where:
- No local MCP server needed
- Connects directly to Cloud Run deployed server
- Tests full end-to-end remote deployment
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add the project root to Python path to import reward_kit
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]


async def test_remote_north_star_interface():
    """Test the exact north star interface with REMOTE server."""
    print("🌟 Testing Remote North Star Interface")
    print("=" * 50)
    print("🌐 REMOTE MODE: Connecting to Google Cloud Run")
    print("📡 Server URL: https://frozen-lake-mcp-zfdbl7ykrq-uc.a.run.app")
    print("🚀 This tests pure remote deployment (no local server)")
    print("=" * 50)

    try:
        # Exact north star code from the document
        import reward_kit as rk

        # Load dataset with environment configuration and prompts
        dataset = load_jsonl("rollouts.jsonl")
        print(f"📊 Loaded dataset with {len(dataset)} rows")

        # Create general policy (environment-agnostic via tool calling)
        policy = rk.FireworksPolicy(
            model_id="accounts/fireworks/models/qwen3-235b-a22b",
            temperature=0.2,
            trajectory_file="remote_trajectory.jsonl",
            openai_format_file="remote_openai_format.jsonl",
        )
        print("✅ Policy created successfully")

        # 1️⃣ create vector of MCP sessions - REMOTE SERVER
        envs = rk.make(
            "https://frozen-lake-mcp-zfdbl7ykrq-uc.a.run.app/mcp",  # 🌐 REMOTE URL
            dataset=dataset,
            model_id=policy.model_id,
        )
        print("✅ Remote MCP environments created successfully")

        # 2️⃣ parallel tool-calling rollouts
        trajectories = await rk.rollout(envs, policy=policy, steps=8)  # Short test
        print(f"✅ Generated {len(trajectories)} trajectories from REMOTE server")

        # Show sample trajectory
        if trajectories:
            traj = trajectories[0]
            print(
                f"📝 Sample remote trajectory: {traj.steps} steps, reward: {traj.total_reward}"
            )
            print(f"   Actions: {traj.actions[:3]}...")
            print(f"   Rewards: {traj.rewards[:3]}...")
            print(f"   Terminated: {traj.terminated}, Duration: {traj.duration:.2f}s")

        print("🏆 Remote north star interface test completed successfully!")
        print("🌐 Verification: Remote Cloud Run deployment is working!")
        return True

    except Exception as e:
        print(f"❌ Remote north star test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_remote_connection_simple():
    """Simple test to verify remote MCP connection works."""
    print("\n🔌 Testing basic remote MCP connection...")
    print("-" * 40)

    try:
        # Import MCP client directly for basic connectivity test
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        remote_url = "https://frozen-lake-mcp-zfdbl7ykrq-uc.a.run.app/mcp"
        print(f"📡 Connecting to: {remote_url}")

        async with streamablehttp_client(remote_url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                print("✅ Remote MCP session initialized successfully")

                # List available tools
                tools_response = await session.list_tools()
                print(f"✅ Found {len(tools_response.tools)} tools on remote server:")
                for tool in tools_response.tools:
                    print(f"   - {tool.name}: {tool.description}")

                # List available resources
                resources_response = await session.list_resources()
                print(
                    f"✅ Found {len(resources_response.resources)} resources on remote server:"
                )
                for resource in resources_response.resources:
                    print(f"   - {resource.uri}: {resource.description}")

                # Try reading initial state resource
                if resources_response.resources:
                    initial_state_uri = "game://frozen_lake/initial_state"
                    content = await session.read_resource(initial_state_uri)
                    print(f"✅ Read initial state from remote server")
                    print(f"   Content preview: {content.content[:100]}...")

                # Try making a move
                if tools_response.tools:
                    result = await session.call_tool("lake_move", {"action": "DOWN"})
                    print(f"✅ Made move on remote server")
                    print(f"   Result preview: {str(result.content[0])[:100]}...")

        print("🎉 Remote MCP connection test passed!")
        return True

    except Exception as e:
        print(f"❌ Remote connection test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main test runner for remote setup."""
    print("🌐 REMOTE ROLLOUT TEST SUITE")
    print("=" * 60)
    print("🎯 Purpose: Test purely remote MCP deployment")
    print("📡 Server: Google Cloud Run (no local dependencies)")
    print("🧪 Tests: Connection + North Star Interface")
    print("=" * 60)

    # Test 1: Basic remote connection
    connection_success = await test_remote_connection_simple()

    # Test 2: Full north star interface (only if connection works)
    if connection_success:
        print("\n" + "=" * 60)
        interface_success = await test_remote_north_star_interface()
    else:
        print("\n⚠️  Skipping north star test due to connection failure")
        interface_success = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 REMOTE TEST RESULTS SUMMARY")
    print("=" * 60)

    if connection_success:
        print("✅ Remote Connection: PASSED")
    else:
        print("❌ Remote Connection: FAILED")

    if interface_success:
        print("✅ North Star Interface: PASSED")
    else:
        print("❌ North Star Interface: FAILED")

    overall_success = connection_success and interface_success

    if overall_success:
        print("\n🎉 ALL REMOTE TESTS PASSED!")
        print("🌐 Remote Cloud Run MCP deployment is fully functional")
        print("🚀 Ready for production remote rollouts!")
    else:
        print("\n💥 Some remote tests failed")
        print("🔧 Check connection and server deployment")

    return overall_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
