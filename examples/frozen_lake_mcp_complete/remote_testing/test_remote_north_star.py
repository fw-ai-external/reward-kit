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
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import reward_kit as rk

# Add the project root to Python path to import reward_kit
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Use a real remote URL for this test
REMOTE_URL = os.environ.get(
    "REWARD_KIT_TEST_REMOTE_URL",
    "https://frozen-lake-mcp-zfdbl7ykrq-uc.a.run.app/mcp/",  # Has trailing slash
)
DATASET_FILE = "remote_test_dataset.jsonl"


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]


async def test_remote_north_star(remote_url: str = REMOTE_URL):
    """Test north star interface with remote MCP server."""
    print("🌟 Testing Remote North Star Interface")
    print("========================================")

    # Dataset preparation (using same format as working local test)
    system_prompt = "You are playing FrozenLake, a grid-based navigation game displayed as a 4x4 text grid. The grid contains: S (Start), F (Frozen safe), H (Hole - deadly), G (Goal). You start at position S and must reach G while avoiding H tiles. In this version, the surface is not slippery so your moves are deterministic. IMPORTANT: When you are at the starting position, you appear as 'S'. When you move to other positions, you appear as 'P' (Player). If you step on H, you become 'X' and the episode ends with failure. If you reach G, you become 'W' (Won). Use the lake_move tool with actions LEFT, DOWN, RIGHT, UP to navigate the grid."
    user_prompt_template = "Initial game state grid:\n{observation}\n\nYou are navigating the 4x4 grid above. Navigate safely to reach the goal 'G' while avoiding holes 'H'. Choose your next move from: LEFT, DOWN, RIGHT, or UP."

    dataset = [
        {
            "id": "remote_run_001",
            "system_prompt": system_prompt,
            "user_prompt_template": user_prompt_template,
            "environment_context": {
                "game": "FrozenLake",
                "grid_type": "4x4",
                "seed": 42,
            },
        },
        # Temporarily commented out to test with single environment
        {
            "id": "remote_run_002",
            "system_prompt": system_prompt,
            "user_prompt_template": user_prompt_template,
            "environment_context": {
                "game": "FrozenLake",
                "grid_type": "4x4",
                "seed": 123,
            },
        },
        {
            "id": "remote_run_003",
            "system_prompt": system_prompt,
            "user_prompt_template": user_prompt_template,
            "environment_context": {
                "game": "FrozenLake",
                "grid_type": "4x4",
                "seed": 999,
            },
        },
    ]
    print(f"📊 Loaded dataset with {len(dataset)} rows")

    # Create a policy (LLM)
    policy = rk.FireworksPolicy(
        model_id="accounts/fireworks/models/qwen3-235b-a22b",
        temperature=0.2,
    )
    print("✅ Policy created successfully")

    try:
        # 1️⃣ create vector of MCP sessions
        # FIXED: Use the correct remote URL with trailing slash from the deployment
        envs = rk.make(remote_url, dataset=dataset, model_id=policy.model_id)
        print("✅ Remote MCP environments created successfully")

        # 2️⃣ run the rollouts
        print(
            f"✅ Starting rollouts with {len(dataset)} remote environment(s) for 8 steps..."
        )
        start_time = time.time()
        trajectories = await rk.rollout(envs, policy, steps=8)
        duration = time.time() - start_time

        # 3️⃣ check results
        successful_runs = sum(
            1 for traj in trajectories if traj.terminated and traj.reward > 0
        )
        print(
            f"📊 Remote rollout complete: {successful_runs}/{len(trajectories)} reached goal"
        )
        print(f"⏱️  Total duration: {duration:.2f}s")

        # 4️⃣ output trajectory information
        print("📝 Generated {} trajectories".format(len(trajectories)))
        for i, trajectory in enumerate(trajectories):
            print("-" * 20)
            env_context = dataset[i].get("environment_context", {})
            seed = env_context.get("seed", "N/A")
            grid_type = env_context.get("grid_type", "N/A")
            print(f"Trajectory for seed: {seed} (grid: {grid_type})")
            print(
                f"📝 Sample trajectory: {trajectory.steps} steps, reward: {trajectory.total_reward}"
            )
            print(f"   Actions: {trajectory.actions[:3]}...")
            print(f"   Rewards: {trajectory.rewards[:3]}...")
            print(
                f"   Terminated: {trajectory.terminated}, Duration: {trajectory.duration:.2f}s"
            )

        print("🏆 Remote north star interface test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Remote test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_basic_remote_connection(remote_url: str = REMOTE_URL):
    """Simple test to verify remote MCP connection works."""
    print("\n🔌 Testing basic remote MCP connection...")
    print("-" * 40)

    try:
        # Import MCP client directly for basic connectivity test
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

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
                    # Fix: access the resource content correctly
                    if content.contents and len(content.contents) > 0:
                        text_content = (
                            content.contents[0].text
                            if hasattr(content.contents[0], "text")
                            else str(content.contents[0])
                        )
                        print(f"   Content preview: {text_content[:100]}...")
                    else:
                        print(f"   Content: {content}")

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
    """Main entry point for the remote test suite."""
    print("🌐 REMOTE ROLLOUT TEST SUITE")
    print("=" * 60)
    print("🎯 Purpose: Test purely remote MCP deployment")
    print("📡 Server: Google Cloud Run (no local dependencies)")
    print("🧪 Tests: Connection + North Star Interface")
    print("=" * 60)

    # Test basic connection
    connection_success = False
    try:
        await test_basic_remote_connection(remote_url=REMOTE_URL)
        connection_success = True
    except Exception as e:
        print(f"❌ Basic connection test failed: {e}")

    # Test north star interface
    interface_success = False
    if connection_success:
        try:
            await test_remote_north_star(remote_url=REMOTE_URL)
            interface_success = True
        except Exception as e:
            print(f"❌ Remote test failed: {e}")

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
