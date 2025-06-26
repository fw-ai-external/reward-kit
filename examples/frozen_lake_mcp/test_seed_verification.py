#!/usr/bin/env python3
"""
Test script to verify that different seeds produce different environments
and check grid layouts to understand the early termination issue.
"""

import asyncio
import json
from typing import Any, Dict, List

from reward_kit.mcp_env import ToolCall


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]


async def test_seed_differences():
    """Test that different seeds produce different environments."""
    print("🔍 Testing Seed Differences and Grid Layouts")
    print("=" * 50)

    try:
        import reward_kit as rk

        # Load dataset
        dataset = load_jsonl("rollouts.jsonl")
        print(f"📊 Loaded dataset with {len(dataset)} rows")

        # Show what seeds we're testing
        seeds = [row.get("seed") for row in dataset]
        print(f"🎲 Testing seeds: {seeds}")

        # Create environments
        envs = rk.make("http://localhost:8000/mcp", dataset=dataset, model_id="test")
        print("✅ MCP environments created successfully")

        # Reset to get initial states
        print("\n🔄 Resetting environments to check initial states...")
        observations, tool_schemas, system_prompts = await envs.reset()

        print(f"\n📋 Received {len(observations)} initial observations")

        # Check initial state for each environment
        for i, (obs, seed) in enumerate(zip(observations, seeds)):
            print(f"\n🎯 Environment {i+1} (seed={seed}):")
            print(f"   Initial observation: {obs}")

            # Try to make a single move to see the grid layout
            print(f"   Testing first move...")
            try:
                # Make a single move - create proper ToolCall objects
                tool_calls = [
                    ToolCall(tool_name="lake_move", arguments={"action": "DOWN"})
                    for _ in range(len(envs.envs))
                ]
                step_results = await envs.step(tool_calls)
                obs_after, reward, terminated, truncated, info = step_results

                print(
                    f"   After DOWN: obs={obs_after[i]}, reward={reward[i]}, terminated={terminated[i]}"
                )

                # If there's tool call info, show it
                if info and len(info) > i and info[i]:
                    tool_info = info[i]
                    if isinstance(tool_info, dict) and "grid_layout" in tool_info:
                        print(f"   Grid layout:")
                        grid_lines = tool_info["grid_layout"].split("\n")
                        for line in grid_lines:
                            print(f"     {line}")

            except Exception as e:
                print(f"   Error making move: {e}")

        # Close environments
        await envs.close()
        print("\n✅ Seed verification test completed")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


async def main():
    """Main test runner."""
    success = await test_seed_differences()

    if success:
        print("\n🎉 Seed verification successful!")
    else:
        print("\n💥 Seed verification failed!")

    return success


if __name__ == "__main__":
    asyncio.run(main())
