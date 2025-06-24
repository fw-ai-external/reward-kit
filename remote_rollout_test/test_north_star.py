#!/usr/bin/env python3
"""
Test the exact north star interface design from mcp_north_star.md
"""

import asyncio
import json
from typing import Any, Dict, List


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]


async def test_north_star_interface():
    """Test the exact north star interface."""
    print("🌟 Testing North Star Interface")
    print("=" * 40)

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
            trajectory_file="trajectory.jsonl",
            openai_format_file="openai_format.jsonl",
        )
        print("✅ Policy created successfully")

        # 1️⃣ create vector of MCP sessions
        envs = rk.make(
            "http://localhost:8000/mcp", dataset=dataset, model_id=policy.model_id
        )
        print("✅ MCP environments created successfully")

        # 2️⃣ parallel tool-calling rollouts
        trajectories = await rk.rollout(envs, policy=policy, steps=8)  # Short test
        print(f"✅ Generated {len(trajectories)} trajectories")

        # Show sample trajectory
        if trajectories:
            traj = trajectories[0]
            print(
                f"📝 Sample trajectory: {traj.steps} steps, reward: {traj.total_reward}"
            )
            print(f"   Actions: {traj.actions[:3]}...")
            print(f"   Rewards: {traj.rewards[:3]}...")
            print(f"   Terminated: {traj.terminated}, Duration: {traj.duration:.2f}s")

        print("🏆 North star interface test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ North star test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main test runner."""
    success = await test_north_star_interface()

    if success:
        print("\n🎉 North star interface working!")
    else:
        print("\n💥 North star interface needs implementation!")

    return success


if __name__ == "__main__":
    asyncio.run(main())
