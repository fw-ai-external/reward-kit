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
        # No longer need trajectory_file and openai_format_file in policy - we'll use rollout logging
        policy = rk.FireworksPolicy(
            model_id="accounts/fireworks/models/qwen3-235b-a22b",
            temperature=0.2,
        )
        print("✅ Policy created successfully")

        # 1️⃣ create vector of MCP sessions
        envs = rk.make(
            "http://localhost:8000/mcp", dataset=dataset, model_id=policy.model_id
        )
        print("✅ MCP environments created successfully")

        # 2️⃣ parallel tool-calling rollouts with CLEAN LOGGING
        trajectories = await rk.rollout(
            envs,
            policy=policy,
            steps=8,  # Short test
            trajectory_log_file="clean_trajectories.jsonl",  # ✨ NEW: Clean trajectory logging
            openai_format_log_file="clean_openai_format.jsonl",  # ✨ NEW: Clean OpenAI format logging
        )
        print(f"✅ Generated {len(trajectories)} trajectories")

        # Show sample trajectory with seed verification
        if trajectories:
            for i, traj in enumerate(trajectories):
                # Get the seed from the environment_context in the new structure
                env_context = dataset[i].get("environment_context", {})
                seed = env_context.get("seed", "N/A")
                grid_type = env_context.get("grid_type", "N/A")
                print("-" * 20)
                print(f"Trajectory for seed: {seed} (grid: {grid_type})")
                print(
                    f"📝 Sample trajectory: {traj.steps} steps, reward: {traj.total_reward}"
                )
                print(f"   Actions: {traj.actions[:3]}...")
                print(f"   Rewards: {traj.rewards[:3]}...")
                print(
                    f"   Terminated: {traj.terminated}, Duration: {traj.duration:.2f}s"
                )

        print("🏆 North star interface test completed successfully!")
        print("📝 Clean logs created:")
        print("   - clean_trajectories.jsonl: Detailed step-by-step trajectory data")
        print("   - clean_openai_format.jsonl: OpenAI-compatible conversation format")
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
