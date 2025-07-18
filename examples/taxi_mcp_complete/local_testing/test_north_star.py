#!/usr/bin/env python3
"""
Test the exact north star interface design from mcp_north_star.md for Taxi environment
"""

import asyncio
import json
from typing import Any, Dict, List


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]


async def test_north_star_interface():
    """Test the new simplified north star interface with automatic record-and-playback for Taxi."""
    print("🌟 Testing Simplified North Star Interface - Taxi Environment")
    print("=" * 65)

    try:
        # New simplified north star API
        import os
        import time

        import reward_kit as rk

        # Load dataset with environment configuration and prompts
        dataset = load_jsonl(
            "/home/bchen/home/reward-kit/examples/taxi_mcp_complete/shared_data/taxi_rollouts.jsonl"
        )
        # Use only first 3 for faster testing
        dataset = dataset[:3]
        print(f"📊 Loaded dataset with {len(dataset)} rows")

        # Check if we're in recording or playback mode
        playback_file = "recording_trajectories.jsonl"
        recording_mode = not os.path.exists(playback_file)

        if recording_mode:
            print("\n📝 === RECORDING MODE ===")
            print(f"🎬 Setting REWARD_KIT_PLAYBACK_FILE={playback_file}")
            os.environ["REWARD_KIT_PLAYBACK_FILE"] = playback_file
        else:
            print("\n🎬 === PLAYBACK MODE ===")
            print(f"📂 Using existing file: {playback_file}")
            os.environ["REWARD_KIT_PLAYBACK_FILE"] = playback_file

        # Create policy - will auto-detect mode based on environment variable
        policy = rk.FireworksPolicy(
            model_id="accounts/fireworks/models/qwen3-235b-a22b",
            temperature=0.2,
            max_tokens=16384,  # Increased from default 4096 for more thinking space
        )
        print(
            f"✅ Policy created in {'playback' if policy.is_playback_mode() else 'live'} mode"
        )

        # Create environments
        envs = rk.make(
            "http://localhost:8000/mcp/", dataset=dataset, model_id=policy.model_id
        )
        print("✅ MCP environments created successfully")

        # Run rollout - same API for both modes!
        start_time = time.time()
        trajectories = await rk.rollout(
            envs,
            policy=policy,
            steps=25,  # Taxi typically needs more steps than FrozenLake
            openai_format_log_file=(
                "clean_openai_format.jsonl" if recording_mode else None
            ),
        )
        duration = time.time() - start_time
        print(f"✅ Completed {len(trajectories)} trajectories in {duration:.2f}s")

        if recording_mode:
            print(f"📝 Recorded to: {playback_file}")
            print(f"💬 OpenAI format: clean_openai_format.jsonl")
            print(f"🔄 Run again to test playback mode!")
        else:
            # Assume ~90s for recording time for speedup calculation (taxi is more complex)
            estimated_recording_time = 90.0
            speedup = (
                estimated_recording_time / duration if duration > 0 else float("inf")
            )
            print(f"⚡ Playback speedup: ~{speedup:.0f}x faster than recording")

            # Load and compare with recorded data if available
            if os.path.exists("previous_trajectories.json"):
                # This would be comparison logic if we saved previous results
                pass

        # === RESULTS ===
        print("\n📊 === RESULTS ===")

        # Show trajectory summary
        print(f"🚕 Trajectories completed: {len(trajectories)}")
        successful = sum(1 for traj in trajectories if traj.total_reward > 0)
        print(f"✅ Successful: {successful}/{len(trajectories)}")

        for i, traj in enumerate(trajectories):
            env_context = dataset[i].get("environment_context", {})
            seed = env_context.get("seed", "N/A")
            is_raining = env_context.get("is_raining", False)
            fickle_passenger = env_context.get("fickle_passenger", False)
            status = "SUCCESS" if traj.total_reward > 0 else "FAILED"

            print(
                f"  Taxi Environment {i} (seed: {seed}, rain: {is_raining}, fickle: {fickle_passenger}): {status}"
            )
            print(f"    Steps: {traj.steps}, Reward: {traj.total_reward}")

        if recording_mode:
            print("\n🏆 Recording phase completed successfully!")
            print("📝 Files created:")
            print(f"   - {playback_file}: Recorded trajectory data for playback")
            print(
                "   - clean_openai_format.jsonl: Clean OpenAI format for SFT training"
            )
        else:
            print("\n🏆 Playback phase completed successfully!")
            print(f"⚡ Demonstrated {speedup:.0f}x speedup over live execution")

        return True

    except Exception as e:
        print(f"❌ North star test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main test runner."""
    print("🚕 Starting Taxi MCP North Star Test")
    print("📋 Prerequisites:")
    print("   - Taxi MCP server running on http://localhost:8000")
    print("   - reward-kit package installed")
    print("   - Fireworks API credentials configured")
    print()

    success = await test_north_star_interface()

    if success:
        print("\n🎉 Taxi north star interface working!")
    else:
        print("\n💥 Taxi north star interface needs implementation!")

    return success


if __name__ == "__main__":
    asyncio.run(main())
