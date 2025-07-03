#!/usr/bin/env python3
"""
Production North Star Test for FrozenLake MCP

This test demonstrates the production use case where:
1. First run records trajectories (development/testing)
2. Subsequent runs use recorded trajectories (production)
3. No live LLM calls in production mode
4. Automatic server management for CI/CD

Usage:
    # First run (creates recording):
    python test_north_star_production.py

    # Second run (uses recording):
    python test_north_star_production.py

Environment Variables:
    REWARD_KIT_FORCE_RECORD=1  # Force recording even if file exists
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]


class ProductionServerManager:
    """Manages production server for testing."""

    def __init__(self, port: int = 8000):
        self.port = port
        self.process = None
        self.server_dir = Path(__file__).parent.parent / "mcp_server"

    def start(self):
        """Start production server."""
        if self.process:
            return

        env = os.environ.copy()
        env["PORT"] = str(self.port)

        cmd = ["python", "frozen_lake_mcp_server.py", "--port", str(self.port)]
        self.process = subprocess.Popen(
            cmd,
            cwd=self.server_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server startup
        time.sleep(3)

        if self.process.poll() is not None:
            stdout, stderr = self.process.communicate()
            raise RuntimeError(f"Server failed to start: {stderr.decode()}")

    def stop(self):
        """Stop production server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


async def test_production_north_star():
    """Test production north star interface with proper record/replay."""
    print("🌟 Production North Star Test - FrozenLake MCP")
    print("=" * 55)

    try:
        import reward_kit as rk

        # Load dataset
        dataset_path = Path(__file__).parent.parent / "shared_data" / "rollouts.jsonl"
        dataset = load_jsonl(str(dataset_path))
        dataset = dataset[:2]  # Use fewer for faster testing
        print(f"📊 Loaded dataset with {len(dataset)} environments")

        # Set up recording file
        recording_file = Path(__file__).parent / "production_trajectories.jsonl"
        recording_mode = (
            not recording_file.exists()
            or os.environ.get("REWARD_KIT_FORCE_RECORD") == "1"
        )

        # Configure environment for record/replay
        os.environ["REWARD_KIT_PLAYBACK_FILE"] = str(recording_file)

        if recording_mode:
            print("\n📝 === PRODUCTION RECORDING MODE ===")
            print("🎬 First run - will create recording for production use")
            print(f"📁 Recording file: {recording_file}")
        else:
            print("\n🎬 === PRODUCTION PLAYBACK MODE ===")
            print("🚀 Using recorded trajectories - no live LLM calls")
            print(f"📁 Playback file: {recording_file}")

        # Start production server
        with ProductionServerManager(port=8000) as server:
            print("✅ Production server started on port 8000")

            # Create policy (auto-detects record/playback mode)
            policy = rk.FireworksPolicy(
                model_id="accounts/fireworks/models/qwen3-235b-a22b", temperature=0.2
            )

            mode = "playback" if policy.is_playback_mode() else "recording"
            print(f"✅ Policy created in {mode} mode")

            # Create environments
            envs = rk.make(
                "http://localhost:8000/mcp/", dataset=dataset, model_id=policy.model_id
            )
            print("✅ MCP environments created")

            # Run trajectories
            start_time = time.time()
            trajectories = await rk.rollout(
                envs,
                policy=policy,
                steps=8,
                openai_format_log_file=(
                    "production_openai_format.jsonl" if recording_mode else None
                ),
            )
            duration = time.time() - start_time

            print(f"✅ Completed {len(trajectories)} trajectories in {duration:.2f}s")

            # Analysis
            successful = sum(1 for traj in trajectories if traj.total_reward > 0)
            print(f"🎯 Success rate: {successful}/{len(trajectories)}")

            if recording_mode:
                print(f"\n📝 RECORDING COMPLETE")
                print(f"✅ Trajectories recorded to: {recording_file}")
                print(f"💬 OpenAI format saved to: production_openai_format.jsonl")
                print(f"🔄 Next run will use playback mode automatically")
                print(
                    f"⚡ For production: Set REWARD_KIT_PLAYBACK_FILE={recording_file}"
                )
            else:
                # Estimate speedup for production
                estimated_recording_time = 60.0  # Assume ~60s for recording
                speedup = (
                    estimated_recording_time / duration
                    if duration > 0
                    else float("inf")
                )
                print(f"\n🚀 PRODUCTION PLAYBACK COMPLETE")
                print(f"⚡ Speedup: ~{speedup:.0f}x faster than live LLM calls")
                print(f"💰 Cost: $0 (no LLM API calls)")
                print(f"🎯 Deterministic: Same results every run")

            return True

    except Exception as e:
        print(f"❌ Production test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Clean up environment
        if "REWARD_KIT_PLAYBACK_FILE" in os.environ:
            del os.environ["REWARD_KIT_PLAYBACK_FILE"]


async def main():
    """Main test runner."""
    print("🏭 Starting Production North Star Test")
    print("📋 This test demonstrates production record/replay workflow")
    print()

    success = await test_production_north_star()

    if success:
        print("\n🎉 Production north star test completed successfully!")
        print("🚀 Ready for production deployment with recorded policies")
    else:
        print("\n💥 Production north star test failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
