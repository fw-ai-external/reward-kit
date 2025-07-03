#!/usr/bin/env python3
"""
Local North Star Test for FrozenLake MCP - Always uses local recordings

This test is designed for development and CI environments where we always want
to use pre-recorded trajectories instead of making live LLM calls.

Environment Variables:
    REWARD_KIT_PLAYBACK_FILE   - Path to recording file (default: ./recordings/frozen_lake_trajectories.jsonl)
    REWARD_KIT_FORCE_RECORD    - Set to "1" to force recording mode

Usage:
    # Use local recordings (default)
    python test_north_star_local.py

    # Force new recording
    REWARD_KIT_FORCE_RECORD=1 python test_north_star_local.py

    # Use custom recording file
    REWARD_KIT_PLAYBACK_FILE=./my_recordings.jsonl python test_north_star_local.py
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
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class LocalServerManager:
    """Manages local MCP server for testing."""

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
            _, stderr = self.process.communicate()
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


async def test_local_north_star():
    """Test north star interface using local recordings."""
    print("🌟 Local North Star Test - FrozenLake MCP")
    print("=" * 50)

    try:
        import reward_kit as rk

        # Load dataset
        dataset_path = Path(__file__).parent.parent / "shared_data" / "rollouts.jsonl"
        dataset = load_jsonl(str(dataset_path))
        dataset = dataset[:2]  # Use fewer for faster testing
        print(f"📊 Loaded dataset with {len(dataset)} environments")

        # Set up recording file path
        default_recording_file = Path("./recordings/frozen_lake_trajectories.jsonl")
        recording_file = Path(
            os.environ.get("REWARD_KIT_PLAYBACK_FILE", str(default_recording_file))
        )

        # Ensure parent directory exists
        recording_file.parent.mkdir(parents=True, exist_ok=True)
        force_record = os.environ.get("REWARD_KIT_FORCE_RECORD") == "1"

        recording_mode = force_record or not recording_file.exists()

        # Configure environment for record/replay
        os.environ["REWARD_KIT_PLAYBACK_FILE"] = str(recording_file)

        if recording_mode:
            print(f"\n📝 === LOCAL RECORDING MODE ===")
            print(f"🎬 Creating local recording for development/CI")
            print(f"📁 Recording file: {recording_file}")
            if force_record:
                print("🔄 Forced recording mode (REWARD_KIT_FORCE_RECORD=1)")
        else:
            print(f"\n🎬 === LOCAL PLAYBACK MODE ===")
            print(f"🚀 Using local recordings - no LLM calls")
            print(f"📁 Playback file: {recording_file}")

        # Start local server
        with LocalServerManager(port=8000):
            print("✅ Local server started on port 8000")

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
            trajectories = await rk.rollout(envs, policy=policy, steps=8)
            duration = time.time() - start_time

            print(f"✅ Completed {len(trajectories)} trajectories in {duration:.2f}s")

            # Analysis
            successful = sum(1 for traj in trajectories if traj.total_reward > 0)
            print(f"🎯 Success rate: {successful}/{len(trajectories)}")

            if recording_mode:
                print(f"\n📝 LOCAL RECORDING COMPLETE")
                print(f"✅ Trajectories recorded to: {recording_file}")
                print(f"🔄 Next run will use local playback automatically")
                print(f"💡 For CI: Recording is now available for fast local testing")
            else:
                # Estimate speedup for local testing
                estimated_recording_time = 60.0  # Assume ~60s for recording
                speedup = (
                    estimated_recording_time / duration
                    if duration > 0
                    else float("inf")
                )
                print(f"\n🚀 LOCAL PLAYBACK COMPLETE")
                print(f"⚡ Speedup: ~{speedup:.0f}x faster than live LLM calls")
                print(f"💰 Cost: $0 (no LLM API calls)")
                print(f"🎯 Deterministic: Same results every run")
                print(f"🏗️ Perfect for CI/CD and development")

            return True

    except Exception as e:
        print(f"❌ Local test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Clean up environment
        if "REWARD_KIT_PLAYBACK_FILE" in os.environ:
            del os.environ["REWARD_KIT_PLAYBACK_FILE"]


async def main():
    """Main test runner."""
    print("🏠 Starting Local North Star Test")
    print("📋 This test uses local recordings for development/CI")
    print()

    success = await test_local_north_star()

    if success:
        print("\n🎉 Local north star test completed successfully!")
        print("🏗️ Ready for development and CI with local recordings")
    else:
        print("\n💥 Local north star test failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
