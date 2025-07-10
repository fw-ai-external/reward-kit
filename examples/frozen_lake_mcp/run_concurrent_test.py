#!/usr/bin/env python3
"""
Simple test runner for the multi-environment concurrent rollouts test.
This script will run the test and generate trajectory files for review.
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Run the concurrent rollouts test to generate trajectory files."""

    print("🚀 Running Multi-Environment Concurrent Rollouts Test")
    print("=" * 60)

    # Set up environment
    base_dir = Path(__file__).parent
    test_file = base_dir / "tests" / "test_record_and_replay_e2e.py"

    # Make sure we're in the right directory
    os.chdir(base_dir)

    # Run the specific test we want
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(test_file) + "::test_multi_environment_concurrent_rollouts",
        "-v",
        "-s",  # Verbose and no capture for real-time output
        "--tb=short",  # Shorter traceback on failures
    ]

    try:
        print(f"Running command: {' '.join(cmd)}")
        print("This will take a few minutes due to conda environment setup...")
        print()

        # Run the test
        result = subprocess.run(cmd, check=False)

        if result.returncode == 0:
            print("\n✅ Test completed successfully!")
        else:
            print(f"\n❌ Test failed with return code: {result.returncode}")

        # Show where files were created
        recordings_dir = base_dir / "tests" / "recordings"
        if recordings_dir.exists():
            print(f"\n📁 Trajectory files created in: {recordings_dir}")

            # List the files
            for file_path in recordings_dir.glob("concurrent_rollout_*.jsonl"):
                print(f"  • {file_path.name}")

            print(
                f"\nYou can now review these files to verify multi-environment isolation!"
            )

    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error running test: {e}")
        return 1

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
