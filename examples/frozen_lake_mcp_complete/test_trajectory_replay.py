#!/usr/bin/env python3
"""
Trajectory Replay Integration Test

This test replays recorded trajectories to validate that the MCP server
produces consistent results and handles grid sizes correctly.

The recorded trajectories serve as golden test data for reliable integration testing.

Usage:
    # Run trajectory replay test
    python test_trajectory_replay.py

    # Run with specific trajectory file
    python test_trajectory_replay.py --trajectories custom_trajectories.jsonl
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Add the mcp_server directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mcp_server"))

from mcp_server.frozen_lake_adapter import FrozenLakeAdapter


class TrajectoryReplayTest:
    """Test class for replaying recorded trajectories."""

    def __init__(
        self, trajectory_file: str = "shared_data/recorded_e2e_trajectories.jsonl"
    ):
        # Make trajectory_file path relative to this test file
        if not os.path.isabs(trajectory_file):
            test_dir = os.path.dirname(os.path.abspath(__file__))
            trajectory_file = os.path.join(test_dir, trajectory_file)

        self.trajectory_file = trajectory_file
        self.adapter = FrozenLakeAdapter()
        self.replay_results = []

    def load_trajectories(self) -> List[Dict[str, Any]]:
        """Load recorded trajectories from file."""
        trajectories = []

        if not os.path.exists(self.trajectory_file):
            print(f"❌ Trajectory file not found: {self.trajectory_file}")
            print(
                "💡 Run 'python test_e2e_integration.py' first to generate trajectories"
            )
            return []

        with open(self.trajectory_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    trajectories.append(json.loads(line))

        return trajectories

    def replay_trajectory(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Replay a single trajectory and validate results.

        This simulates the exact same sequence of actions and validates that:
        1. Grid size is created correctly
        2. Initial state matches
        3. Each action produces the expected result
        4. Final outcome is consistent
        """
        rollout_id = trajectory["rollout_id"]
        environment_context = trajectory["environment_context"]
        grid_type = environment_context.get("grid_type", "4x4")
        seed = environment_context.get("seed", 42)

        print(f"🔄 Replaying trajectory: {rollout_id}")
        print(
            f"   Expected: {grid_type}, seed {seed}, {trajectory['total_moves']} moves"
        )

        # Create environment like the original session
        config = {"grid_type": grid_type}
        env, initial_obs, info = self.adapter.create_environment_with_seed(
            config, seed=seed
        )

        # Validate environment setup
        desc = env.desc
        rows, cols = desc.shape
        actual_grid_size = f"{rows}x{cols}"

        validation_results = {
            "rollout_id": rollout_id,
            "grid_size_match": actual_grid_size == grid_type,
            "initial_state_match": int(initial_obs)
            == trajectory["initial_observation"],
            "step_validations": [],
            "final_outcome_match": False,
            "all_steps_valid": False,
            "replay_successful": False,
        }

        if not validation_results["grid_size_match"]:
            print(
                f"   ❌ Grid size mismatch: expected {grid_type}, got {actual_grid_size}"
            )
            return validation_results

        if not validation_results["initial_state_match"]:
            print(
                f"   ❌ Initial state mismatch: expected {trajectory['initial_observation']}, got {initial_obs}"
            )
            return validation_results

        print(
            f"   ✅ Environment setup: {actual_grid_size}, initial state {initial_obs}"
        )

        # Replay each step
        position = initial_obs
        total_reward = 0

        # Skip initial step (step 0 with no action)
        trajectory_steps = [
            step for step in trajectory["steps"] if step["action"] is not None
        ]

        for i, expected_step in enumerate(trajectory_steps):
            action_str = expected_step["action"]
            expected_position = expected_step["position"]
            expected_reward = expected_step["reward"]
            expected_terminated = expected_step["terminated"]

            # Execute action
            action_int = self.adapter.parse_action(action_str)
            new_obs, reward, terminated, truncated, step_info = (
                self.adapter.step_environment(env, action_int)
            )

            # Validate step results
            step_valid = (
                int(new_obs) == expected_position
                and abs(float(reward) - expected_reward) < 0.001
                and bool(terminated) == expected_terminated
            )

            validation_results["step_validations"].append(
                {
                    "step": i + 1,
                    "action": action_str,
                    "position_match": int(new_obs) == expected_position,
                    "reward_match": abs(float(reward) - expected_reward) < 0.001,
                    "termination_match": bool(terminated) == expected_terminated,
                    "step_valid": step_valid,
                }
            )

            if not step_valid:
                print(f"   ❌ Step {i+1} mismatch:")
                print(f"      Action: {action_str}")
                print(f"      Position: expected {expected_position}, got {new_obs}")
                print(f"      Reward: expected {expected_reward}, got {reward}")
                print(
                    f"      Terminated: expected {expected_terminated}, got {terminated}"
                )

            position = new_obs
            total_reward += reward

            if terminated or truncated:
                break

        # Validate final outcome
        final_success = total_reward > 0
        expected_success = trajectory["success"]
        expected_total_reward = trajectory["total_reward"]

        validation_results.update(
            {
                "final_outcome_match": (
                    final_success == expected_success
                    and abs(total_reward - expected_total_reward) < 0.001
                ),
                "all_steps_valid": all(
                    step["step_valid"]
                    for step in validation_results["step_validations"]
                ),
                "actual_moves": len(trajectory_steps),
                "expected_moves": trajectory["total_moves"],
            }
        )

        # Overall success
        validation_results["replay_successful"] = (
            validation_results["grid_size_match"]
            and validation_results["initial_state_match"]
            and validation_results["all_steps_valid"]
            and validation_results["final_outcome_match"]
        )

        if validation_results["replay_successful"]:
            print(
                f"   ✅ Replay successful: {validation_results['actual_moves']} moves, {'WON' if final_success else 'LOST'}"
            )
        else:
            failed_aspects = []
            if not validation_results["grid_size_match"]:
                failed_aspects.append("grid_size")
            if not validation_results["initial_state_match"]:
                failed_aspects.append("initial_state")
            if not validation_results["all_steps_valid"]:
                failed_aspects.append("steps")
            if not validation_results["final_outcome_match"]:
                failed_aspects.append("outcome")
            print(f"   ❌ Replay failed: {', '.join(failed_aspects)}")

        return validation_results

    def run_replay_tests(self) -> bool:
        """Run all trajectory replay tests."""
        print("🔄 TRAJECTORY REPLAY INTEGRATION TEST")
        print("Validating MCP server consistency using recorded trajectories")
        print("=" * 70)

        trajectories = self.load_trajectories()
        if not trajectories:
            return False

        print(f"✅ Loaded {len(trajectories)} recorded trajectories")
        print()

        # Group trajectories by grid type for organized testing
        by_grid_type = {}
        for traj in trajectories:
            grid_type = traj["grid_type"]
            if grid_type not in by_grid_type:
                by_grid_type[grid_type] = []
            by_grid_type[grid_type].append(traj)

        all_results = []

        for grid_type, grid_trajectories in by_grid_type.items():
            print(f"📋 {grid_type} Trajectory Replays")
            print("-" * 50)

            for trajectory in grid_trajectories:
                result = self.replay_trajectory(trajectory)
                all_results.append(result)
                self.replay_results.append(result)

                status = "✅ VALID" if result["replay_successful"] else "❌ INVALID"
                print(f"   {status} {result['rollout_id']}")

            print()

        # Analysis
        print("=" * 70)
        print("📊 TRAJECTORY REPLAY ANALYSIS")
        print("=" * 70)

        total_replays = len(all_results)
        successful_replays = sum(1 for r in all_results if r["replay_successful"])
        grid_size_correct = sum(1 for r in all_results if r["grid_size_match"])
        initial_state_correct = sum(1 for r in all_results if r["initial_state_match"])
        all_steps_valid = sum(1 for r in all_results if r["all_steps_valid"])
        outcome_match = sum(1 for r in all_results if r["final_outcome_match"])

        print(f"Total Trajectory Replays: {total_replays}")
        print(
            f"Successful Replays: {successful_replays}/{total_replays} ({successful_replays/total_replays*100:.1f}%)"
        )
        print(
            f"Grid Size Correct: {grid_size_correct}/{total_replays} ({grid_size_correct/total_replays*100:.1f}%)"
        )
        print(
            f"Initial State Correct: {initial_state_correct}/{total_replays} ({initial_state_correct/total_replays*100:.1f}%)"
        )
        print(
            f"All Steps Valid: {all_steps_valid}/{total_replays} ({all_steps_valid/total_replays*100:.1f}%)"
        )
        print(
            f"Final Outcome Match: {outcome_match}/{total_replays} ({outcome_match/total_replays*100:.1f}%)"
        )

        # Grid type breakdown
        print(f"\n📈 Grid Type Breakdown:")
        for grid_type, grid_trajectories in by_grid_type.items():
            grid_results = [
                r
                for r in all_results
                if any(t["rollout_id"] == r["rollout_id"] for t in grid_trajectories)
            ]
            grid_success = sum(1 for r in grid_results if r["replay_successful"])
            print(
                f"  • {grid_type}: {grid_success}/{len(grid_results)} successful ({grid_success/len(grid_results)*100:.1f}%)"
            )

        # Final verdict
        integration_success = successful_replays == total_replays

        print(f"\n🎯 REPLAY INTEGRATION VERDICT:")
        if integration_success:
            print("✅ TRAJECTORY REPLAY SUCCESS - All replays validated!")
            print("✅ MCP server produces consistent, reproducible results")
            print("✅ Grid size handling is deterministic and correct")
            print("✅ Integration test reliability confirmed")
        else:
            print("❌ TRAJECTORY REPLAY ISSUES - Some replays failed")
            print("🔧 Check for non-deterministic behavior or configuration issues")
            print("💡 Failed replays indicate potential reliability problems")

        return integration_success

    def get_replay_summary(self) -> Dict[str, Any]:
        """Get summary of replay test results."""
        if not self.replay_results:
            return {"status": "no_replays_run"}

        return {
            "total_replays": len(self.replay_results),
            "successful_replays": sum(
                1 for r in self.replay_results if r["replay_successful"]
            ),
            "grid_size_accuracy": sum(
                1 for r in self.replay_results if r["grid_size_match"]
            )
            / len(self.replay_results),
            "step_accuracy": sum(1 for r in self.replay_results if r["all_steps_valid"])
            / len(self.replay_results),
            "outcome_accuracy": sum(
                1 for r in self.replay_results if r["final_outcome_match"]
            )
            / len(self.replay_results),
            "reliability_score": sum(
                1 for r in self.replay_results if r["replay_successful"]
            )
            / len(self.replay_results),
        }


def main():
    """Main function for trajectory replay testing."""
    parser = argparse.ArgumentParser(description="Trajectory Replay Integration Test")
    parser.add_argument(
        "--trajectories",
        type=str,
        default="shared_data/recorded_e2e_trajectories.jsonl",
        help="Path to recorded trajectories file",
    )

    args = parser.parse_args()

    print("🔄 FrozenLake Trajectory Replay Integration Test")
    print("This validates MCP server consistency using recorded golden trajectories")
    print()

    tester = TrajectoryReplayTest(trajectory_file=args.trajectories)
    success = tester.run_replay_tests()

    summary = tester.get_replay_summary()

    print(f"\n📋 FINAL SUMMARY:")
    print(f"   Reliability Score: {summary['reliability_score']:.1%}")
    print(f"   Grid Size Accuracy: {summary['grid_size_accuracy']:.1%}")
    print(f"   Step Accuracy: {summary['step_accuracy']:.1%}")
    print(f"   Outcome Accuracy: {summary['outcome_accuracy']:.1%}")

    if success:
        print(f"\n🚀 INTEGRATION VALIDATED!")
        print(f"Your MCP server produces reliable, consistent results!")
        return 0
    else:
        print(f"\n🔧 INTEGRATION NEEDS FIXING")
        print(f"Non-deterministic behavior or configuration issues detected")
        return 1


if __name__ == "__main__":
    sys.exit(main())


# Pytest test functions
def test_trajectory_file_exists():
    """Test that trajectory file exists."""
    import os

    # Get path relative to this test file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    trajectory_file = os.path.join(
        test_dir, "shared_data", "recorded_e2e_trajectories.jsonl"
    )
    assert os.path.exists(
        trajectory_file
    ), f"Trajectory file {trajectory_file} should exist. Run test_e2e_integration.py first."


def test_trajectory_replay_4x4():
    """Test trajectory replay for 4x4 scenarios."""
    tester = TrajectoryReplayTest()
    trajectories = tester.load_trajectories()
    assert len(trajectories) > 0, "Should have recorded trajectories"

    # Test 4x4 trajectories
    for trajectory in trajectories:
        if trajectory.get("grid_type") == "4x4":
            result = tester.replay_trajectory(trajectory)
            assert result[
                "replay_successful"
            ], f"4x4 trajectory {trajectory['rollout_id']} should replay successfully"
            assert result[
                "grid_size_match"
            ], f"4x4 trajectory {trajectory['rollout_id']} should have matching grid size"
            assert result[
                "all_steps_valid"
            ], f"4x4 trajectory {trajectory['rollout_id']} should have all valid steps"


def test_trajectory_replay_8x8():
    """Test trajectory replay for 8x8 scenarios."""
    tester = TrajectoryReplayTest()
    trajectories = tester.load_trajectories()
    assert len(trajectories) > 0, "Should have recorded trajectories"

    # Test 8x8 trajectories
    for trajectory in trajectories:
        if trajectory.get("grid_type") == "8x8":
            result = tester.replay_trajectory(trajectory)
            assert result[
                "replay_successful"
            ], f"8x8 trajectory {trajectory['rollout_id']} should replay successfully"
            assert result[
                "grid_size_match"
            ], f"8x8 trajectory {trajectory['rollout_id']} should have matching grid size"
            assert result[
                "all_steps_valid"
            ], f"8x8 trajectory {trajectory['rollout_id']} should have all valid steps"


def test_trajectory_replay_deterministic():
    """Test that trajectory replay is 100% deterministic."""
    tester = TrajectoryReplayTest()
    trajectories = tester.load_trajectories()
    assert len(trajectories) > 0, "Should have recorded trajectories"

    # Run all replays and check that they all succeed
    all_successful = True
    for trajectory in trajectories:
        result = tester.replay_trajectory(trajectory)
        if not result["replay_successful"]:
            all_successful = False
            break

    assert (
        all_successful
    ), "All trajectory replays should be successful (100% deterministic)"
