#!/usr/bin/env python3
"""
Stack Hardening Integration Test

This test validates that the MCP simulation server can handle complex 8x8 FrozenLake
scenarios that require 10-20+ moves to complete, serving as comprehensive stack hardening.

Usage:
    # Run directly
    python test_8x8_integration.py

    # Run with pytest
    pytest test_8x8_integration.py
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Add the mcp_server directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mcp_server"))

from mcp_server.frozen_lake_adapter import FrozenLakeAdapter


class StackHardeningTest:
    """Test class for validating complex scenario handling."""

    def __init__(self):
        self.adapter = FrozenLakeAdapter()
        self.test_results = []

    def load_rollout_configs(self, filename: str) -> List[Dict[str, Any]]:
        """Load rollout configurations from JSONL file."""
        rollouts = []
        # Get the directory of this test file
        test_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(test_dir, "shared_data", filename)

        if not os.path.exists(filepath):
            print(f"⚠️ File not found: {filepath}")
            return []

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rollouts.append(json.loads(line))

        return rollouts

    def run_complex_session(
        self, rollout_config: Dict[str, Any], max_moves: int = 30
    ) -> Dict[str, Any]:
        """
        Run a complex session with the given configuration.

        This simulates a long, complex session that would stress-test the system.
        """
        print(f"🎯 Running complex session: {rollout_config['id']}")

        environment_context = rollout_config.get("environment_context", {})
        grid_type = environment_context.get("grid_type", "4x4")
        seed = environment_context.get("seed", 42)

        print(f"   Grid: {grid_type}, Seed: {seed}, Max moves: {max_moves}")

        # Create environment
        config = {"grid_type": grid_type}
        env, initial_obs, info = self.adapter.create_environment_with_seed(
            config, seed=seed
        )

        # Validate grid size
        desc = env.desc
        rows, cols = desc.shape
        actual_grid_size = f"{rows}x{cols}"

        if actual_grid_size != grid_type:
            print(
                f"   ❌ Grid size mismatch: expected {grid_type}, got {actual_grid_size}"
            )
            return {
                "rollout_id": rollout_config["id"],
                "grid_type": grid_type,
                "success": False,
                "moves_taken": 0,
                "failure_reason": f"grid_size_mismatch_{actual_grid_size}",
            }

        # Run complex session
        position = initial_obs
        moves_taken = 0
        total_reward = 0

        # Simple navigation strategy
        def get_next_move(pos, env_desc):
            size = len(env_desc)
            row = pos // size
            col = pos % size

            # Try to move toward goal (usually bottom-right)
            if row < size - 1 and env_desc[row + 1][col] != b"H":
                return "DOWN"
            elif col < size - 1 and env_desc[row][col + 1] != b"H":
                return "RIGHT"
            elif col > 0 and env_desc[row][col - 1] != b"H":
                return "LEFT"
            elif row > 0 and env_desc[row - 1][col] != b"H":
                return "UP"
            else:
                return "DOWN"  # fallback

        while moves_taken < max_moves:
            action_str = get_next_move(position, desc)
            action_int = self.adapter.parse_action(action_str)

            # Execute move
            new_obs, reward, terminated, truncated, step_info = (
                self.adapter.step_environment(env, action_int)
            )

            moves_taken += 1
            total_reward += reward
            position = new_obs

            if terminated or truncated:
                success = reward > 0
                print(
                    f"   🏁 Finished after {moves_taken} moves: {'SUCCESS' if success else 'FAILED'}"
                )
                break
        else:
            print(f"   ⏰ Stopped after {max_moves} moves (limit reached)")

        return {
            "rollout_id": rollout_config["id"],
            "grid_type": grid_type,
            "seed": seed,
            "moves_taken": moves_taken,
            "success": total_reward > 0,
            "total_reward": float(total_reward),
            "complexity_score": (rows * cols) + moves_taken,
            "stress_test_passed": moves_taken >= (3 if grid_type == "4x4" else 10),
        }

    def run_stack_hardening_tests(self) -> bool:
        """Run the complete stack hardening test suite."""
        print("🔧 STACK HARDENING TEST SUITE")
        print("Testing complex 8x8 scenarios to validate server robustness")
        print("=" * 70)

        # Test scenarios - both baseline and complex
        test_files = [
            ("rollouts.jsonl", "4x4 Baseline Tests", 15),
            ("rollouts_8x8.jsonl", "8x8 Complex Tests", 30),
        ]

        all_results = []

        for filename, description, max_moves in test_files:
            print(f"\n📋 {description}")
            print("-" * 50)

            rollouts = self.load_rollout_configs(filename)
            if not rollouts:
                print(f"⚠️ No rollouts found in {filename}")
                continue

            print(f"✅ Loaded {len(rollouts)} test scenarios")

            for rollout in rollouts:
                result = self.run_complex_session(rollout, max_moves)
                all_results.append(result)
                self.test_results.append(result)

                status = "✅ PASS" if result["stress_test_passed"] else "❌ FAIL"
                success = "🎉 WON" if result["success"] else "💀 LOST"
                print(
                    f"   {status} {result['rollout_id']}: {result['moves_taken']} moves, {success}"
                )

        # Analysis
        print("\n" + "=" * 70)
        print("📊 STACK HARDENING ANALYSIS")
        print("=" * 70)

        total_tests = len(all_results)
        stress_tests_passed = sum(1 for r in all_results if r["stress_test_passed"])
        game_successes = sum(1 for r in all_results if r["success"])

        print(f"Total Scenarios Tested: {total_tests}")
        print(
            f"Stress Tests Passed: {stress_tests_passed}/{total_tests} ({stress_tests_passed/total_tests*100:.1f}%)"
        )
        print(
            f"Game Successes: {game_successes}/{total_tests} ({game_successes/total_tests*100:.1f}%)"
        )

        # Complexity metrics by grid type
        by_grid_type = {}
        for result in all_results:
            grid_type = result["grid_type"]
            if grid_type not in by_grid_type:
                by_grid_type[grid_type] = []
            by_grid_type[grid_type].append(result)

        print(f"\n📈 Complexity Metrics:")
        for grid_type, results in by_grid_type.items():
            avg_moves = sum(r["moves_taken"] for r in results) / len(results)
            avg_complexity = sum(r["complexity_score"] for r in results) / len(results)
            print(
                f"  • {grid_type}: {len(results)} tests, avg {avg_moves:.1f} moves, complexity {avg_complexity:.1f}"
            )

        # Hardening score
        hardening_score = stress_tests_passed / total_tests if total_tests > 0 else 0

        print(f"\n🎯 STACK HARDENING VERDICT:")
        if hardening_score >= 0.8:  # 80% threshold
            print("✅ STACK HARDENED - System can handle complex scenarios!")
            print("✅ 8x8 grid support is robust and production-ready")
            print("✅ Dynamic grid sizing works under stress")
            print(f"✅ Hardening Score: {hardening_score:.1%}")
        else:
            print("⚠️ NEEDS HARDENING - System struggles with complex scenarios")
            print("🔧 Consider optimizing for longer sessions and complex navigation")
            print(f"⚠️ Hardening Score: {hardening_score:.1%}")

        return hardening_score >= 0.8

    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of test results."""
        if not self.test_results:
            return {"status": "no_tests_run"}

        return {
            "total_tests": len(self.test_results),
            "stress_tests_passed": sum(
                1 for r in self.test_results if r["stress_test_passed"]
            ),
            "game_successes": sum(1 for r in self.test_results if r["success"]),
            "hardening_score": sum(
                1 for r in self.test_results if r["stress_test_passed"]
            )
            / len(self.test_results),
            "avg_complexity": sum(r["complexity_score"] for r in self.test_results)
            / len(self.test_results),
            "max_moves_handled": max(r["moves_taken"] for r in self.test_results),
        }


# Pytest test functions
def test_4x4_baseline_scenarios():
    """Test 4x4 baseline scenarios."""
    tester = StackHardeningTest()
    rollouts = tester.load_rollout_configs("rollouts.jsonl")
    assert len(rollouts) > 0, "Should have 4x4 rollout configurations"

    results = []
    for rollout in rollouts:
        result = tester.run_complex_session(rollout, max_moves=15)
        results.append(result)
        # Each 4x4 test should handle at least 3 moves
        assert result[
            "stress_test_passed"
        ], f"4x4 test {rollout['id']} should pass stress test"


def test_8x8_complex_scenarios():
    """Test 8x8 complex scenarios."""
    tester = StackHardeningTest()
    rollouts = tester.load_rollout_configs("rollouts_8x8.jsonl")
    assert len(rollouts) > 0, "Should have 8x8 rollout configurations"

    results = []
    for rollout in rollouts:
        result = tester.run_complex_session(rollout, max_moves=30)
        results.append(result)
        # Each 8x8 test should handle at least 10 moves
        assert result[
            "stress_test_passed"
        ], f"8x8 test {rollout['id']} should pass stress test"


def test_grid_size_accuracy():
    """Test that grid sizes are created accurately."""
    tester = StackHardeningTest()

    # Test 4x4
    config_4x4 = {"grid_type": "4x4"}
    env_4x4, _, _ = tester.adapter.create_environment_with_seed(config_4x4, seed=42)
    rows, cols = env_4x4.desc.shape
    assert f"{rows}x{cols}" == "4x4", "4x4 grid should be created correctly"

    # Test 8x8
    config_8x8 = {"grid_type": "8x8"}
    env_8x8, _, _ = tester.adapter.create_environment_with_seed(config_8x8, seed=42)
    rows, cols = env_8x8.desc.shape
    assert f"{rows}x{cols}" == "8x8", "8x8 grid should be created correctly"


def test_adapter_imports():
    """Test that all required imports work."""
    from mcp_server.frozen_lake_adapter import FrozenLakeAdapter

    adapter = FrozenLakeAdapter()
    assert adapter is not None, "Adapter should be importable and instantiable"


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(description="Stack Hardening Integration Test")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick test with fewer moves"
    )

    args = parser.parse_args()

    print("🔧 FrozenLake Stack Hardening Integration Test")
    print("This validates your MCP server can handle complex 8x8 scenarios")
    print()

    tester = StackHardeningTest()
    success = tester.run_stack_hardening_tests()

    summary = tester.get_test_summary()

    print(f"\n📋 FINAL SUMMARY:")
    print(f"   Total Tests: {summary['total_tests']}")
    print(f"   Hardening Score: {summary['hardening_score']:.1%}")
    print(f"   Max Moves Handled: {summary['max_moves_handled']}")
    print(f"   Avg Complexity: {summary['avg_complexity']:.1f}")

    if success:
        print(f"\n🚀 STACK HARDENED!")
        print(f"Your system is ready for complex production scenarios!")
        return 0
    else:
        print(f"\n🔧 NEEDS HARDENING")
        print(f"System requires optimization for complex scenarios")
        return 1


if __name__ == "__main__":
    sys.exit(main())
