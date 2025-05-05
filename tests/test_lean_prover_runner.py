"""
Basic test runner for the DeepSeek-Prover-V2 reward functions.

This is a simple script to test the lean_prover_reward and deepseek_prover_v2_reward
functions outside of the main test framework, to ensure they're working properly.

Run with:
    python tests/test_lean_prover_runner.py
"""

import json
import sys
import os

# Add project directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reward_kit.rewards.lean_prover import (
    lean_prover_reward,
    deepseek_prover_v2_reward,
)
from reward_kit.models import MetricRewardOutput


def run_tests():
    """Run basic tests for the Lean Prover reward functions"""

    print("Testing lean_prover_reward...")

    # Test with an empty response
    print("\nTest: Empty response")
    result = lean_prover_reward("", "")
    print(f"Score: {result.score}")
    assert result.score == 0.0

    # Skip to complete proof and subgoal tests for basic functionality

    # Test with a complete proof
    print("\nTest: Complete proof")
    statement = "If n is a natural number, then n + 1 > n."
    response = """theorem n_lt_n_plus_one (n : ℕ) : n < n + 1 :=
begin
  apply Nat.lt_succ_self,
end
    """
    result = lean_prover_reward(response, statement, verbose=True)
    print(f"Score: {result.score}")
    # Print metrics if verbose mode was enabled
    if hasattr(result, "metrics") and result.metrics:
        print(
            f"Metrics: {json.dumps({k: {'score': v.score, 'reason': v.reason} for k, v in result.metrics.items()}, indent=2)}"
        )
    assert result.score >= 0.5

    print("\nTesting deepseek_prover_v2_reward...")

    # Test with a complex proof with subgoals
    print("\nTest: Complex proof with subgoals")
    statement = (
        "For all natural numbers n, the sum of the first n natural numbers is n(n+1)/2."
    )
    response = """theorem sum_naturals (n : ℕ) : ∑ i in range n, i = n * (n + 1) / 2 :=
begin
  -- We'll prove this by induction on n
  induction n with d hd,
  -- Base case: n = 0
  { simp, },
  -- Inductive step: assume true for n = d, prove for n = d + 1
  { 
    have step1 : ∑ i in range (d + 1), i = (∑ i in range d, i) + d,
      by simp [sum_range_succ],
    have step2 : (∑ i in range d, i) + d = d * (d + 1) / 2 + d,
      by rw [hd],
    have step3 : d * (d + 1) / 2 + d = (d * (d + 1) + 2 * d) / 2,
      by ring,
    calc
      ∑ i in range (d + 1), i = (∑ i in range d, i) + d : by simp [sum_range_succ]
      ... = d * (d + 1) / 2 + d : by rw [hd]
      ... = (d * (d + 1) + 2 * d) / 2 : by ring
      ... = (d + 1) * ((d + 1) + 1) / 2 : by ring,
  }
end
    """
    result = deepseek_prover_v2_reward(response, statement, verbose=True)
    print(f"Score: {result.score}")
    # Print metrics if verbose mode was enabled
    if hasattr(result, "metrics") and result.metrics:
        print(
            f"Metrics: {json.dumps({k: {'score': v.score, 'reason': v.reason} for k, v in result.metrics.items()}, indent=2)}"
        )
    assert result.score > 0.7

    print("\nAll tests passed!")


def test_lean_prover_functions():
    """Run tests for lean prover functions through pytest integration"""
    run_tests()

if __name__ == "__main__":
    run_tests()
