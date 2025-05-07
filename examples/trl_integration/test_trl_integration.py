"""
Tests for the TRL integration functionality.

This file contains tests for:
1. The TRL adapter in RewardFunction class
2. The helper functions in trl_adapter.py
3. Basic integration with TRL's expected interfaces
"""

import os
import sys
import unittest
from typing import List, Dict, Any, Optional

# Ensure reward-kit is in the path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

# Import reward-kit components
from reward_kit.reward_function import RewardFunction, reward_function
from reward_kit.models import (
    RewardOutput,
    MetricRewardOutput,
    EvaluateResult,
    MetricResult,
)
from reward_kit.rewards.length import length_reward

# Import TRL adapter utilities
from trl_adapter import (
    create_combined_reward,
    grpo_format_reward,
    create_grpo_reward,
    apply_reward_to_responses,
)


# Define a simple test reward function
@reward_function
def test_reward(
    messages: List[Dict[str, Any]],
    original_messages: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> EvaluateResult:
    """Simple test reward that returns 1.0 if text contains 'good' and 0.0 otherwise."""
    if not messages or len(messages) == 0:
        return EvaluateResult(score=0.0, reason="No messages")

    response = messages[-1]
    if response.get("role") != "assistant" or not response.get("content"):
        return EvaluateResult(score=0.0, reason="No assistant response")

    text = response.get("content", "")

    score = 1.0 if "good" in text.lower() else 0.0
    reason = "Contains 'good'" if score > 0 else "Does not contain 'good'"

    return EvaluateResult(score=score, reason=reason)


class TestTRLIntegration(unittest.TestCase):
    """Test cases for TRL integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_messages = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me about cats."},
                {"role": "assistant", "content": "Cats are good pets."},
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me about dogs."},
                {"role": "assistant", "content": "Dogs are loyal companions."},
            ],
        ]

        self.grpo_messages = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me about cats."},
                {
                    "role": "assistant",
                    "content": "<think>Cats are domesticated animals.</think><answer>Cats are good pets.</answer>",
                },
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me about dogs."},
                {"role": "assistant", "content": "Dogs are loyal companions."},
            ],
        ]

        # Create reward functions
        self.test_rf = RewardFunction(func=test_reward)
        self.length_rf = RewardFunction(func=length_reward)
        self.format_rf = RewardFunction(func=grpo_format_reward)

    def test_basic_adapter(self):
        """Test that the basic TRL adapter works correctly."""
        adapter = self.test_rf.get_trl_adapter()

        # Apply to test messages
        rewards = adapter(self.test_messages)

        # Check results
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # Contains 'good'
        self.assertEqual(rewards[1], 0.0)  # Doesn't contain 'good'

    def test_combined_reward(self):
        """Test that combined rewards work correctly."""
        combined = create_combined_reward(
            reward_functions=[self.test_rf, self.length_rf], weights=[0.7, 0.3]
        )

        # Apply to test messages
        rewards = combined(self.test_messages)

        # Check results
        self.assertEqual(len(rewards), 2)

        # Calculate expected results
        test_scores = [1.0, 0.0]

        # Get length scores (use the adapter directly)
        length_adapter = self.length_rf.get_trl_adapter()
        length_scores = length_adapter(self.test_messages)

        # Calculate expected combined scores
        expected = [
            0.7 * test_scores[0] + 0.3 * length_scores[0],
            0.7 * test_scores[1] + 0.3 * length_scores[1],
        ]

        # Allow for small floating point differences
        self.assertAlmostEqual(rewards[0], expected[0], places=5)
        self.assertAlmostEqual(rewards[1], expected[1], places=5)

    def test_grpo_format_reward(self):
        """Test the GRPO format reward function."""
        format_adapter = self.format_rf.get_trl_adapter()

        # Apply to GRPO messages
        rewards = format_adapter(self.grpo_messages)

        # Check results
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # Has correct format
        self.assertEqual(rewards[1], 0.0)  # Missing format tags

    def test_create_grpo_reward(self):
        """Test the GRPO reward creator."""
        grpo_reward = create_grpo_reward(
            content_reward=self.test_rf, format_weight=0.4, content_weight=0.6
        )

        # Apply to GRPO messages
        rewards = grpo_reward(self.grpo_messages)

        # Check results
        self.assertEqual(len(rewards), 2)

        # Calculate expected results
        format_adapter = self.format_rf.get_trl_adapter()
        format_scores = format_adapter(self.grpo_messages)

        test_adapter = self.test_rf.get_trl_adapter()
        test_scores = test_adapter(self.grpo_messages)

        # Expected combined scores
        expected = [
            0.4 * format_scores[0] + 0.6 * test_scores[0],
            0.4 * format_scores[1] + 0.6 * test_scores[1],
        ]

        # Allow for small floating point differences
        self.assertAlmostEqual(rewards[0], expected[0], places=5)
        self.assertAlmostEqual(rewards[1], expected[1], places=5)

    def test_apply_reward_to_responses(self):
        """Test applying reward to text responses."""
        responses = [
            "<think>Cats are domesticated animals.</think><answer>Cats are good pets.</answer>",
            "Dogs are loyal companions.",
        ]

        # Apply test reward
        rewards = apply_reward_to_responses(self.test_rf, responses)

        # Check results
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # Contains 'good'
        self.assertEqual(rewards[1], 0.0)  # Doesn't contain 'good'

        # Apply format reward
        format_rewards = apply_reward_to_responses(self.format_rf, responses)

        # Check results
        self.assertEqual(len(format_rewards), 2)
        self.assertEqual(format_rewards[0], 1.0)  # Has correct format
        self.assertEqual(format_rewards[1], 0.0)  # Missing format tags


if __name__ == "__main__":
    unittest.main()
