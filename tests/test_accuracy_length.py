"""
Tests for cosine-scaled accuracy + length reward function.
"""

import sys
import os
import unittest

# Add the parent directory to sys.path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from reward_kit.rewards.accuracy_length import (
    cosine_scaled_accuracy_length_reward,
)
from reward_kit.models import Message


class TestCosineScaledAccuracyLengthReward(unittest.TestCase):
    """Test the cosine-scaled accuracy + length reward function."""

    def test_correct_short_answer(self):
        """Test with a correct short answer."""
        content = "The answer is 4."

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": content},
        ]

        result = cosine_scaled_accuracy_length_reward(
            messages=messages, ground_truth="4", max_length=50
        )

        # Should get a high score for short correct answer
        self.assertGreaterEqual(result["score"], 0.7)
        self.assertTrue(result["metrics"]["combined_reward"]["success"])
        self.assertTrue(result["metrics"]["accuracy"]["success"])

    def test_correct_long_answer(self):
        """Test with a correct but verbose answer."""
        content = """
        To solve this addition problem, I'll work through it step by step.
        
        Starting with 2+2:
        - I have 2 units
        - I need to add 2 more units
        - 2 + 2 = 4
        
        Therefore, the answer is 4.
        """

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": content},
        ]

        result = cosine_scaled_accuracy_length_reward(
            messages=messages, ground_truth="4", max_length=50
        )

        # Should get a medium score for correct but verbose answer
        self.assertGreaterEqual(result["score"], 0.5)
        self.assertTrue(result["metrics"]["combined_reward"]["success"])
        self.assertTrue(result["metrics"]["accuracy"]["success"])

    def test_incorrect_short_answer(self):
        """Test with an incorrect short answer."""
        content = "The answer is 5."

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": content},
        ]

        result = cosine_scaled_accuracy_length_reward(
            messages=messages, ground_truth="4", max_length=50
        )

        # Should get a low score for incorrect answer
        self.assertLess(result["score"], 0.5)
        self.assertFalse(result["metrics"]["combined_reward"]["success"])
        self.assertFalse(result["metrics"]["accuracy"]["success"])

    def test_incorrect_long_answer(self):
        """Test with an incorrect verbose answer."""
        content = """
        To solve this addition problem, I'll work through it step by step.
        
        Starting with 2+2:
        - I have 2 units
        - I need to add 2 more units
        - Since 2 is followed by 3, which is followed by 5
        - Therefore 2 + 2 = 5
        
        So the answer is 5.
        """

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": content},
        ]

        result = cosine_scaled_accuracy_length_reward(
            messages=messages, ground_truth="4", max_length=50
        )

        # Should get a low score for incorrect verbose answer
        # But slightly higher than incorrect short answer
        self.assertLess(result["score"], 0.5)
        self.assertFalse(result["metrics"]["combined_reward"]["success"])
        self.assertFalse(result["metrics"]["accuracy"]["success"])

    def test_correct_short_vs_correct_long(self):
        """Test that correct short answers score higher than correct long answers."""
        content_short = "The answer is 4."
        content_long = """
        To solve this addition problem, I'll work through it step by step.
        
        Starting with 2+2:
        - I have 2 units
        - I need to add 2 more units
        - 2 + 2 = 4
        
        Therefore, the answer is 4.
        """

        messages_short = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": content_short},
        ]

        messages_long = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": content_long},
        ]

        result_short = cosine_scaled_accuracy_length_reward(
            messages=messages_short, ground_truth="4", max_length=50
        )

        result_long = cosine_scaled_accuracy_length_reward(
            messages=messages_long, ground_truth="4", max_length=50
        )

        # Short correct should score higher than long correct
        self.assertGreater(result_short["score"], result_long["score"])

    def test_incorrect_short_vs_incorrect_long(self):
        """Test that incorrect long answers score slightly higher than incorrect short."""
        content_short = "The answer is 5."
        content_long = """
        To solve this addition problem, I'll work through it step by step.
        
        Starting with 2+2:
        - I have 2 units
        - I need to add 2 more units
        - Since 2 is followed by 3, which is followed by 5
        - Therefore 2 + 2 = 5
        
        So the answer is 5.
        """

        messages_short = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": content_short},
        ]

        messages_long = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": content_long},
        ]

        result_short = cosine_scaled_accuracy_length_reward(
            messages=messages_short, ground_truth="4", max_length=50
        )

        result_long = cosine_scaled_accuracy_length_reward(
            messages=messages_long, ground_truth="4", max_length=50
        )

        # Long incorrect should score slightly higher than short incorrect
        # This rewards showing work even if the answer is wrong
        self.assertGreaterEqual(result_long["score"], result_short["score"])

    def test_custom_weights(self):
        """Test with custom accuracy and length weights."""
        content = "The answer is 4."

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": content},
        ]

        # Default weights (accuracy_weight=0.7, length_weight=0.3)
        result_default = cosine_scaled_accuracy_length_reward(
            messages=messages, ground_truth="4", max_length=50
        )

        # Custom weights (accuracy_weight=0.9, length_weight=0.1)
        result_custom = cosine_scaled_accuracy_length_reward(
            messages=messages,
            ground_truth="4",
            max_length=50,
            correctness_weight=0.9,
            length_weight=0.1,
        )

        # Scores should be different with different weights
        self.assertNotEqual(result_default["score"], result_custom["score"])

    def test_correct_beats_incorrect(self):
        """Test that even long correct answers beat short incorrect answers."""
        content_long_correct = """
        To solve this addition problem, I'll work through it step by step.
        I'll start by adding 2 and 2 together.
        2 represents a quantity of two units.
        When I add another 2 units, I get a total of 4 units.
        This is a simple addition problem that demonstrates the basic concept of addition.
        In mathematics, addition is one of the four basic operations, alongside subtraction, multiplication, and division.
        For this specific problem, I'll now calculate:
        2 + 2 = 4
        The answer is 4.
        """

        content_short_incorrect = "The answer is 5."

        messages_long_correct = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": content_long_correct},
        ]

        messages_short_incorrect = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": content_short_incorrect},
        ]

        result_long_correct = cosine_scaled_accuracy_length_reward(
            messages=messages_long_correct, ground_truth="4", max_length=50
        )

        result_short_incorrect = cosine_scaled_accuracy_length_reward(
            messages=messages_short_incorrect, ground_truth="4", max_length=50
        )

        # Long correct should always beat short incorrect
        self.assertGreater(
            result_long_correct["score"], result_short_incorrect["score"]
        )


if __name__ == "__main__":
    unittest.main()
