"""
Tests for length reward functions.
"""

import sys
import os
import unittest

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reward_kit.rewards.length import (
    length_reward, cosine_length_reward, count_tokens
)
from reward_kit.models import Message


class TestLengthReward(unittest.TestCase):
    """Test the length reward functions."""

    def test_count_tokens(self):
        """Test token counting with different methods."""
        text = "This is a test sentence with 8 words."
        
        # Test whitespace tokenization
        self.assertEqual(count_tokens(text, "whitespace"), 8)
        
        # Test character tokenization
        self.assertEqual(count_tokens(text, "character"), len(text))
        
        # Test word tokenization
        self.assertEqual(count_tokens(text, "words"), 8)  # All 8 words are counted
        
    def test_target_length(self):
        """Test reward calculation with target length."""
        content_short = "This is a short response."
        content_target = "This is a response that has exactly the target length we are looking for."
        content_long = "This is a longer response that exceeds the target length by a significant margin and contains many more tokens than we would ideally want in a concise answer that gets to the point efficiently."
        
        target_length = count_tokens(content_target)
        
        # Test with response matching target length
        messages_target = [
            {"role": "user", "content": "Write a response"},
            {"role": "assistant", "content": content_target}
        ]
        
        result_target = length_reward(
            messages=messages_target,
            target_length=target_length
        )
        
        # Should be high score for matching target length
        self.assertEqual(result_target["score"], 1.0)
        self.assertTrue(result_target["metrics"]["length"]["success"])
        
        # Test with response shorter than target length
        messages_short = [
            {"role": "user", "content": "Write a response"},
            {"role": "assistant", "content": content_short}
        ]
        
        result_short = length_reward(
            messages=messages_short,
            target_length=target_length
        )
        
        # Should be lower score for not matching target length
        self.assertLess(result_short["score"], 1.0)
        
        # Test with response longer than target length
        messages_long = [
            {"role": "user", "content": "Write a response"},
            {"role": "assistant", "content": content_long}
        ]
        
        result_long = length_reward(
            messages=messages_long,
            target_length=target_length
        )
        
        # Should be lower score for not matching target length
        self.assertLess(result_long["score"], 1.0)
        
    def test_min_length(self):
        """Test reward calculation with minimum length."""
        content_short = "Too short."
        content_adequate = "This response meets the minimum length requirement."
        
        min_length = count_tokens(content_adequate) - 1
        
        # Test with response meeting minimum length
        messages_adequate = [
            {"role": "user", "content": "Write a response"},
            {"role": "assistant", "content": content_adequate}
        ]
        
        result_adequate = length_reward(
            messages=messages_adequate,
            min_length=min_length
        )
        
        # Should be high score for meeting minimum length
        self.assertEqual(result_adequate["score"], 1.0)
        self.assertTrue(result_adequate["metrics"]["length"]["success"])
        
        # Test with response below minimum length
        messages_short = [
            {"role": "user", "content": "Write a response"},
            {"role": "assistant", "content": content_short}
        ]
        
        result_short = length_reward(
            messages=messages_short,
            min_length=min_length
        )
        
        # Should be lower score for not meeting minimum length
        self.assertLess(result_short["score"], 1.0)
        self.assertFalse(result_short["metrics"]["length"]["success"])
        
    def test_max_length(self):
        """Test reward calculation with maximum length."""
        content_short = "This is concise."
        content_long = "This response exceeds the maximum length by including unnecessary details and being overly verbose when a shorter answer would suffice."
        
        max_length = count_tokens(content_short) + 2
        
        # Test with response within maximum length
        messages_short = [
            {"role": "user", "content": "Write a response"},
            {"role": "assistant", "content": content_short}
        ]
        
        result_short = length_reward(
            messages=messages_short,
            max_length=max_length
        )
        
        # Should be high score for staying within maximum length
        self.assertEqual(result_short["score"], 1.0)
        self.assertTrue(result_short["metrics"]["length"]["success"])
        
        # Test with response exceeding maximum length
        messages_long = [
            {"role": "user", "content": "Write a response"},
            {"role": "assistant", "content": content_long}
        ]
        
        result_long = length_reward(
            messages=messages_long,
            max_length=max_length
        )
        
        # Should be lower score for exceeding maximum length
        self.assertLess(result_long["score"], 1.0)
        self.assertFalse(result_long["metrics"]["length"]["success"])
        
    def test_min_max_range(self):
        """Test reward calculation with both min and max length."""
        content_short = "Too short."
        content_good = "This response is just right."
        content_long = "This response is too long with unnecessary details and verbosity."
        
        min_length = count_tokens(content_short) + 1
        max_length = count_tokens(content_long) - 1
        
        # Test with response within range
        messages_good = [
            {"role": "user", "content": "Write a response"},
            {"role": "assistant", "content": content_good}
        ]
        
        result_good = length_reward(
            messages=messages_good,
            min_length=min_length,
            max_length=max_length
        )
        
        # Should be high score for staying within range
        self.assertEqual(result_good["score"], 1.0)
        self.assertTrue(result_good["metrics"]["length"]["success"])
        
        # Test with response too short
        messages_short = [
            {"role": "user", "content": "Write a response"},
            {"role": "assistant", "content": content_short}
        ]
        
        result_short = length_reward(
            messages=messages_short,
            min_length=min_length,
            max_length=max_length
        )
        
        # Should be lower score for being too short
        self.assertLess(result_short["score"], 1.0)
        self.assertFalse(result_short["metrics"]["length"]["success"])
        
        # Test with response too long
        messages_long = [
            {"role": "user", "content": "Write a response"},
            {"role": "assistant", "content": content_long}
        ]
        
        result_long = length_reward(
            messages=messages_long,
            min_length=min_length,
            max_length=max_length
        )
        
        # Should be lower score for being too long
        self.assertLess(result_long["score"], 1.0)
        self.assertFalse(result_long["metrics"]["length"]["success"])
        
    def test_cosine_scaling(self):
        """Test cosine scaling for reward calculation."""
        content_short = "Short answer."
        content_medium = "This is a medium length response with adequate detail."
        content_long = "This is a much longer response that provides extensive details and explanations beyond what is necessary to answer the question effectively."
        
        target_length = count_tokens(content_medium)
        
        # Test with cosine scaling
        messages_short = [
            {"role": "user", "content": "Write a response"},
            {"role": "assistant", "content": content_short}
        ]
        
        result_linear = length_reward(
            messages=messages_short,
            target_length=target_length,
            scaling="linear"
        )
        
        result_cosine = length_reward(
            messages=messages_short,
            target_length=target_length,
            scaling="cosine"
        )
        
        # Cosine scaling should result in different scores than linear
        self.assertNotEqual(result_linear["score"], result_cosine["score"])
        
    def test_cosine_length_reward(self):
        """Test the cosine length reward function."""
        content_short = "Short correct answer."
        content_long = "This is a much longer correct answer that provides extensive details and explanations that are technically accurate but not as concise as they could be."
        
        # Test with correct short answer
        messages_short = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": content_short}
        ]
        
        result_short_correct = cosine_length_reward(
            messages=messages_short,
            is_correct=True,
            max_length=50  # Short max length to highlight difference
        )
        
        # Test with correct long answer
        messages_long = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": content_long}
        ]
        
        result_long_correct = cosine_length_reward(
            messages=messages_long,
            is_correct=True,
            max_length=50
        )
        
        # Shorter correct answer should score higher than longer correct answer
        self.assertGreater(result_short_correct["score"], result_long_correct["score"])
        
        # Test with incorrect answers
        result_short_incorrect = cosine_length_reward(
            messages=messages_short,
            is_correct=False,
            max_length=50
        )
        
        result_long_incorrect = cosine_length_reward(
            messages=messages_long,
            is_correct=False,
            max_length=50
        )
        
        # Longer incorrect answer should be penalized less than shorter incorrect
        self.assertGreater(result_long_incorrect["score"], result_short_incorrect["score"])
        
        # Correct answers should score higher than incorrect answers
        self.assertGreater(result_short_correct["score"], result_short_incorrect["score"])
        self.assertGreater(result_long_correct["score"], result_long_incorrect["score"])
        
    def test_correctness_parameter(self):
        """Test cosine length reward with correctness parameter."""
        content = "This is an answer."
        
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": content}
        ]
        
        # Test with different correctness values
        result_high = cosine_length_reward(
            messages=messages,
            correctness=0.95,  # High correctness
            max_length=50
        )
        
        result_low = cosine_length_reward(
            messages=messages,
            correctness=0.5,  # Low correctness
            max_length=50
        )
        
        # Higher correctness should result in higher score
        self.assertGreater(result_high["score"], result_low["score"])


if __name__ == "__main__":
    unittest.main()