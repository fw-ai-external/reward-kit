"""
Tests for repetition reward functions.
"""

import sys
import os
import unittest

# Add the parent directory to sys.path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from reward_kit.rewards.repetition import (
    repetition_penalty_reward,
    diversity_reward,
    get_ngrams,
)
from reward_kit.models import Message


class TestRepetitionReward(unittest.TestCase):
    """Test the repetition reward functions."""

    def test_get_ngrams(self):
        """Test n-gram extraction."""
        text = "This is a test sentence for testing n-grams"

        # Test unigrams (1-grams)
        unigrams, uni_count = get_ngrams(text, 1)
        self.assertEqual(uni_count, 8)  # 8 words
        self.assertEqual(
            len(set(unigrams)), 8
        )  # All 8 words are unique in this example

        # Test bigrams (2-grams)
        bigrams, bi_count = get_ngrams(text, 2)
        self.assertEqual(bi_count, 7)  # 7 possible bigrams in 8 words
        self.assertEqual(len(set(bigrams)), 7)  # All unique

        # Test 3-grams
        trigrams, tri_count = get_ngrams(text, 3)
        self.assertEqual(tri_count, 6)  # 6 possible trigrams in 8 words

    def test_no_repetition(self):
        """Test with text that has no repetition."""
        content = """
        This is a response with unique words.
        It does not repeat phrases or use redundant language.
        Each sentence contains different vocabulary items.
        The diversity of expression is maintained throughout.
        """

        messages = [
            {"role": "user", "content": "Write a response"},
            {"role": "assistant", "content": content},
        ]

        result = repetition_penalty_reward(messages=messages, ngram_size=3)

        # Should be high score (low penalty) for non-repetitive text
        self.assertGreaterEqual(result["score"], 0.9)
        self.assertTrue(result["metrics"]["repetition"]["success"])

    def test_high_repetition(self):
        """Test with highly repetitive text."""
        content = """
        This is a test. This is a test. This is a test.
        This is a test. This is a test. This is a test.
        This is a test. This is a test. This is a test.
        """

        messages = [
            {"role": "user", "content": "Write a response"},
            {"role": "assistant", "content": content},
        ]

        result = repetition_penalty_reward(messages=messages, ngram_size=3)

        # Should be lower score (higher penalty) for repetitive text
        self.assertLess(result["score"], 0.8)
        self.assertFalse(result["metrics"]["repetition"]["success"])

    def test_moderate_repetition(self):
        """Test with moderately repetitive text."""
        content = """
        Let me explain this concept. This concept is important.
        When implementing this concept, remember the key points.
        The key points of this concept include understanding the basics.
        Understanding the basics is essential for mastery.
        """

        messages = [
            {"role": "user", "content": "Explain a concept"},
            {"role": "assistant", "content": content},
        ]

        result = repetition_penalty_reward(
            messages=messages,
            ngram_size=2,  # Bigrams will detect phrases like "this concept" repeating
        )

        # Should be intermediate score for moderately repetitive text
        self.assertGreater(result["score"], 0.5)
        self.assertLess(result["score"], 1.0)

    def test_different_ngram_sizes(self):
        """Test with different n-gram sizes."""
        content = """
        Let me explain this concept. This concept is important.
        When implementing this concept, remember the key points.
        The key points of this concept include understanding the basics.
        Understanding the basics is essential for mastery.
        """

        messages = [
            {"role": "user", "content": "Explain a concept"},
            {"role": "assistant", "content": content},
        ]

        # Test with unigrams (individual words)
        result_1gram = repetition_penalty_reward(
            messages=messages, ngram_size=1
        )

        # Test with trigrams (three-word phrases)
        result_3gram = repetition_penalty_reward(
            messages=messages, ngram_size=3
        )

        # Unigrams should detect more repetition than trigrams
        # (words repeat more often than three-word phrases)
        self.assertLess(result_1gram["score"], result_3gram["score"])

    def test_max_penalty(self):
        """Test with different max_penalty values."""
        content = """
        This is a test. This is a test. This is a test.
        This is a test. This is a test. This is a test.
        """

        messages = [
            {"role": "user", "content": "Write a response"},
            {"role": "assistant", "content": content},
        ]

        # Test with lower max penalty
        result_low = repetition_penalty_reward(
            messages=messages, ngram_size=3, max_penalty=0.3
        )

        # Test with higher max penalty
        result_high = repetition_penalty_reward(
            messages=messages, ngram_size=3, max_penalty=0.9
        )

        # Higher max_penalty should result in lower score
        self.assertGreater(result_low["score"], result_high["score"])

    def test_empty_response(self):
        """Test with empty response."""
        messages = [
            {"role": "user", "content": "Write a response"},
            {"role": "assistant", "content": ""},
        ]

        result = repetition_penalty_reward(messages=messages)

        # Empty response should not be penalized
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["metrics"]["repetition"]["success"])

    def test_diversity_reward(self):
        """Test the diversity reward function."""
        diverse_content = """
        This response utilizes a varied vocabulary with minimal repetition.
        It incorporates different sentence structures and expressions.
        The language employed demonstrates lexical richness.
        Word choice is deliberately diverse to showcase linguistic range.
        Phrasing avoids redundancy through careful selection of terms.
        """

        repetitive_content = """
        This is repetitive. This is repetitive. This is repetitive.
        This is repetitive. This is repetitive. This is repetitive.
        This is repetitive. This is repetitive. This is repetitive.
        """

        messages_diverse = [
            {"role": "user", "content": "Write a response"},
            {"role": "assistant", "content": diverse_content},
        ]

        messages_repetitive = [
            {"role": "user", "content": "Write a response"},
            {"role": "assistant", "content": repetitive_content},
        ]

        result_diverse = diversity_reward(messages=messages_diverse)

        result_repetitive = diversity_reward(messages=messages_repetitive)

        # Diverse content should score higher
        self.assertGreater(result_diverse["score"], result_repetitive["score"])
        self.assertTrue(result_diverse["metrics"]["diversity"]["success"])
        self.assertFalse(result_repetitive["metrics"]["diversity"]["success"])

    def test_diversity_with_custom_weights(self):
        """Test diversity reward with custom n-gram weights."""
        content = """
        This text has some repetition of phrases like this text.
        However, overall vocabulary is reasonably diverse and varied.
        Sentence structures alternate between simple and complex forms.
        """

        messages = [
            {"role": "user", "content": "Write a response"},
            {"role": "assistant", "content": content},
        ]

        # Test with default weights
        result_default = diversity_reward(messages=messages)

        # Test with custom weights prioritizing unigrams
        result_unigram = diversity_reward(
            messages=messages,
            ngram_sizes=[1, 2, 3],
            weights=[
                0.7,
                0.2,
                0.1,
            ],  # Higher weight on unigrams (individual words)
        )

        # Test with custom weights prioritizing trigrams
        result_trigram = diversity_reward(
            messages=messages,
            ngram_sizes=[1, 2, 3],
            weights=[
                0.1,
                0.2,
                0.7,
            ],  # Higher weight on trigrams (three-word phrases)
        )

        # Different weight distributions should produce different scores
        self.assertNotEqual(result_unigram["score"], result_trigram["score"])


if __name__ == "__main__":
    unittest.main()
