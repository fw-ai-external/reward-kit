import unittest
from typing import List, Dict, Any

from reward_kit.rewards.multiple_choice_math_reward import multiple_choice_math_reward, extract_mcq_option
from reward_kit.models import EvaluateResult, Message

class TestExtractMCQOption(unittest.TestCase):
    """Test the MCQ option extraction utility."""

    def test_basic_extraction(self):
        self.assertEqual(extract_mcq_option("The answer is (A)."), [("(A)", "A")])
        self.assertEqual(extract_mcq_option("Choose B. for this one."), [("B.", "B")])
        self.assertEqual(extract_mcq_option("It must be [C]"), [("[C]", "C")])
        self.assertEqual(extract_mcq_option("Perhaps D is correct."), [("D", "D")]) # Standalone D followed by space
        self.assertEqual(extract_mcq_option("The final choice is E"), [("E", "E")]) # Standalone E at end of string
        self.assertEqual(extract_mcq_option("Answer: {A}"), [("{A}", "A")])

    def test_multiple_options_found(self):
        # Should extract all unique options it finds based on the pattern
        self.assertEqual(extract_mcq_option("Is it (A) or (B)?"), [("(A)", "A"), ("(B)", "B")])

    def test_no_mcq_option(self):
        self.assertEqual(extract_mcq_option("The answer is 123."), [])
        self.assertEqual(extract_mcq_option("This is just text."), [])
        self.assertEqual(extract_mcq_option("Variable v_A should be used."), []) # Avoid 'A' in 'v_A'

    def test_case_insensitivity(self):
        self.assertEqual(extract_mcq_option("the option is (c)"), [("(c)", "C")])

    def test_various_formats(self):
        self.assertEqual(extract_mcq_option(" (A) "), [("(A)", "A")])
        self.assertEqual(extract_mcq_option("A. B. C."), [("A.", "A"), ("B.", "B"), ("C.", "C")])
        self.assertEqual(extract_mcq_option("The answer is A"), [("A", "A")])


class TestMultipleChoiceMathReward(unittest.TestCase):
    """Test the multiple_choice_math_reward function."""

    def _create_messages(self, assistant_content: str) -> List[Dict[str, str]]:
        return [
            {"role": "user", "content": "What is the answer?"},
            {"role": "assistant", "content": assistant_content}
        ]

    def test_perfect_match_parentheses(self):
        gen_msgs = self._create_messages("The correct option is (B).")
        orig_msgs = self._create_messages("The answer is (B).")
        result = multiple_choice_math_reward(messages=gen_msgs, original_messages=orig_msgs)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["metrics"]["mcq_comparison"]["success"])
        self.assertIn("Gen: '(B)' (B) vs Orig: '(B)' (B)", result["reason"])

    def test_perfect_match_dot(self):
        gen_msgs = self._create_messages("My choice is C.")
        orig_msgs = self._create_messages("C. is the one.")
        result = multiple_choice_math_reward(messages=gen_msgs, original_messages=orig_msgs)
        self.assertEqual(result["score"], 1.0)

    def test_mismatch(self):
        gen_msgs = self._create_messages("I think it's (A).")
        orig_msgs = self._create_messages("The answer is definitely (D).")
        result = multiple_choice_math_reward(messages=gen_msgs, original_messages=orig_msgs)
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["metrics"]["mcq_comparison"]["success"])
        self.assertIn("Gen: '(A)' (A) vs Orig: '(D)' (D)", result["reason"])

    def test_gen_no_mcq_orig_has_mcq(self):
        gen_msgs = self._create_messages("The answer is 42.")
        orig_msgs = self._create_messages("The answer is (A).")
        result = multiple_choice_math_reward(messages=gen_msgs, original_messages=orig_msgs)
        self.assertEqual(result["score"], 0.0)
        self.assertIn("Could not extract MCQ option from generated message", result["reason"])

    def test_orig_no_mcq(self):
        gen_msgs = self._create_messages("The answer is (B).")
        orig_msgs = self._create_messages("The answer is two.")
        result = multiple_choice_math_reward(messages=gen_msgs, original_messages=orig_msgs)
        self.assertEqual(result["score"], 0.0)
        self.assertIn("Could not extract MCQ option from original message", result["reason"])
        self.assertTrue(result["metrics"]["extracted_generated_mcq"]["success"])
        self.assertFalse(result["metrics"]["extracted_original_mcq"]["success"])
        
    def test_ambiguous_generated_answer(self):
        gen_msgs = self._create_messages("It could be (A) or maybe (B).")
        orig_msgs = self._create_messages("The answer is (A).")
        result = multiple_choice_math_reward(messages=gen_msgs, original_messages=orig_msgs)
        self.assertEqual(result["score"], 0.0) # Penalized for ambiguity
        self.assertIn("Generated answer is ambiguous", result["reason"])
        self.assertTrue(result["metrics"]["ambiguous_generated_mcq"]["success"] == False) # success is False for this metric

    def test_ambiguous_original_answer_still_compares_first(self):
        # If original is ambiguous, current logic picks the first and compares.
        gen_msgs = self._create_messages("The answer is (A).")
        orig_msgs = self._create_messages("The options are (A) and (C).")
        result = multiple_choice_math_reward(messages=gen_msgs, original_messages=orig_msgs)
        self.assertEqual(result["score"], 1.0) # Matches first extracted from original
        self.assertTrue(result["metrics"]["ambiguous_original_mcq"]["success"] == False)
        self.assertIn("Gen: '(A)' (A) vs Orig: '(A)' (A)", result["reason"])

    def test_both_ambiguous_compares_first(self):
        gen_msgs = self._create_messages("Let's say (D), or perhaps (E).")
        orig_msgs = self._create_messages("Is it (D) or (A)?")
        result = multiple_choice_math_reward(messages=gen_msgs, original_messages=orig_msgs)
        self.assertEqual(result["score"], 1.0) # D vs D
        self.assertTrue(result["metrics"]["ambiguous_generated_mcq"]["success"] == False)
        self.assertTrue(result["metrics"]["ambiguous_original_mcq"]["success"] == False)
        self.assertIn("Gen: '(D)' (D) vs Orig: '(D)' (D)", result["reason"])

    def test_empty_messages(self):
        result = multiple_choice_math_reward(messages=[], original_messages=[])
        self.assertEqual(result["score"], 0.0)
        self.assertIn("Missing messages", result["reason"])

    def test_missing_assistant_message_gen(self):
        gen_msgs = [{"role": "user", "content": "Query"}]
        orig_msgs = self._create_messages("(A)")
        result = multiple_choice_math_reward(messages=gen_msgs, original_messages=orig_msgs)
        self.assertEqual(result["score"], 0.0)
        self.assertIn("Last generated message not from assistant", result["reason"])

    def test_missing_assistant_message_orig(self):
        gen_msgs = self._create_messages("(A)")
        orig_msgs = [{"role": "user", "content": "Query"}]
        result = multiple_choice_math_reward(messages=gen_msgs, original_messages=orig_msgs)
        self.assertEqual(result["score"], 0.0)
        self.assertIn("Last original message not from assistant", result["reason"])

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
