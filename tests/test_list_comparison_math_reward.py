import unittest
from typing import List, Dict, Any, Optional

from reward_kit.rewards.list_comparison_math_reward import (
    list_comparison_math_reward,
    extract_number_list,
    parse_number_list_from_string
)
from reward_kit.models import EvaluateResult, Message

class TestParseNumberListFromString(unittest.TestCase):
    """Test the raw string to number list parsing utility."""
    def test_simple_list(self):
        self.assertEqual(parse_number_list_from_string("1,2,3"), [1.0, 2.0, 3.0])
        self.assertEqual(parse_number_list_from_string("1, 2.5, 3"), [1.0, 2.5, 3.0])

    def test_with_spaces(self):
        self.assertEqual(parse_number_list_from_string("  1 , 2 , 3  "), [1.0, 2.0, 3.0])

    def test_with_dollar_signs(self):
        # parse_number_list_from_string itself removes $
        self.assertEqual(parse_number_list_from_string("$1,2,3$"), [1.0, 2.0, 3.0])
        self.assertEqual(parse_number_list_from_string("1, $2$, 3"), [1.0, 2.0, 3.0])

    def test_single_number(self):
        self.assertEqual(parse_number_list_from_string("42"), [42.0])
        self.assertEqual(parse_number_list_from_string("$123.45$"), [123.45])

    def test_empty_and_invalid(self):
        self.assertIsNone(parse_number_list_from_string(""))
        self.assertIsNone(parse_number_list_from_string("  "))
        self.assertIsNone(parse_number_list_from_string("1,two,3"))
        self.assertIsNone(parse_number_list_from_string("1, 2, three"))
        self.assertIsNone(parse_number_list_from_string("abc"))

    def test_list_with_empty_elements(self):
        # "1,,2" -> [1.0, 2.0] because empty string from split is skipped
        self.assertEqual(parse_number_list_from_string("1,,2"), [1.0, 2.0])
        self.assertEqual(parse_number_list_from_string("1, ,2"), [1.0, 2.0])


class TestExtractNumberList(unittest.TestCase):
    """Test the number list extraction from text with delimiters."""

    def test_extract_from_boxed(self):
        self.assertEqual(extract_number_list("The answer is \\boxed{1,2,3}."), [[1.0, 2.0, 3.0]])
        self.assertEqual(extract_number_list("\\boxed{1,2} and \\boxed{3,4}"), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(extract_number_list("Boxed: \\boxed{1, 2.5, 3}"), [[1.0, 2.5, 3.0]])

    def test_extract_from_dollar(self):
        self.assertEqual(extract_number_list("The set is $1,2,3$."), [[1.0, 2.0, 3.0]])
        self.assertEqual(extract_number_list("Numbers are $$1, 2, 3$$"), [[1.0, 2.0, 3.0]])
        self.assertEqual(extract_number_list("$1,2$ and also $3,4$"), [[1.0, 2.0], [3.0, 4.0]])

    def test_priority_boxed_over_dollar(self):
        # If boxed is found, dollar outside is ignored by current logic (returns after finding in boxed)
        self.assertEqual(extract_number_list("$\\boxed{1,2}$ $3,4$"), [[1.0, 2.0]])
        # If multiple boxed, all are returned
        self.assertEqual(extract_number_list("$\\boxed{1,2}$ and \\boxed{5,6} $3,4$"), [[1.0, 2.0], [5.0, 6.0]])


    def test_priority_dollar_over_full_text(self):
        self.assertEqual(extract_number_list("The list $1,2,3$ is correct, not 4,5,6."), [[1.0, 2.0, 3.0]])

    def test_extract_from_full_text_fallback(self):
        # Fallback only works if the *entire string* (after $ stripping) is a valid list
        self.assertEqual(extract_number_list("1, 2.5, 3"), [[1.0, 2.5, 3.0]])
        self.assertEqual(extract_number_list("  $1, 2, 3$  "), [[1.0, 2.0, 3.0]]) # Handled by parse_number_list_from_string
        # These should not parse as lists because of surrounding text
        self.assertEqual(extract_number_list("The numbers are 1,2,3."), []) 
        self.assertEqual(extract_number_list("The number is 123."), [])


    def test_no_valid_list(self):
        self.assertEqual(extract_number_list("No numbers here."), [])
        self.assertEqual(extract_number_list("\\boxed{a,b,c}"), [])
        self.assertEqual(extract_number_list("$one, two$"), [])
        # This case is covered by test_extract_from_full_text_fallback, should be []
        # self.assertEqual(extract_number_list("The number is 123."), [[123.0]]) 
        self.assertEqual(extract_number_list("The number is 123."), [])


    def test_mixed_delimiters_behavior(self):
        # Current logic: if boxed found, returns only boxed. If no boxed but dollar found, returns only dollar.
        self.assertEqual(extract_number_list("Box \\boxed{1,2} and dollar $3,4$"), [[1.0, 2.0]])
        self.assertEqual(extract_number_list("Dollar $3,4$ and text 5,6"), [[3.0, 4.0]])


class TestListComparisonMathReward(unittest.TestCase):
    """Test the list_comparison_math_reward function."""

    def _create_messages(self, assistant_content: str) -> List[Dict[str, str]]:
        return [
            {"role": "user", "content": "What are the numbers?"},
            {"role": "assistant", "content": assistant_content}
        ]

    def test_exact_match_set_comparison(self):
        # Default: order_matters=False (set comparison)
        gen_msgs = self._create_messages("The divisors are \\boxed{1,2,3}.")
        orig_msgs = self._create_messages("Answer: $3,1,2$")
        result = list_comparison_math_reward(messages=gen_msgs, original_messages=orig_msgs)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["score"], 1.0)
        self.assertIn("Set match", result["reason"])
        self.assertIn("Gen: [1.0, 2.0, 3.0] vs Orig: [1.0, 2.0, 3.0]", result["reason"])

    def test_exact_match_list_comparison(self):
        gen_msgs = self._create_messages("\\boxed{1,2,3}")
        orig_msgs = self._create_messages("$1,2,3$")
        result = list_comparison_math_reward(messages=gen_msgs, original_messages=orig_msgs, order_matters=True)
        self.assertEqual(result["score"], 1.0)
        self.assertIn("Exact list match", result["reason"])

    def test_mismatch_set_comparison(self):
        gen_msgs = self._create_messages("$1,2,4$")
        orig_msgs = self._create_messages("\\boxed{1,2,3}")
        result = list_comparison_math_reward(messages=gen_msgs, original_messages=orig_msgs)
        self.assertEqual(result["score"], 0.0)
        self.assertIn("Set mismatch", result["reason"])
        self.assertIn("Missing in generated: [3.0]", result["reason"])
        self.assertIn("Extra in generated: [4.0]", result["reason"])

    def test_mismatch_list_comparison_order(self):
        gen_msgs = self._create_messages("$1,3,2$")
        orig_msgs = self._create_messages("\\boxed{1,2,3}")
        result = list_comparison_math_reward(messages=gen_msgs, original_messages=orig_msgs, order_matters=True)
        self.assertEqual(result["score"], 0.0)
        self.assertIn("List mismatch (order matters)", result["reason"])

    def test_mismatch_list_comparison_value(self):
        gen_msgs = self._create_messages("$1,2,4$")
        orig_msgs = self._create_messages("\\boxed{1,2,3}")
        result = list_comparison_math_reward(messages=gen_msgs, original_messages=orig_msgs, order_matters=True)
        self.assertEqual(result["score"], 0.0)
        self.assertIn("List mismatch (order matters)", result["reason"])

    def test_subset_set_comparison(self):
        # Gen is subset of Orig
        gen_msgs = self._create_messages("$1,2$")
        orig_msgs = self._create_messages("\\boxed{1,2,3}")
        result = list_comparison_math_reward(messages=gen_msgs, original_messages=orig_msgs)
        self.assertEqual(result["score"], 0.0)
        self.assertIn("Missing in generated: [3.0]", result["reason"])
        self.assertNotIn("Extra in generated", result["reason"])

    def test_superset_set_comparison(self):
        # Gen is superset of Orig
        gen_msgs = self._create_messages("$1,2,3,4$")
        orig_msgs = self._create_messages("\\boxed{1,2,3}")
        result = list_comparison_math_reward(messages=gen_msgs, original_messages=orig_msgs)
        self.assertEqual(result["score"], 0.0)
        self.assertIn("Extra in generated: [4.0]", result["reason"])
        self.assertNotIn("Missing in generated", result["reason"])

    def test_no_list_in_gen(self):
        gen_msgs = self._create_messages("The answer is three.")
        orig_msgs = self._create_messages("\\boxed{1,2,3}")
        result = list_comparison_math_reward(messages=gen_msgs, original_messages=orig_msgs)
        self.assertEqual(result["score"], 0.0)
        self.assertIn("Could not extract any number list from generated message", result["reason"])

    def test_no_list_in_orig(self):
        gen_msgs = self._create_messages("\\boxed{1,2,3}")
        orig_msgs = self._create_messages("The answer is three.")
        result = list_comparison_math_reward(messages=gen_msgs, original_messages=orig_msgs)
        self.assertEqual(result["score"], 0.0)
        self.assertIn("Could not extract any number list from original message", result["reason"])

    def test_example_from_issue(self):
        # "1,3,59,177"
        gen_msgs = self._create_messages("$1,3,59,177$")
        orig_msgs = self._create_messages("1,3,59,177") # Fallback extraction for original
        result = list_comparison_math_reward(messages=gen_msgs, original_messages=orig_msgs)
        self.assertEqual(result["score"], 1.0)
        self.assertIn("Set match", result["reason"])
        self.assertIn("Gen: [1.0, 3.0, 59.0, 177.0] vs Orig: [1.0, 3.0, 59.0, 177.0]", result["reason"])

    def test_empty_messages(self):
        result = list_comparison_math_reward(messages=[], original_messages=[])
        self.assertEqual(result["score"], 0.0)
        self.assertIn("Missing messages", result["reason"])

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
