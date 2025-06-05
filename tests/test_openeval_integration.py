import unittest

from openevals import exact_match
from openevals.string import levenshtein_distance
from openevals.json import create_json_match_evaluator

from reward_kit.integrations.openeval import adapt
from reward_kit.models import EvaluateResult


class TestOpenEvalIntegration(unittest.TestCase):
    def test_exact_match_wrapper(self) -> None:
        wrapped = adapt(exact_match)
        messages = [{"role": "assistant", "content": "hi"}]
        result = wrapped(messages=messages, ground_truth="hi")
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 1.0)

    def test_levenshtein_distance_wrapper(self) -> None:
        wrapped = adapt(levenshtein_distance)
        messages = [{"role": "assistant", "content": "foo"}]
        result = wrapped(messages=messages, ground_truth="fooo")
        self.assertIsInstance(result, EvaluateResult)
        self.assertAlmostEqual(result.score, 0.75)

    def test_json_match_wrapper(self) -> None:
        evaluator = create_json_match_evaluator(aggregator="average")
        wrapped = adapt(evaluator)
        messages = [{"role": "assistant", "content": {"a": 1, "b": 2}}]
        result = wrapped(messages=messages, ground_truth={"a": 1, "b": 2})
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 1.0)


if __name__ == "__main__":
    unittest.main()
