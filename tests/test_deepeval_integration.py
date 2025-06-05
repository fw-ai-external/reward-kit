import unittest

try:
    from deepeval.metrics.base_metric import BaseMetric
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    DEEPEVAL_AVAILABLE = False

from reward_kit.integrations.deepeval import adapt_metric
from reward_kit.models import EvaluateResult


class DummyMetric(BaseMetric):
    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold
        self.score = None
        self.reason = None
        self.success = None

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        self.score = 1.0 if test_case.actual_output == test_case.expected_output else 0.0
        self.reason = "match" if self.score == 1.0 else "mismatch"
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.success is True


class TestDeepevalIntegration(unittest.TestCase):
    @unittest.skipUnless(DEEPEVAL_AVAILABLE, "deepeval package is required")
    def test_dummy_metric_wrapper(self) -> None:
        metric = DummyMetric()
        wrapped = adapt_metric(metric)
        messages = [{"role": "assistant", "content": "hi"}]
        result = wrapped(messages=messages, ground_truth="hi")
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.metrics[metric.__class__.__name__].reason, "match")


if __name__ == "__main__":
    unittest.main()
