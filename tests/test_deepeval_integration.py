import unittest

try:
    from deepeval.metrics.base_metric import BaseMetric
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    DEEPEVAL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    DEEPEVAL_AVAILABLE = False

if not DEEPEVAL_AVAILABLE:  # Provide dummy classes so the module imports
    class BaseMetric:  # type: ignore
        pass

    class LLMTestCase:  # type: ignore[too-many-instance-attributes]
        def __init__(self, input: str = "", actual_output: str = "", expected_output: str = "") -> None:
            self.input = input
            self.actual_output = actual_output
            self.expected_output = expected_output

    class LLMTestCaseParams:
        INPUT = type("Enum", (), {"value": "input"})
        ACTUAL_OUTPUT = type("Enum", (), {"value": "actual_output"})
        EXPECTED_OUTPUT = type("Enum", (), {"value": "expected_output"})

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


class DummyGEval(BaseMetric):
    evaluation_params = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ]

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold
        self.score = None
        self.reason = None
        self.success = None
        self.last_case = None

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        self.last_case = test_case
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

    @unittest.skipUnless(DEEPEVAL_AVAILABLE, "deepeval package is required")
    def test_dummy_geval_wrapper(self) -> None:
        metric = DummyGEval()
        wrapped = adapt_metric(metric)
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hi"},
        ]
        result = wrapped(messages=messages, ground_truth="hi")
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(metric.last_case.input, "hi")


if __name__ == "__main__":
    unittest.main()
