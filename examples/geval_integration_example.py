"""GEval integration example using Reward Kit.

This script demonstrates how to adapt a deepeval GEval metric into a reward
function for LLM-as-a-judge evaluation. It configures the GEval metric to use
Fireworks' ``accounts/fireworks/models/qwen3-235b-a22b`` model as the judge.
Make sure the ``FIREWORKS_API_KEY`` environment variable is set before running
the script.
"""

import os

from deepeval.metrics import GEval
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCaseParams

from reward_kit.integrations.deepeval import adapt_metric


def main() -> None:
    """Run a simple GEval evaluation."""
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY environment variable must be set.")

    fireworks_llm = GPTModel(
        model="accounts/fireworks/models/qwen3-235b-a22b",
        _openai_api_key=api_key,
        base_url="https://api.fireworks.ai/inference/v1",
    )

    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine whether the answer is factually correct",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=fireworks_llm,
    )

    reward_fn = adapt_metric(correctness_metric)

    messages = [
        {"role": "user", "content": "Translate 'Hello, world!' to Spanish."},
        {"role": "assistant", "content": "Hola, mundo!"},
    ]

    result = reward_fn(messages=messages, ground_truth="Hola, mundo!")

    print(f"Score: {result.score}")
    if result.reason:
        print(f"Reason: {result.reason}")


if __name__ == "__main__":
    main()
