"""
Pytest configuration file for reward-kit tests.
"""

import pytest
from typing import List, Dict, Any, Optional
from reward_kit.models import RewardOutput, MetricRewardOutput


@pytest.fixture
def sample_messages():
    """Sample conversation messages."""
    return [
        {"role": "user", "content": "What is the weather like today?"},
        {
            "role": "assistant",
            "content": "I don't have real-time weather data. You should check a weather service.",
        },
    ]


@pytest.fixture
def sample_original_messages(sample_messages):
    """Sample original (user only) messages."""
    return [sample_messages[0]]


@pytest.fixture
def sample_reward_output():
    """Sample reward output structure."""
    metrics = {
        "helpfulness": MetricRewardOutput(
            score=0.7, reason="Response acknowledges limitations"
        ),
        "accuracy": MetricRewardOutput(
            score=0.8,
            reason="Response correctly states lack of access to weather data",
        ),
    }
    return RewardOutput(score=0.75, metrics=metrics)


@pytest.fixture
def sample_function_call_schema():
    """Sample function call schema for testing."""
    return {
        "name": "get_weather",
        "arguments": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
    }
