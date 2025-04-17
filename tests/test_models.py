import pytest
import json
from typing import Dict
from reward_kit.models import MetricRewardOutput, RewardOutput


def test_metric_reward_output_creation():
    """Test creating a MetricRewardOutput."""
    metric = MetricRewardOutput(score=0.5, reason="Test reason")
    assert metric.score == 0.5
    assert metric.reason == "Test reason"


def test_metric_reward_output_serialization():
    """Test serializing MetricRewardOutput to JSON."""
    metric = MetricRewardOutput(score=0.75, reason="Test serialization")
    json_str = metric.to_json()
    data = json.loads(json_str)
    assert data["score"] == 0.75
    assert data["reason"] == "Test serialization"


def test_metric_reward_output_deserialization():
    """Test deserializing MetricRewardOutput from JSON."""
    json_str = '{"score": 0.9, "reason": "Test deserialization"}'
    metric = MetricRewardOutput.from_json(json_str)
    assert metric.score == 0.9
    assert metric.reason == "Test deserialization"


def test_reward_output_creation():
    """Test creating a RewardOutput."""
    metrics: Dict[str, MetricRewardOutput] = {
        "metric1": MetricRewardOutput(score=0.5, reason="Reason 1"),
        "metric2": MetricRewardOutput(score=0.7, reason="Reason 2")
    }
    reward = RewardOutput(score=0.6, metrics=metrics)
    assert reward.score == 0.6
    assert len(reward.metrics) == 2
    assert reward.metrics["metric1"].score == 0.5
    assert reward.metrics["metric2"].reason == "Reason 2"


def test_reward_output_serialization():
    """Test serializing RewardOutput to JSON."""
    metrics = {
        "metric1": MetricRewardOutput(score=0.5, reason="Reason 1"),
        "metric2": MetricRewardOutput(score=0.7, reason="Reason 2")
    }
    reward = RewardOutput(score=0.6, metrics=metrics)
    json_str = reward.to_json()
    data = json.loads(json_str)
    assert data["score"] == 0.6
    assert len(data["metrics"]) == 2
    assert data["metrics"]["metric1"]["score"] == 0.5
    assert data["metrics"]["metric2"]["reason"] == "Reason 2"


def test_reward_output_deserialization():
    """Test deserializing RewardOutput from JSON."""
    json_str = (
        '{"score": 0.8, "metrics": {'
        '"metric1": {"score": 0.4, "reason": "Reason A"}, '
        '"metric2": {"score": 0.9, "reason": "Reason B"}'
        '}}'
    )
    reward = RewardOutput.from_json(json_str)
    assert reward.score == 0.8
    assert len(reward.metrics) == 2
    assert reward.metrics["metric1"].score == 0.4
    assert reward.metrics["metric2"].reason == "Reason B"


def test_empty_metrics():
    """Test RewardOutput with empty metrics dictionary."""
    reward = RewardOutput(score=1.0, metrics={})
    assert reward.score == 1.0
    assert reward.metrics == {}
    
    json_str = reward.to_json()
    data = json.loads(json_str)
    assert data["score"] == 1.0
    assert data["metrics"] == {}