import pytest
import json
from typing import Dict
from reward_kit.models import MetricResult, EvaluateResult


def test_metric_result_creation():
    """Test creating a MetricResult."""
    metric = MetricResult(score=0.5, reason="Test reason", success=False)
    assert metric.score == 0.5
    assert metric.reason == "Test reason"
    assert metric.success is False


def test_metric_result_serialization():
    """Test serializing MetricResult to JSON."""
    metric = MetricResult(score=0.75, reason="Test serialization", success=True)
    json_str = metric.model_dump_json()
    data = json.loads(json_str)
    assert data["score"] == 0.75
    assert data["reason"] == "Test serialization"
    assert data["success"] is True


def test_metric_result_deserialization():
    """Test deserializing MetricResult from JSON."""
    json_str = '{"score": 0.9, "reason": "Test deserialization", "success": null}'
    metric = MetricResult.model_validate_json(json_str)
    assert metric.score == 0.9
    assert metric.reason == "Test deserialization"
    assert metric.success is None


def test_evaluate_result_creation():
    """Test creating an EvaluateResult."""
    metrics: Dict[str, MetricResult] = {
        "metric1": MetricResult(score=0.5, reason="Reason 1", success=False),
        "metric2": MetricResult(score=0.7, reason="Reason 2", success=True),
    }
    result = EvaluateResult(score=0.6, reason="Overall assessment", metrics=metrics)
    assert result.score == 0.6
    assert result.reason == "Overall assessment"
    assert len(result.metrics) == 2
    assert result.metrics["metric1"].score == 0.5
    assert result.metrics["metric2"].reason == "Reason 2"
    assert result.metrics["metric2"].success is True


def test_evaluate_result_serialization():
    """Test serializing EvaluateResult to JSON."""
    metrics = {
        "metric1": MetricResult(score=0.5, reason="Reason 1", success=False),
        "metric2": MetricResult(score=0.7, reason="Reason 2", success=True),
    }
    result = EvaluateResult(score=0.6, reason="Overall assessment", metrics=metrics)
    json_str = result.model_dump_json()
    data = json.loads(json_str)
    assert data["score"] == 0.6
    assert data["reason"] == "Overall assessment"
    assert len(data["metrics"]) == 2
    assert data["metrics"]["metric1"]["score"] == 0.5
    assert data["metrics"]["metric1"]["success"] is False
    assert data["metrics"]["metric2"]["reason"] == "Reason 2"


def test_evaluate_result_deserialization():
    """Test deserializing EvaluateResult from JSON."""
    json_str = (
        '{"score": 0.8, "reason": "Overall", "metrics": {'
        '"metric1": {"score": 0.4, "reason": "Reason A", "success": false}, '
        '"metric2": {"score": 0.9, "reason": "Reason B", "success": true}'
        '}, "error": null}'
    )
    result = EvaluateResult.model_validate_json(json_str)
    assert result.score == 0.8
    assert result.reason == "Overall"
    assert len(result.metrics) == 2
    assert result.metrics["metric1"].score == 0.4
    assert result.metrics["metric1"].success is False
    assert result.metrics["metric2"].reason == "Reason B"
    assert result.error is None


def test_empty_metrics_evaluate_result():
    """Test EvaluateResult with empty metrics dictionary."""
    result = EvaluateResult(score=1.0, reason="Perfect score", metrics={})
    assert result.score == 1.0
    assert result.reason == "Perfect score"
    assert result.metrics == {}

    json_str = result.model_dump_json()
    data = json.loads(json_str)
    assert data["score"] == 1.0
    assert data["reason"] == "Perfect score"
    assert data["metrics"] == {}
