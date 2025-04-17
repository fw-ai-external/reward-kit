from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import json

@dataclass_json
@dataclass
class MetricRewardOutput:
    """
    Individual component metric for reward output.
    
    Args:
        score: The score value for this metric component
        reason: Optional explanation for why this score was assigned
    """
    score: float
    reason: Optional[str] = None

@dataclass_json
@dataclass
class RewardOutput:
    """
    Complete output from a reward function including overall score and component metrics.
    
    Args:
        score: The final aggregate reward score
        metrics: Dictionary of component metrics and their individual scores/reasons
    """
    score: float
    metrics: Dict[str, MetricRewardOutput] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert RewardOutput to a dictionary representation."""
        return {
            "score": self.score,
            "metrics": {
                k: {"score": v.score, "reason": v.reason} 
                for k, v in self.metrics.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RewardOutput":
        """Create a RewardOutput from a dictionary representation."""
        metrics = {
            k: MetricRewardOutput(
                score=v.get("score", 0.0),
                reason=v.get("reason")
            )
            for k, v in data.get("metrics", {}).items()
        }
        return cls(score=data.get("score", 0.0), metrics=metrics)
    
    def __str__(self) -> str:
        """String representation of the reward output."""
        return json.dumps(self.to_dict(), indent=2)