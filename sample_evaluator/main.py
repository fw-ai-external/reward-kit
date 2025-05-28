from typing import List, Dict, Any
from reward_kit import reward_function, EvaluateResult, Message, MetricResult

@reward_function
def evaluate(messages: List[Message], **kwargs: Any) -> EvaluateResult:
    """
    A simple reward function that rewards based on the length of the last assistant message.
    """
    if not messages or messages[-1].role != "assistant":
        return EvaluateResult(
            score=0.0,
            reason="No assistant response found.",
            metrics={"length": MetricResult(score=0.0, success=False, reason="No assistant response.")}
        )

    assistant_message_content = messages[-1].content or ""
    length = len(assistant_message_content)
    score = min(1.0, length / 100.0) # Score is 1.0 if length is 100 or more

    return EvaluateResult(
        score=score,
        reason=f"Assistant response length: {length} characters.",
        metrics={"length": MetricResult(score=score, success=length > 0, reason=f"Length: {length}")}
    )
