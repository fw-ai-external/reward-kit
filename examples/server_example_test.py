"""
Test script for the server example that doesn't actually start a server.
This is used by run_all_examples.py to test the server functionality
without having to deal with threading or servers.
"""

from typing import Any, Dict, List, Optional

from reward_kit import reward_function
from reward_kit.models import (  # Added MetricResult
    EvaluateResult,
    Message,
    MetricResult,
)


@reward_function
def server_reward(messages: List[Message], **kwargs) -> EvaluateResult:
    """
    Reward function that would be served via API.

    This function evaluates an assistant's response based on several criteria:
    1. Length - Prefers responses of reasonable length
    2. Informativeness - Rewards responses with specific keywords or phrases
    3. Clarity - Rewards clear, structured explanations
    """
    # Get the last message content
    last_response_content = messages[-1].content
    last_response = (
        last_response_content if last_response_content is not None else ""
    )  # Default to empty string if None
    metrics = {}

    # 1. Length score
    response_length = len(last_response)  # Now last_response is guaranteed to be str
    length_score = min(
        response_length / 500, 1.0
    )  # Cap at 1.0 for responses ≥ 500 chars
    length_success = True  # Default, adjust based on logic if needed
    if response_length < 50:
        length_reason = "Response is too short"
        length_success = False
    elif response_length < 200:
        length_reason = "Response is somewhat brief"
        length_success = False  # Or True depending on desired strictness
    elif response_length < 500:
        length_reason = "Response has good length"
    else:
        length_reason = "Response is comprehensive"

    metrics["length"] = MetricResult(
        score=length_score, is_score_valid=length_success, reason=length_reason
    )

    # 2. Informativeness score
    # Keywords that suggest an informative response about RLHF
    keywords = [
        "reinforcement learning",
        "human feedback",
        "reward model",
        "preference",
        "fine-tuning",
        "alignment",
        "training",
    ]

    found_keywords = [kw for kw in keywords if kw.lower() in last_response.lower()]
    informativeness_score = min(
        len(found_keywords) / 4, 1.0
    )  # Cap at 1.0 for ≥4 keywords
    info_success = len(found_keywords) > 0

    if found_keywords:
        info_reason = f"Found informative keywords: {', '.join(found_keywords)}"
    else:
        info_reason = "No informative keywords detected"

    metrics["informativeness"] = MetricResult(
        score=informativeness_score,
        is_score_valid=info_success,
        reason=info_reason,
    )

    # 3. Clarity score (simple heuristic - paragraphs, bullet points, headings add clarity)
    has_paragraphs = len(last_response.split("\n\n")) > 1
    has_bullets = "* " in last_response or "- " in last_response
    has_structure = has_paragraphs or has_bullets

    clarity_score = 0.5  # Base score
    clarity_success = False
    if has_structure:
        clarity_score += 0.5
        clarity_reason = "Response has good structure with paragraphs or bullet points"
        clarity_success = True
    else:
        clarity_reason = "Response could be improved with better structure"

    metrics["clarity"] = MetricResult(
        score=clarity_score, is_score_valid=clarity_success, reason=clarity_reason
    )

    # Calculate final score (weighted average)
    weights = {"length": 0.2, "informativeness": 0.5, "clarity": 0.3}
    final_score = sum(
        metrics[key].score * weight for key, weight in weights.items()
    )  # Access .score
    overall_reason = f"Final score based on weighted average of length ({metrics['length'].score:.2f}), informativeness ({metrics['informativeness'].score:.2f}), and clarity ({metrics['clarity'].score:.2f})."

    return EvaluateResult(
        score=final_score, reason=overall_reason, metrics=metrics, is_score_valid=True
    )


if __name__ == "__main__":
    """
    This is a test script for the server example that doesn't actually start a server.
    It just runs the reward function directly on a test message.
    """
    test_messages = [
        {"role": "user", "content": "Tell me about RLHF"},
        {
            "role": "assistant",
            "content": "RLHF (Reinforcement Learning from Human Feedback) is a technique to align language models with human preferences. It involves training a reward model using human feedback and then fine-tuning an LLM using reinforcement learning to maximize this learned reward function.",
        },
    ]

    # Run the reward function
    result = server_reward(messages=test_messages)

    # Display the result
    print("Score:", result["score"])
    print("Metrics:")
    for name, metric in result["metrics"].items():
        print(f"  {name}: {metric['score']} - {metric['reason']}")

    print(
        "\nThis is a demonstration of the reward function that would normally be served via API."
    )
    print("To run the actual server, use: python examples/server_example.py")
