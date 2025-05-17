"""
Test script for the server example that doesn't actually start a server.
This is used by run_all_examples.py to test the server functionality
without having to deal with threading or servers.
"""

from typing import Any, Dict, List, Optional

from reward_kit import reward_function
from reward_kit.models import EvaluateResult, Message


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
    last_response = messages[-1].content
    metrics = {}

    # 1. Length score
    response_length = len(last_response)
    length_score = min(
        response_length / 500, 1.0
    )  # Cap at 1.0 for responses ≥ 500 chars

    if response_length < 50:
        length_reason = "Response is too short"
    elif response_length < 200:
        length_reason = "Response is somewhat brief"
    elif response_length < 500:
        length_reason = "Response has good length"
    else:
        length_reason = "Response is comprehensive"

    metrics["length"] = {"score": length_score, "reason": length_reason}

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

    if found_keywords:
        info_reason = f"Found informative keywords: {', '.join(found_keywords)}"
    else:
        info_reason = "No informative keywords detected"

    metrics["informativeness"] = {
        "score": informativeness_score,
        "reason": info_reason,
    }

    # 3. Clarity score (simple heuristic - paragraphs, bullet points, headings add clarity)
    has_paragraphs = len(last_response.split("\n\n")) > 1
    has_bullets = "* " in last_response or "- " in last_response
    has_structure = has_paragraphs or has_bullets

    clarity_score = 0.5  # Base score
    if has_structure:
        clarity_score += 0.5
        clarity_reason = "Response has good structure with paragraphs or bullet points"
    else:
        clarity_reason = "Response could be improved with better structure"

    metrics["clarity"] = {"score": clarity_score, "reason": clarity_reason}

    # Calculate final score (weighted average)
    weights = {"length": 0.2, "informativeness": 0.5, "clarity": 0.3}
    final_score = sum(metrics[key]["score"] * weight for key, weight in weights.items())

    return EvaluateResult(score=final_score, metrics=metrics)


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
