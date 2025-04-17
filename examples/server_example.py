"""
Example of running a reward function server.

To run this example:
1. Make sure reward-kit is installed: `pip install -e .`
2. Run this script: `python examples/server_example.py`
3. In another terminal, test the server:

```
curl -X POST http://localhost:8000/reward \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Tell me about RLHF"},
      {"role": "assistant", "content": "RLHF (Reinforcement Learning from Human Feedback) is a technique to align language models with human preferences. It involves training a reward model using human feedback and then fine-tuning an LLM using reinforcement learning to maximize this learned reward function."}
    ]
  }'
```
"""

from typing import List, Dict, Any, Optional
from reward_kit import RewardOutput, MetricRewardOutput, reward_function
from reward_kit.server import serve

@reward_function
def server_reward(
    messages: List[Dict[str, str]],
    original_messages: List[Dict[str, str]],
    **kwargs
) -> RewardOutput:
    """
    Reward function to be served via API.
    
    This function evaluates an assistant's response based on several criteria:
    1. Length - Prefers responses of reasonable length
    2. Informativeness - Rewards responses with specific keywords or phrases
    3. Clarity - Rewards clear, structured explanations
    
    Args:
        messages: List of conversation messages
        original_messages: Original conversation context
        **kwargs: Additional arguments
        
    Returns:
        RewardOutput with score and metrics
    """
    last_response = messages[-1]["content"]
    metrics = {}
    
    # 1. Length score
    response_length = len(last_response)
    length_score = min(response_length / 500, 1.0)  # Cap at 1.0 for responses ≥ 500 chars
    
    if response_length < 50:
        length_reason = "Response is too short"
    elif response_length < 200:
        length_reason = "Response is somewhat brief"
    elif response_length < 500:
        length_reason = "Response has good length"
    else:
        length_reason = "Response is comprehensive"
        
    metrics["length"] = MetricRewardOutput(
        score=length_score,
        reason=length_reason
    )
    
    # 2. Informativeness score
    # Keywords that suggest an informative response about RLHF
    keywords = [
        "reinforcement learning", "human feedback", "reward model", 
        "preference", "fine-tuning", "alignment", "training"
    ]
    
    found_keywords = [kw for kw in keywords if kw.lower() in last_response.lower()]
    informativeness_score = min(len(found_keywords) / 4, 1.0)  # Cap at 1.0 for ≥4 keywords
    
    if found_keywords:
        info_reason = f"Found informative keywords: {', '.join(found_keywords)}"
    else:
        info_reason = "No informative keywords detected"
        
    metrics["informativeness"] = MetricRewardOutput(
        score=informativeness_score,
        reason=info_reason
    )
    
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
        
    metrics["clarity"] = MetricRewardOutput(
        score=clarity_score,
        reason=clarity_reason
    )
    
    # Calculate final score (weighted average)
    weights = {"length": 0.2, "informativeness": 0.5, "clarity": 0.3}
    final_score = sum(
        metrics[key].score * weight 
        for key, weight in weights.items()
    )
    
    return RewardOutput(score=final_score, metrics=metrics)


if __name__ == "__main__":
    # Serve the reward function
    print("Starting reward function server on http://localhost:8000")
    print("Use the /reward endpoint to evaluate messages")
    print("Try the example curl command from the docstring")
    
    # In a real deployment, you would provide the module path
    # to the function rather than using __name__
    module_path = __name__
    function_name = "server_reward"
    func_path = f"{module_path}:{function_name}"
    
    serve(func_path=func_path, host="0.0.0.0", port=8000)