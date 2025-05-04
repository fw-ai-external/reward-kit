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

For automated testing, you can use:
```
python examples/server_example.py --test-mode
```
This will start the server, send a test request, and then exit.
"""

import argparse
import threading
import time
import json
import requests
from typing import List, Dict, Any, Optional, Union
from reward_kit import RewardOutput, MetricRewardOutput, reward_function
from reward_kit.models import Message, EvaluateResult
from reward_kit.server import serve

@reward_function
def server_reward(
    messages: List[Message],
    **kwargs
) -> EvaluateResult:
    """
    Reward function to be served via API.
    
    This function evaluates an assistant's response based on several criteria:
    1. Length - Prefers responses of reasonable length
    2. Informativeness - Rewards responses with specific keywords or phrases
    3. Clarity - Rewards clear, structured explanations
    
    Args:
        messages: List of conversation messages
        **kwargs: Additional arguments
        
    Returns:
        EvaluateResult with score and metrics
    """
    # Get the last message content - using the Message object interface
    last_response = messages[-1].content
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
        
    metrics["length"] = {
        "score": length_score,
        "reason": length_reason
    }
    
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
        
    metrics["informativeness"] = {
        "score": informativeness_score,
        "reason": info_reason
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
        
    metrics["clarity"] = {
        "score": clarity_score,
        "reason": clarity_reason
    }
    
    # Calculate final score (weighted average)
    weights = {"length": 0.2, "informativeness": 0.5, "clarity": 0.3}
    final_score = sum(
        metrics[key]["score"] * weight 
        for key, weight in weights.items()
    )
    
    return EvaluateResult(score=final_score, metrics=metrics)


def run_test_request():
    """Send a test request to the server and return the result."""
    # Wait a moment for the server to start
    time.sleep(2)
    
    # Sample request data
    data = {
        "messages": [
            {"role": "user", "content": "Tell me about RLHF"},
            {"role": "assistant", "content": "RLHF (Reinforcement Learning from Human Feedback) is a technique to align language models with human preferences. It involves training a reward model using human feedback and then fine-tuning an LLM using reinforcement learning to maximize this learned reward function."}
        ]
    }
    
    # Send request to local server
    try:
        response = requests.post(
            "http://localhost:8000/reward",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        # Parse and return the result
        if response.status_code == 200:
            result = response.json()
            return True, result
        else:
            return False, f"Error: Status code {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Error: Could not connect to server"
    except Exception as e:
        return False, f"Error: {str(e)}"


def test_mode():
    """Run the server in test mode - start server, run test, then exit."""
    # Start server in a separate thread
    server_thread = threading.Thread(
        target=serve,
        kwargs={
            "func_path": f"{__name__}:server_reward",
            "host": "0.0.0.0",
            "port": 8000
        },
        daemon=True  # This allows the thread to be terminated when the main program exits
    )
    server_thread.start()
    
    # Send test request
    success, result = run_test_request()
    
    # Print the result
    if success:
        print("Test request successful!")
        print(f"Score: {result.get('score', 'N/A')}")
        print("Metrics:")
        for name, metric in result.get("metrics", {}).items():
            print(f"  {name}: {metric.get('score', 'N/A')} - {metric.get('reason', 'N/A')}")
    else:
        print(f"Test request failed: {result}")
    
    # Return success status (for exit code)
    return success


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Reward function server example")
    parser.add_argument("--test-mode", action="store_true", 
                       help="Run in test mode: start server, send test request, then exit")
    args = parser.parse_args()
    
    if args.test_mode:
        # Run in test mode
        print("Running server in test mode")
        success = test_mode()
        # Exit with appropriate code
        import sys
        sys.exit(0 if success else 1)
    else:
        # Normal mode - serve indefinitely
        print("Starting reward function server on http://localhost:8000")
        print("Use the /reward endpoint to evaluate messages")
        print("Try the example curl command from the docstring")
        print("Press Ctrl+C to stop the server")
        
        # In a real deployment, you would provide the module path
        # to the function rather than using __name__
        module_path = __name__
        function_name = "server_reward"
        func_path = f"{module_path}:{function_name}"
        
        serve(func_path=func_path, host="0.0.0.0", port=8000)