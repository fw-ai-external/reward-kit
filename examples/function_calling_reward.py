"""
Example of using the function calling reward.
"""

from typing import List, Dict, Any, Tuple, Optional
import json
from reward_kit import RewardFunction, RewardOutput, MetricRewardOutput, reward_function
from reward_kit.rewards import function_calling

# Define a weather function schema
weather_function_schema = {
    "name": "get_weather",
    "arguments": {
        "location": {"type": "string"},
        "unit": {"type": "string"},
        "days": {"type": "number"}
    }
}

def parse_potential_function_call(text: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Parse a potential function call from text.
    
    This is a simple parser for demonstration purposes. In a real-world scenario,
    you would want to use a more robust parser or an LLM-specific format.
    
    Args:
        text: The text to parse
        
    Returns:
        A tuple of (is_function_call, parsed_call)
    """
    # Try to find json-like content (very simple approach for example)
    try:
        # Look for content between ```json and ``` markers
        if "```json" in text and "```" in text.split("```json", 1)[1]:
            json_str = text.split("```json", 1)[1].split("```", 1)[0].strip()
            data = json.loads(json_str)
            
            # Check if it has the expected structure
            if "name" in data and "arguments" in data:
                return True, data
    except (json.JSONDecodeError, IndexError):
        pass
        
    # Try a fallback approach - look for function call pattern
    # Very basic regex-like search (not actual regex for simplicity)
    if "get_weather" in text and "{" in text and "}" in text:
        try:
            # Try to extract the JSON object that might contain the arguments
            json_str = text[text.find("{"):text.rfind("}")+1]
            args = json.loads(json_str)
            return True, {"name": "get_weather", "arguments": args}
        except (json.JSONDecodeError, IndexError):
            pass
    
    return False, {}


@reward_function
def function_call_reward(
    messages: List[Dict[str, str]],
    original_messages: List[Dict[str, str]],
    expected_schema: Dict[str, Any],
    strictness: str = "exact",
    **kwargs
) -> RewardOutput:
    """
    Evaluate a function call against an expected schema.
    
    Args:
        messages: List of conversation messages
        original_messages: Original conversation context
        expected_schema: The expected function call schema
        strictness: How strict to be with matching ("exact", "partial", "flexible")
        
    Returns:
        RewardOutput with score and metrics
    """
    last_response = messages[-1]["content"]
    
    # Parse the function call
    is_function_call, parsed_call = parse_potential_function_call(last_response)
    
    if not is_function_call:
        # If it's not a function call, return a low score
        return RewardOutput(
            score=0.0,
            metrics={
                "is_function_call": MetricRewardOutput(
                    score=0.0,
                    reason="No function call detected in the response"
                )
            }
        )
    
    # Use the OOTB function calling reward
    return function_calling.match_function_call(
        messages=messages,
        original_messages=original_messages,
        function_name=parsed_call.get("name", ""),
        parsed_arguments=parsed_call.get("arguments", {}),
        expected_call_schema=expected_schema,
        argument_match_strictness=strictness
    )


if __name__ == "__main__":
    # Example LLM responses
    perfect_response = {
        "role": "assistant", 
        "content": """To get the weather, I'll call the weather API with the provided location.
```json
{
  "name": "get_weather",
  "arguments": {
    "location": "San Francisco",
    "unit": "celsius",
    "days": 5
  }
}
```
This will retrieve a 5-day forecast for San Francisco in Celsius.
"""
    }
    
    missing_arg_response = {
        "role": "assistant", 
        "content": """I'll check the weather for you.
```json
{
  "name": "get_weather",
  "arguments": {
    "location": "San Francisco",
    "unit": "celsius"
  }
}
```
"""
    }
    
    wrong_name_response = {
        "role": "assistant", 
        "content": """Let me get that weather information.
```json
{
  "name": "fetch_weather",
  "arguments": {
    "location": "San Francisco",
    "unit": "celsius",
    "days": 5
  }
}
```
"""
    }
    
    not_function_response = {
        "role": "assistant", 
        "content": "The weather in San Francisco is usually mild, with temperatures ranging from 10-20°C throughout the year."
    }
    
    # Test messages
    test_messages = [
        {"role": "user", "content": "What's the weather like in San Francisco for the next 5 days?"}
    ]
    
    # Create a reward function instance with the weather schema
    reward_fn = RewardFunction(
        func=function_call_reward,
        mode="local",
        expected_schema=weather_function_schema,
        strictness="exact"
    )
    
    # Test all responses
    for name, response in [
        ("Perfect", perfect_response),
        ("Missing Argument", missing_arg_response),
        ("Wrong Function Name", wrong_name_response),
        ("Not a Function Call", not_function_response)
    ]:
        test_with_response = test_messages + [response]
        result = reward_fn(messages=test_with_response, original_messages=test_messages)
        
        print(f"\n{name} Response Result:")
        print(f"Score: {result.score}")
        print("Metrics:")
        for metric_name, metric in result.metrics.items():
            print(f"  {metric_name}: {metric.score}")
            print(f"    Reason: {metric.reason}")
        print("-" * 50)
    
    # Demonstrate using a more flexible strictness
    print("\nTesting with 'flexible' strictness:")
    flexible_reward_fn = RewardFunction(
        func=function_call_reward,
        mode="local",
        expected_schema=weather_function_schema,
        strictness="flexible"
    )
    
    # Test the missing arg response with flexible strictness
    missing_arg_test = test_messages + [missing_arg_response]
    flexible_result = flexible_reward_fn(messages=missing_arg_test, original_messages=test_messages)
    
    print(f"Score with flexible strictness: {flexible_result.score}")
    print("Metrics:")
    for metric_name, metric in flexible_result.metrics.items():
        print(f"  {metric_name}: {metric.score}")
        print(f"    Reason: {metric.reason}")