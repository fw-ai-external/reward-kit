# Core Data Types

This guide explains the primary data types used in the Reward Kit, including the input and output structures for reward functions.

## Overview

The Reward Kit uses several core data types to represent:
- Conversation messages
- Evaluation results
- Component metrics

Understanding these types is crucial for creating effective reward functions.

## Message Types

### The `Message` Class

```python
from reward_kit import Message

message = Message(
    role="assistant",
    content="This is the response content",
    name=None,  # Optional
    tool_call_id=None,  # Optional, for tool calling
    tool_calls=None,  # Optional, for tool calling
    function_call=None  # Optional, for function calling
)
```

The `Message` class represents a single message in a conversation and is compatible with the OpenAI message format.

### Message Dictionary Format

When working with reward functions, messages are often passed as dictionaries:

```python
message_dict = {
    "role": "assistant",
    "content": "This is the response content"
}
```

The minimum required fields are:
- `role`: The sender of the message (`"user"`, `"assistant"`, or `"system"`)
- `content`: The text content of the message

Additional fields for function/tool calling may include:
- `name`: Name of the sender (for named system messages)
- `tool_calls`: Tool call information
- `function_call`: Function call information (legacy format)

## Reward Output Types

### `RewardOutput` Class

```python
from reward_kit import RewardOutput, MetricRewardOutput

result = RewardOutput(
    score=0.75,  # Overall score between 0.0 and 1.0
    metrics={    # Component metrics dictionary
        "clarity": MetricRewardOutput(
            score=0.8,
            reason="The response is clear and concise"
        ),
        "accuracy": MetricRewardOutput(
            score=0.7,
            reason="Contains one minor factual error"
        )
    }
)
```

The `RewardOutput` class represents the complete result of a reward function evaluation, containing:
- An overall score (typically 0.0 to 1.0)
- A dictionary of component metrics

### `MetricRewardOutput` Class

```python
from reward_kit import MetricRewardOutput

metric = MetricRewardOutput(
    score=0.8,  # Score for this specific metric
    reason="Explanation for why this score was assigned"  # Optional
)
```

The `MetricRewardOutput` class represents a single component metric in the evaluation, containing:
- A score value (typically 0.0 to 1.0)
- An optional reason/explanation for the score

### Using Methods on RewardOutput

The `RewardOutput` class provides several useful methods:

```python
# Convert to dictionary
result_dict = result.to_dict()

# Create from dictionary
from_dict = RewardOutput.from_dict(result_dict)

# String representation (JSON)
result_json = str(result)
```

## Evaluation Types

### `EvaluateResult` Class

```python
from reward_kit import EvaluateResult, MetricResult

eval_result = EvaluateResult(
    score=0.75,  # Overall score
    reason="The response meets quality requirements",  # Optional
    metrics={
        "clarity": MetricResult(
            score=0.8,
            reason="Clear and well-structured"
        )
    }
)
```

The `EvaluateResult` class is used in the evaluation preview and creation APIs. It's similar to `RewardOutput` but follows a slightly different structure for compatibility with the Fireworks evaluation system.

### `MetricResult` Class

```python
from reward_kit import MetricResult

metric_result = MetricResult(
    score=0.8,
    reason="Explanation of the score",
    success=True  # Optional, indicates success status
)
```

The `MetricResult` class represents a single metric in an evaluation, similar to `MetricRewardOutput` but with an additional `success` field.

## Type Conversion

The Reward Kit handles conversion between these types:

```python
# Converting EvaluateResult to RewardOutput
evaluate_result = EvaluateResult(score=0.8, metrics={"quality": MetricResult(score=0.8, reason="Good quality")})
reward_output = convert_to_reward_output(evaluate_result)

# Converting RewardOutput to EvaluateResult
reward_output = RewardOutput(score=0.9, metrics={"clarity": MetricRewardOutput(score=0.9, reason="Very clear")})
evaluate_result = convert_to_evaluate_result(reward_output)
```

## Using Types in Reward Functions

Here's how to use these types properly in your reward functions:

```python
from reward_kit import reward_function, RewardOutput, MetricRewardOutput, Message
from typing import List, Optional, Dict, Any

@reward_function
def my_reward_function(
    messages: List[Dict[str, str]],
    original_messages: Optional[List[Dict[str, str]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> RewardOutput:
    """
    Example reward function with proper type annotations.
    """
    # Default values
    metadata = metadata or {}
    
    # Get the assistant's response
    response = messages[-1].get("content", "")
    
    # Evaluate the response
    clarity_score = evaluate_clarity(response)
    
    # Create metrics
    metrics = {
        "clarity": MetricRewardOutput(
            score=clarity_score,
            reason=f"Clarity score: {clarity_score:.2f}"
        )
    }
    
    return RewardOutput(score=clarity_score, metrics=metrics)
```

## Best Practices for Data Types

1. **Use Type Hints**: Always include proper type annotations in your functions
2. **Default Values**: Provide defaults for optional parameters
3. **Validation**: Validate input data before processing
4. **Error Handling**: Handle missing or malformed data gracefully
5. **Documentation**: Document the expected format for your inputs and outputs

## Next Steps

Now that you understand the core data types:

1. Learn about [Evaluation Workflows](evaluation_workflows.md) for testing and deploying your functions
2. Explore [Advanced Reward Functions](../examples/advanced_reward_functions.md) to see these types in action
3. Check the [API Reference](../api_reference/data_models.md) for complete details on all data types