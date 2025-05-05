# Reward Function Anatomy

This guide provides a detailed explanation of how reward functions are structured in the Reward Kit, focusing on the `@reward_function` decorator and the components that make up a complete reward function.

## The `@reward_function` Decorator

The `@reward_function` decorator is the core mechanism that transforms a regular Python function into a reward function that can be used for evaluation and deployment.

```python
from reward_kit import reward_function

@reward_function
def my_reward_function(messages, original_messages=None, **kwargs):
    # Your evaluation logic here
    return RewardOutput(...)
```

### What the Decorator Does

The `@reward_function` decorator performs several important functions:

1. **Input Validation**: Ensures the function receives the expected parameters
2. **Output Standardization**: Ensures the function returns a properly formatted `RewardOutput` object
3. **Deployment Capability**: Adds a `.deploy()` method to the function for easy deployment
4. **Backward Compatibility**: Handles legacy return formats (tuples of score and metrics)

### Under the Hood

Internally, the decorator wraps your function with logic that:

1. Processes the input parameters
2. Calls your function with the standardized inputs
3. Handles any exceptions that occur during execution
4. Formats the output as a `RewardOutput` object
5. Provides deployment capabilities through the `.deploy()` method

## Function Parameters

A standard reward function has these parameters:

```python
def reward_function(
    messages: List[Dict[str, str]],
    original_messages: Optional[List[Dict[str, str]]] = None,
    **kwargs
) -> RewardOutput:
    # ...
```

### Required Parameters

- **`messages`**: A list of message dictionaries in the conversation, where each message has at least `"role"` and `"content"` keys. The last message is typically the one being evaluated.

### Optional Parameters

- **`original_messages`**: The conversation context, usually messages before the response being evaluated. If not provided, it defaults to `messages[:-1]`.
- **`**kwargs`**: Additional parameters that can be used to customize the evaluation.

## Return Value

A reward function must return a `RewardOutput` object:

```python
return RewardOutput(
    score=0.75,  # Overall score between 0.0 and 1.0
    metrics={    # Component metrics
        "clarity": MetricRewardOutput(
            score=0.8,
            reason="The response clearly explains the concept"
        ),
        "accuracy": MetricRewardOutput(
            score=0.7,
            reason="Contains one minor factual error"
        )
    }
)
```

### RewardOutput Structure

- **`score`**: The final aggregate score (typically between 0.0 and 1.0)
- **`metrics`**: A dictionary of component metrics, each with its own score and explanation

## Multi-Component Reward Functions

Complex reward functions often evaluate multiple aspects of a response:

```python
@reward_function
def comprehensive_evaluation(messages, original_messages=None, **kwargs):
    response = messages[-1]["content"]
    metrics = {}
    
    # Evaluate clarity
    clarity_score = evaluate_clarity(response)
    metrics["clarity"] = MetricRewardOutput(
        score=clarity_score,
        reason=f"Clarity score: {clarity_score:.2f}"
    )
    
    # Evaluate accuracy
    accuracy_score = evaluate_accuracy(response)
    metrics["accuracy"] = MetricRewardOutput(
        score=accuracy_score,
        reason=f"Accuracy score: {accuracy_score:.2f}"
    )
    
    # Combine scores (weighted average)
    final_score = clarity_score * 0.4 + accuracy_score * 0.6
    
    return RewardOutput(score=final_score, metrics=metrics)
```

## Deployment Capabilities

The `@reward_function` decorator adds a `.deploy()` method to your function:

```python
# Deploy the function to Fireworks
evaluation_id = my_reward_function.deploy(
    name="my-evaluator",
    description="Evaluates responses based on custom criteria",
    force=True  # Overwrite if already exists
)
```

### Deploy Method Parameters

- **`name`**: ID for the deployed evaluator (required)
- **`description`**: Human-readable description (optional)
- **`force`**: Whether to overwrite an existing evaluator with the same name (optional)
- **`providers`**: List of model providers to use for evaluation (optional)

## Error Handling

Robust reward functions include proper error handling:

```python
@reward_function
def safe_evaluation(messages, original_messages=None, **kwargs):
    try:
        # Ensure we have a valid response to evaluate
        if not messages or messages[-1].get("role") != "assistant":
            return RewardOutput(
                score=0.0,
                metrics={"error": MetricRewardOutput(
                    score=0.0,
                    reason="No assistant response found"
                )}
            )
            
        # Your evaluation logic here
        # ...
        
    except Exception as e:
        # Handle any unexpected errors
        return RewardOutput(
            score=0.0,
            metrics={"error": MetricRewardOutput(
                score=0.0,
                reason=f"Evaluation error: {str(e)}"
            )}
        )
```

## Working with Metadata

You can pass additional configuration through the `**kwargs` parameter:

```python
@reward_function
def configurable_evaluation(messages, original_messages=None, metadata=None, **kwargs):
    """Reward function that supports configuration via metadata."""
    metadata = metadata or {}
    
    # Get configurable thresholds from metadata
    min_length = metadata.get("min_length", 50)
    max_score = metadata.get("max_score", 1.0)
    weight_factor = metadata.get("weight_factor", 1.0)
    
    # Use these parameters in your evaluation
    # ...
    
    # Apply any metadata-based adjustments to the final score
    final_score = base_score * weight_factor
    
    return RewardOutput(score=final_score, metrics=metrics)
```

When calling the function, you can pass this metadata:

```python
result = configurable_evaluation(
    messages=test_messages,
    metadata={"min_length": 100, "weight_factor": 1.2}
)
```

## Next Steps

Now that you understand the structure of reward functions:

1. Learn about the [Core Data Types](core_data_types.md) used in reward functions
2. Explore [Evaluation Workflows](evaluation_workflows.md) for testing and deployment
3. See [Code Examples](../examples/basic_reward_function.md) for practical implementations