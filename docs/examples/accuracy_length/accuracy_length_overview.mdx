# Accuracy + Length Reward Examples

This directory contains examples demonstrating the use of combined accuracy and length-based reward functions.

## Overview

These examples show how to use the `cosine_scaled_accuracy_length_reward` function to evaluate model responses based on both:
1. Accuracy (correctness of the answer)
2. Length efficiency (brevity of the response)

This combined approach rewards responses that are both accurate and concise, penalizing verbosity in correct answers and providing a clear separation between correct and incorrect responses. 

**Note**: The accuracy detection depends on specific text-extraction mechanisms that may need customization for different types of content using the `extract_fn` and `compare_fn` parameters.

## Examples

### Cosine-Scaled Accuracy + Length Example

The [cosine_scaled_example.py](./cosine_scaled_example.py) script demonstrates the reward function's behavior with different types of responses:

- Short correct answers (highest score)
- Long correct answers (moderate score)
- Short incorrect answers (very low score)
- Long incorrect answers (low score, but still penalized for being wrong)

It also shows how to customize the weighting between accuracy and length components.

## Running the Examples

```bash
# Make sure you're in the reward-kit directory
cd /path/to/reward-kit

# Activate the virtual environment
source .venv/bin/activate

# Run the example
python examples/accuracy_length/cosine_scaled_example.py
```

## Expected Output

```
===== Evaluating with Default Parameters =====

Short Correct Answer:
Response (1 words): "Paris..."
Combined Score: 1.00
Accuracy Score: 1.00
Length Score: 1.00

Long Correct Answer:
Response (69 words): "The capital of France is Paris. Paris is located i..."
Combined Score: 0.88
Accuracy Score: 1.00
Length Score: 0.61

Short Incorrect Answer:
Response (1 words): "Lyon..."
Combined Score: 0.00
Accuracy Score: 0.00
Length Score: 0.00

Long Incorrect Answer:
Response (46 words): "I need to identify the capital city of France. Fra..."
Combined Score: 0.04
Accuracy Score: 0.00
Length Score: 0.13

===== Evaluating with Custom Parameters =====

Short Correct Answer (80% accuracy weight, 20% length weight):
Response (1 words): "Paris..."
Combined Score: 1.00
Accuracy Score: 1.00
Length Score: 1.00
```

## Custom Configurations

You can customize the reward function with various parameters:

```python
from reward_kit.rewards.accuracy_length import cosine_scaled_accuracy_length_reward

result = cosine_scaled_accuracy_length_reward(
    messages=messages,
    ground_truth="Expected answer",
    max_length=500,                # Maximum ideal length
    correctness_weight=0.7,        # Weight for accuracy component
    length_weight=0.3,             # Weight for length component
    min_value_correct=0.5,         # Minimum score for correct answers
    max_value_correct=1.0,         # Maximum score for correct answers
    min_value_wrong=0.0,           # Minimum score for wrong answers
    max_value_wrong=0.3,           # Maximum score for wrong answers
    token_method="whitespace"      # Method to count tokens
)
```

## Use Cases

This reward function is particularly useful for:

- Factual QA tasks where concise, correct answers are preferred
- Text summarization evaluation
- Mathematical problem-solving with step-by-step reasoning
- Any task where both accuracy and brevity are important

## Further Reading

For more information, see:
- [Combined Metrics Rewards Documentation](../../docs/examples/combined_metrics_rewards.md)
- [Reward Functions Overview](../../docs/examples/reward_functions_overview.md)