# Reward Kit

Reward Kit is a library for defining, evaluating, and deploying reward functions for LLM fine-tuning. It provides tools to create custom reward functions and use them in reinforcement learning from human feedback (RLHF) workflows.

## Installation

```bash
pip install reward-kit
```

## Getting Started

The Reward Kit simplifies the creation and deployment of reward functions for evaluating AI model outputs.

### 1. Authentication Setup

To use Reward Kit with the Fireworks AI platform, set up your authentication credentials:

```bash
# Set your API key
export FIREWORKS_API_KEY=your_api_key
```

### 2. Creating a Simple Reward Function

Create a reward function to evaluate the quality of AI responses:

```python
from reward_kit import reward_function, RewardOutput, MetricRewardOutput

@reward_function
def informativeness(messages, original_messages=None, **kwargs):
    """Evaluate the informativeness of a response."""
    # Get the assistant's response
    response = messages[-1].get("content", "")
    
    # Simple evaluation: word count
    word_count = len(response.split())
    score = min(word_count / 100, 1.0)  # Cap at 1.0
    
    return RewardOutput(
        score=score,
        reason=f"Word count: {word_count}",
        metrics={
            "word_count": MetricRewardOutput(
                score=score,
                reason=f"Word count: {word_count}"
            )
        }
    )
```

### 3. Testing Your Reward Function

Test your reward function locally:

```python
# Test messages
test_messages = [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a method of data analysis that automates analytical model building."}
]

# Test your reward function
result = informativeness(messages=test_messages)
print(f"Score: {result.score}")
print(f"Reason: {result.reason}")
```

### 4. Evaluating with Sample Data

Create a JSONL file with sample conversations to evaluate:

```json
{"messages": [{"role": "user", "content": "Tell me about AI"}, {"role": "assistant", "content": "AI refers to systems designed to mimic human intelligence."}]}
{"messages": [{"role": "user", "content": "What is machine learning?"}, {"role": "assistant", "content": "Machine learning is a subset of AI that focuses on building systems that can learn from data."}]}
```

Preview your evaluation using the CLI:

```bash
reward-kit preview --metrics-folders "word_count=./path/to/metrics" --samples ./path/to/samples.jsonl
```

### 5. Deploying Your Reward Function

Deploy your reward function to use in training workflows:

```bash
reward-kit deploy --id my-evaluator --metrics-folders "word_count=./path/to/metrics" --force
```

Or deploy programmatically:

```python
from reward_kit.evaluation import create_evaluation

evaluator = create_evaluation(
    evaluator_id="my-evaluator",
    metric_folders=["word_count=./path/to/metrics"],
    display_name="My Word Count Evaluator",
    description="Evaluates responses based on word count",
    force=True  # Update if already exists
)
```

## Advanced Usage

### Multiple Metrics

Combine multiple metrics in a single reward function:

```python
@reward_function
def combined_reward(messages, original_messages=None, **kwargs):
    """Evaluate with multiple metrics."""
    response = messages[-1].get("content", "")
    
    # Word count metric
    word_count = len(response.split())
    word_score = min(word_count / 100, 1.0)
    
    # Specificity metric
    specificity_markers = ["specifically", "for example", "such as"]
    marker_count = sum(1 for marker in specificity_markers if marker.lower() in response.lower())
    specificity_score = min(marker_count / 2.0, 1.0)
    
    # Combined score with weighted components
    final_score = word_score * 0.3 + specificity_score * 0.7
    
    return RewardOutput(
        score=final_score,
        metrics={
            "word_count": MetricRewardOutput(score=word_score, reason=f"Word count: {word_count}"),
            "specificity": MetricRewardOutput(score=specificity_score, reason=f"Found {marker_count} specificity markers")
        }
    )
```

### Custom Model Providers

Deploy your reward function with a specific model provider:

```python
# Deploy with a custom provider
my_function.deploy(
    name="my-evaluator-anthropic",
    description="My evaluator using Claude model",
    providers=[
        {
            "providerType": "anthropic",
            "modelId": "claude-3-sonnet-20240229"
        }
    ],
    force=True
)
```

## Examples

Check the `examples` directory for complete examples:

- `evaluation_preview_example.py`: How to preview an evaluator
- `deploy_example.py`: How to deploy a reward function to Fireworks

## Command Line Interface

The Reward Kit includes a CLI for common operations:

```bash
# Show help
reward-kit --help

# Preview an evaluator
reward-kit preview --metrics-folders "metric=./path" --samples ./samples.jsonl

# Deploy an evaluator
reward-kit deploy --id my-evaluator --metrics-folders "metric=./path" --force
```

## License

Reward Kit is released under the MIT License.