# Reward Kit

**Reward-Kit: Author, reproduce, and evaluate reward functions seamlessly on Fireworks, TRL, and your own infrastructure.**

## Key Features

*   **Easy-to-use Decorator**: Define reward functions with a simple `@reward_function` decorator.
*   **Local Testing**: Quickly test your reward functions with sample data.
*   **Flexible Evaluation**: Evaluate model outputs based on single or multiple custom metrics.
*   **Seamless Deployment**: Deploy your reward functions to platforms like Fireworks AI.
*   **CLI Support**: Manage and interact with your reward functions via a command-line interface.
*   **Simplified Dataset Integration**: Direct integration with HuggingFace datasets and on-the-fly format conversion.
*   **Extensible**: Designed to be adaptable for various LLM evaluation scenarios.

## Installation

```bash
pip install reward-kit
```

## Getting Started

The Reward Kit simplifies the creation and deployment of reward functions for evaluating AI model outputs.

### 1. Authentication Setup

To interact with the Fireworks AI platform for deploying and managing evaluations, Reward Kit needs your Fireworks AI credentials. You can configure these in two ways:

**A. Environment Variables (Highest Priority)**

Set the following environment variables:

*   `FIREWORKS_API_KEY`: Your Fireworks AI API key. This is required for all interactions with the Fireworks API.
*   `FIREWORKS_ACCOUNT_ID`: Your Fireworks AI Account ID. This is often required for operations like creating or listing evaluators under your account.

```bash
export FIREWORKS_API_KEY="your_fireworks_api_key"
export FIREWORKS_ACCOUNT_ID="your_fireworks_account_id"
```

**B. Configuration File (Lower Priority)**

Alternatively, you can store your credentials in a configuration file located at `~/.fireworks/auth.ini`. If environment variables are not set, Reward Kit will look for this file.

Create the file with the following format:

```ini
[fireworks]
api_key = YOUR_FIREWORKS_API_KEY
account_id = YOUR_FIREWORKS_ACCOUNT_ID
```

Replace `YOUR_FIREWORKS_API_KEY` and `YOUR_FIREWORKS_ACCOUNT_ID` with your actual credentials.

**Credential Sourcing Order:**

Reward Kit will prioritize credentials in the following order:
1.  Environment Variables (`FIREWORKS_API_KEY`, `FIREWORKS_ACCOUNT_ID`)
2.  `~/.fireworks/auth.ini` configuration file

Ensure that the `auth.ini` file has appropriate permissions to protect your sensitive credentials.

The `FIREWORKS_API_KEY` is essential for authenticating your requests to the Fireworks AI service. The `FIREWORKS_ACCOUNT_ID` is used to identify your specific account context for operations that are account-specific, such as managing your evaluators. While the API key authenticates *who* you are, the account ID often specifies *where* (under which account) an operation should take place. Some Fireworks API endpoints may require both.

### 2. Creating a Simple Reward Function

Create a reward function to evaluate the quality of AI responses:

```python
from reward_kit import reward_function
from reward_kit.models import EvaluateResult, MetricResult, Message # Assuming models are here
from typing import List, Dict, Any, Optional

@reward_function
def informativeness(
    messages: List[Dict[str, Any]], # Or List[Message] if using Message type directly
    original_messages: Optional[List[Dict[str, Any]]] = None, # Or List[Message]
    **kwargs: Any
) -> EvaluateResult:
    """Evaluate the informativeness of a response."""
    # Get the assistant's response
    response = messages[-1].get("content", "")

    # Simple evaluation: word count
    word_count = len(response.split())
    # Score normalized to 0-1, assuming 100 words is a good target for this example
    score = min(word_count / 100.0, 1.0)
    is_informative_enough = word_count > 10 # Example success condition

    return EvaluateResult(
        score=score,
        reason=f"Word count: {word_count}",
        metrics={
            "word_count": MetricResult(
                score=score,
                success=is_informative_enough,
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
from reward_kit import reward_function
from reward_kit.models import EvaluateResult, MetricResult, Message # Assuming models are here
from typing import List, Dict, Any, Optional

@reward_function
def combined_reward(
    messages: List[Dict[str, Any]], # Or List[Message]
    original_messages: Optional[List[Dict[str, Any]]] = None, # Or List[Message]
    **kwargs: Any
) -> EvaluateResult:
    """Evaluate with multiple metrics."""
    response = messages[-1].get("content", "")

    # Word count metric
    word_count = len(response.split())
    word_score = min(word_count / 100.0, 1.0)
    word_metric_success = word_count > 10

    # Specificity metric
    specificity_markers = ["specifically", "for example", "such as"]
    marker_count = sum(1 for marker in specificity_markers if marker.lower() in response.lower())
    specificity_score = min(marker_count / 2.0, 1.0)
    specificity_metric_success = marker_count > 0

    # Combined score with weighted components
    final_score = word_score * 0.3 + specificity_score * 0.7

    return EvaluateResult(
        score=final_score,
        reason=f"Combined score based on word count ({word_count}) and specificity markers ({marker_count})",
        metrics={
            "word_count": MetricResult(
                score=word_score,
                success=word_metric_success,
                reason=f"Word count: {word_count}"
            ),
            "specificity": MetricResult(
                score=specificity_score,
                success=specificity_metric_success,
                reason=f"Found {marker_count} specificity markers"
            )
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

## Dataset Integration

Reward Kit provides seamless integration with popular datasets through a simplified configuration system:

### Direct HuggingFace Integration

Load datasets directly from HuggingFace Hub without manual preprocessing:

```bash
# Evaluate using GSM8K dataset with math-specific prompts
reward-kit run --config-name run_math_eval.yaml --config-path examples/math_example/conf
```

### Derived Datasets

Create specialized dataset configurations that reference base datasets and apply transformations:

```yaml
# conf/dataset/gsm8k_math_prompts.yaml
defaults:
  - base_derived_dataset
  - _self_

base_dataset: "gsm8k"
system_prompt: "Solve the following math problem. Show your work clearly. Put the final numerical answer between <answer> and </answer> tags."
output_format: "evaluation_format"
derived_max_samples: 5
```

### Key Benefits

- **No Manual Conversion**: Datasets are converted to evaluation format on-the-fly
- **System Prompt Integration**: Prompts are part of dataset configuration, not evaluation logic
- **Flexible Column Mapping**: Automatically adapts different dataset formats
- **Reusable Configurations**: Base datasets can be extended for different use cases

See the [math example](examples/math_example/) for a complete demonstration of the dataset system.

## Detailed Documentation

For more comprehensive information, including API references, tutorials, and advanced guides, please see our [full documentation](docs/documentation_home.mdx).

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

## Community and Support

*   **GitHub Issues**: For bug reports and feature requests, please use [GitHub Issues](https://github.com/fireworks-ai/reward-kit/issues).
*   **GitHub Discussions**: (If enabled) For general questions, ideas, and discussions.
*   Please also review our [Contributing Guidelines](development/CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

## Development

### Type Checking

The codebase uses mypy for static type checking. To run type checking:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run mypy
mypy reward_kit
```

Our CI pipeline enforces type checking, so please ensure your code passes mypy checks before submitting PRs.

### Running Tests

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Code of Conduct

We are dedicated to providing a welcoming and inclusive experience for everyone. Please review and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Reward Kit is released under the Apache License 2.0.
