# Reward Kit Developer Guide

This comprehensive guide is designed for developers who want to contribute to or modify the Reward Kit codebase. It covers all key aspects of development, from environment setup to testing and contributing new reward functions.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/fireworks-ai/reward-kit.git
cd reward-kit

# Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"  # Includes development dependencies

# Run tests
pytest

# Type check
mypy reward_kit

# Lint code
flake8 reward_kit
```

## Development Environment

### Setting Up Your Environment

1. **Clone the repository:**

```bash
git clone https://github.com/fireworks-ai/reward-kit.git
cd reward-kit
```

2. **Create and activate a virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install the package in development mode:**

```bash
pip install -e .            # Basic installation
pip install -e ".[dev]"     # With development dependencies
```

### Required Environment Variables

For development and testing, configure these environment variables:

```bash
# Development environment
export FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY
export FIREWORKS_API_BASE=https://dev.api.fireworks.ai
export FIREWORKS_ACCOUNT_ID=pyroworks-dev  # For specific operations

# For E2B code execution (optional)
export E2B_API_KEY=your_e2b_api_key
```

## Code Structure

```
reward-kit/
├── reward_kit/                 # Main package source code
│   ├── __init__.py             # Package initialization
│   ├── reward_function.py      # Core reward function decorator
│   ├── models.py               # Data models and types
│   ├── typed_interface.py      # Type interfaces for reward functions
│   ├── evaluation.py           # Evaluation pipeline
│   ├── auth.py                 # Authentication utilities
│   ├── cli.py                  # Command line interface
│   ├── rewards/                # Out-of-the-box reward functions
│   │   ├── __init__.py         # Reward functions registry
│   │   ├── code_execution.py   # Code execution rewards
│   │   ├── function_calling.py # Function calling rewards
│   │   ├── json_schema.py      # JSON schema validation
│   │   ├── math.py             # Math evaluation
│   │   ├── format.py           # Format validation 
│   │   ├── tag_count.py        # Tag counting
│   │   ├── accuracy.py         # Accuracy evaluation
│   │   ├── language_consistency.py # Language consistency
│   │   ├── reasoning_steps.py  # Reasoning steps evaluation
│   │   ├── length.py           # Response length evaluation
│   │   ├── repetition.py       # Repetition detection
│   │   ├── cpp_code.py         # C/C++ code evaluation
│   │   └── accuracy_length.py  # Combined accuracy and length evaluation
├── examples/                   # Example code and tutorials
│   ├── metrics/                # Example metric implementations
│   ├── samples/                # Sample data for evaluation
│   └── ...                     # Example scripts
├── tests/                      # Unit and integration tests
├── docs/                       # Documentation
└── ...                         # Project configuration files
```

## Reward Function Development

### Creating a New Reward Function

1. Create a new module in `reward_kit/rewards/` if needed
2. Implement your reward function using the `@reward_function` decorator
3. Update `reward_kit/rewards/__init__.py` to expose your function
4. Add unit tests in the `tests/` directory

Example structure:

```python
from typing import Dict, List, Any, Union, Optional

from ..typed_interface import reward_function
from ..models import Message, EvaluateResult, MetricResult

@reward_function
def my_reward_function(
    messages: Union[List[Dict[str, Any]], List[Message]],
    ground_truth: Optional[str] = None,
    **kwargs: Any
) -> EvaluateResult:
    """
    Evaluate responses based on custom criteria.
    
    Args:
        messages: List of conversation messages
        ground_truth: Expected correct answer
        **kwargs: Additional arguments
        
    Returns:
        EvaluateResult with evaluation score and metrics
    """
    # Your evaluation logic here
    # ...
    
    return EvaluateResult(
        score=score,
        reason=reason,
        metrics={
            "metric_name": MetricResult(
                score=metric_score,
                success=metric_success,
                reason=metric_reason
            )
        }
    )
```

### Coding Standards

Follow these coding standards for all contributions:

- **Imports**: Group standard library, third-party, and local imports separately
- **Types**: Use type hints for all function parameters and return values
- **Naming**: 
  - snake_case for functions, variables, methods
  - PascalCase for classes and dataclasses
  - UPPER_CASE for constants
- **Error handling**: Use specific exceptions with meaningful messages
- **Documentation**: Include docstrings for all public functions and classes
- **Format**: Maintain 88 character line length (Black default)
- **Function length**: Keep functions concise and focused on a single task
- **Testing**: Write unit tests for all public functions

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_evaluation.py

# Run specific test function
pytest tests/test_file.py::test_function

# Run with coverage report
pytest --cov=reward_kit
```

We only care about tests/ folder for now since there are a lot of other repos

### Writing Tests

Create test files in the `tests/` directory following this pattern:

```python
import unittest
# Import the function you're testing
from reward_kit.rewards.your_module import your_function

class TestYourFunction(unittest.TestCase):
    """Test your reward function."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        messages = [
            {"role": "user", "content": "Test question"},
            {"role": "assistant", "content": "Test response"}
        ]
        
        result = your_function(messages=messages)
        
        # Assert expectations
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("score", result)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)
```

### Code Quality Tools

```bash
# Type checking
mypy reward_kit

# Linting
flake8 reward_kit

# Format code
black reward_kit
```

## Available Reward Functions

Reward Kit includes these out-of-the-box reward functions:

| Category | Reward Functions |
|----------|-----------------|
| Format | `format_reward` |
| Tag Count | `tag_count_reward` |
| Accuracy | `accuracy_reward` |
| Language | `language_consistency_reward` |
| Reasoning | `reasoning_steps_reward` |
| Length | `length_reward`, `cosine_length_reward` |
| Repetition | `repetition_penalty_reward` |
| Code Execution | `binary_code_reward`, `fractional_code_reward` |
| C/C++ Code | `ioi_cpp_code_reward`, `binary_cpp_code_reward` |
| Combined | `cosine_scaled_accuracy_length_reward` |
| Function Calling | `schema_jaccard_reward`, `llm_judge_reward`, `composite_function_call_reward` |
| JSON Schema | `json_schema_reward` |
| Math | `math_reward` |

## Running Examples

The examples folder contains sample code for using the Reward Kit:

```bash
# Run evaluation preview example
source .venv/bin/activate && FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY \
FIREWORKS_API_BASE=https://dev.api.fireworks.ai \
python examples/evaluation_preview_example.py

# Run deployment example
source .venv/bin/activate && FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY \
FIREWORKS_API_BASE=https://dev.api.fireworks.ai \
python examples/deploy_example.py
```

## Command Line Interface

Use the Reward Kit CLI for common operations during development:

```bash
# Preview an evaluator
reward-kit preview --metrics-folders "word_count=./examples/metrics/word_count" \
--samples ./examples/samples/samples.jsonl

# Deploy an evaluator
reward-kit deploy --id my-test-evaluator \
--metrics-folders "word_count=./examples/metrics/word_count" --force
```

## Debugging Tips

### Authentication Issues

If you encounter authentication issues:

1. Check environment variables are correctly set
2. Verify API key has necessary permissions
3. Ensure account ID matches the API environment

Verify credentials:

```python
from reward_kit.auth import get_authentication
account_id, auth_token = get_authentication()
print(f"Using account: {account_id}")
print(f"Token (first 10 chars): {auth_token[:10]}...")
```

### API Debugging

For verbose API logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or use the `--verbose` flag with CLI commands:

```bash
reward-kit --verbose preview --metrics-folders "word_count=./examples/metrics/word_count" \
--samples ./examples/samples/samples.jsonl
```

## Building and Distribution

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build the package
python -m build

# Install locally from the built package
pip install dist/reward_kit-*.whl
```

## Contributing

1. Create a new branch for your feature/fix
2. Make your changes following coding standards
3. Add tests for new functionality
4. Update documentation to reflect changes
5. Run tests to ensure nothing breaks
6. Submit a pull request with a clear description

## Documentation

Update the documentation when adding new functionality:

1. Update relevant files in `docs/`
2. Add examples for new reward functions
3. Update the reward functions overview

## Troubleshooting

### Common Issues

1. **API Connection Errors**: Check your internet connection and API base URL
2. **Authentication Failures**: Verify your API key and account ID
3. **Import Errors**: Ensure you're using the correct virtual environment
4. **Deployment Failures**: Check API logs and your account permissions
5. **Type Errors**: Run `mypy reward_kit` to identify typing issues

For more help, consult the [official documentation](https://github.com/fireworks-ai/reward-kit) or file an issue on GitHub.