# Reward Kit Development Guide

This guide is intended for developers who want to contribute to or modify the Reward Kit codebase.

## Development Setup

### Environment Setup

1. Clone the repository:

```bash
git clone https://github.com/fireworks-ai/reward-kit.git
cd reward-kit
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install the package in development mode:

```bash
pip install -e .
pip install -e ".[dev]"  # Includes development dependencies
```

### Environment Variables

For development and testing, set these environment variables:

```bash
# Development environment
export FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY
export FIREWORKS_API_BASE=https://dev.api.fireworks.ai
export FIREWORKS_ACCOUNT_ID=pyroworks-dev  # Only needed for specific operations

# Production environment (if needed)
# export FIREWORKS_API_KEY=your_production_api_key
# export FIREWORKS_API_BASE=https://api.fireworks.ai
# export FIREWORKS_ACCOUNT_ID=your_account_id
```

## Running Examples

The examples folder contains sample code for using the Reward Kit. To run these examples, ensure you have the necessary environment variables set:

### Evaluation Preview Example

```bash
# Run with development environment
source .venv/bin/activate && FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY FIREWORKS_API_BASE=https://dev.api.fireworks.ai python examples/evaluation_preview_example.py
```

This example:
1. Loads a word count metric from `examples/metrics/word_count`
2. Uses sample conversations from `examples/samples/samples.jsonl`
3. Previews the evaluator using the Fireworks API
4. Creates a new evaluator called "word-count-eval"

### Deployment Example

```bash
# Run with development environment
source .venv/bin/activate && FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY FIREWORKS_API_BASE=https://dev.api.fireworks.ai python examples/deploy_example.py
```

This example:
1. Defines an "informativeness" reward function
2. Tests it locally
3. Deploys it to the Fireworks platform

## Using the CLI in Development

The Reward Kit CLI can be used for common operations during development:

```bash
# Preview an evaluator
source .venv/bin/activate && FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY FIREWORKS_API_BASE=https://dev.api.fireworks.ai reward-kit preview --metrics-folders "word_count=./examples/metrics/word_count" --samples ./examples/samples/samples.jsonl

# Deploy an evaluator
source .venv/bin/activate && FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY FIREWORKS_API_BASE=https://dev.api.fireworks.ai reward-kit deploy --id my-test-evaluator --metrics-folders "word_count=./examples/metrics/word_count" --force
```

## Project Structure

```
reward-kit/
├── reward_kit/             # Core library code
│   ├── __init__.py         # Package initialization
│   ├── auth.py             # Authentication utilities
│   ├── cli.py              # Command line interface
│   ├── evaluation.py       # Evaluation functionality
│   └── ...
├── examples/               # Example code and tutorials
│   ├── metrics/            # Example metric implementations
│   ├── samples/            # Sample data for evaluation
│   └── ...
├── tests/                  # Unit and integration tests
└── ...
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_evaluation.py

# Run with coverage report
pytest --cov=reward_kit
```

## Debugging Tips

### Authentication Issues

If you encounter authentication issues, check:

1. Environment variables are correctly set
2. API key has the necessary permissions
3. Account ID matches the API environment

Try the `get_authentication()` function to verify your credentials:

```python
from reward_kit.auth import get_authentication
account_id, auth_token = get_authentication()
print(f"Using account: {account_id}")
print(f"Token (first 10 chars): {auth_token[:10]}...")
```

### API Debugging

For verbose API logging, set the logging level to DEBUG:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or use the `--verbose` flag with CLI commands:

```bash
reward-kit --verbose preview --metrics-folders "word_count=./examples/metrics/word_count" --samples ./examples/samples/samples.jsonl
```

## Building and Distribution

To build the package for distribution:

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
2. Make your changes
3. Run tests to ensure nothing breaks
4. Submit a pull request

Please follow the existing code style and add tests for new functionality.

## Troubleshooting

### Common Issues

1. **API Connection Errors**: Check your internet connection and API base URL
2. **Authentication Failures**: Verify your API key and account ID
3. **Import Errors**: Ensure you're using the correct virtual environment
4. **Deployment Failures**: Check API logs and your account permissions

For more help, consult the [official documentation](https://github.com/fireworks-ai/reward-kit).