# Reward Kit Developer Guide

This comprehensive guide is designed for developers who want to contribute to or modify the Reward Kit codebase. It covers all key aspects of development, from environment setup to testing and contributing new reward functions.

We are committed to fostering an open and welcoming environment. All contributors are expected to adhere to our [Code of Conduct](../../CODE_OF_CONDUCT.md).

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

### Coding Style and Standards

To maintain code quality and consistency, please adhere to the following standards:

- **Formatting**:
    - Use `black` for code formatting. The maximum line length is 88 characters.
    - Use `isort` for organizing imports.
- **Linting**:
    - Adhere to `flake8` guidelines.
- **Type Hinting**:
    - Use type hints for all function parameters, return values, and variables where appropriate.
    - Run `mypy reward_kit` to check for type errors.
- **Naming Conventions**:
    - `snake_case` for functions, methods, and variables.
    - `PascalCase` for classes and dataclasses.
    - `UPPER_SNAKE_CASE` for constants.
- **Imports**:
    - Group imports in the following order:
        1. Standard library imports
        2. Third-party library imports
        3. Local application/library specific imports (e.g., `from ..module import something`)
    - Separate each group with a blank line.
- **Docstrings**:
    - Write clear and concise docstrings for all public modules, classes, functions, and methods. Follow PEP 257 conventions.
    - For functions, explain the arguments, what the function does, and what it returns.
- **Error Handling**:
    - Use specific exception types rather than generic `Exception`.
    - Provide meaningful error messages.
- **Function Design**:
    - Keep functions and methods short and focused on a single responsibility.
    - Aim for readability and maintainability.
- **Testing**:
    - Write unit tests for all new public functions and significant private logic.
    - Ensure tests cover a variety of cases, including edge cases and expected failures.

### Pre-commit Hooks

To help enforce coding standards and catch issues early, we use pre-commit hooks. These hooks run automatically before each commit to check your code for issues like formatting, linting errors, and type errors.

**Installation and Setup:**

1.  **Install pre-commit**:
    If you installed development dependencies with `pip install -e ".[dev]"`, `pre-commit` should already be installed. If not, you can install it via pip:
    ```bash
    pip install pre-commit
    ```

2.  **Install the git hooks**:
    Navigate to the root of the repository and run:
    ```bash
    pre-commit install
    ```
    This will set up the pre-commit hooks to run automatically when you `git commit`.

**Usage:**

*   Once installed, pre-commit hooks will run on any changed files before you commit. If any hook fails, the commit will be aborted. You'll need to fix the reported issues and then `git add` the files again before attempting to commit.
*   Some hooks (like `black` and `isort`) may automatically fix issues. If they do, you'll still need to `git add` the modified files.
*   You can also run the hooks manually on all files at any time:
    ```bash
    pre-commit run --all-files
    ```
    This is useful for checking the entire codebase or after pulling new changes.

By using pre-commit hooks, we can ensure a consistent code style and catch many common errors before they even reach the CI pipeline, saving time and effort.

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

## Contributing Process

We welcome contributions to Reward Kit! Please follow these steps to contribute:

1.  **Find or Create an Issue**:
    *   Look for existing issues on the [GitHub Issues page](https://github.com/fireworks-ai/reward-kit/issues) that you'd like to work on.
    *   If you have a new feature or bug fix, please create a new issue first to discuss it with the maintainers, unless it's a very minor change.

2.  **Fork and Clone the Repository**:
    *   Fork the repository to your own GitHub account.
    *   Clone your fork locally: `git clone https://github.com/YOUR_USERNAME/reward-kit.git`
    *   Add the upstream repository: `git remote add upstream https://github.com/fireworks-ai/reward-kit.git`

3.  **Create a New Branch**:
    *   Create a descriptive branch name for your feature or fix (e.g., `feat/add-new-reward-metric` or `fix/resolve-auth-bug`).
    *   `git checkout -b your-branch-name`

4.  **Make Your Changes**:
    *   Implement your changes, adhering to the [Coding Style and Standards](#coding-style-and-standards).
    *   Ensure your code is well-documented with docstrings.

5.  **Add Tests**:
    *   Write new tests for any new functionality.
    *   Ensure all tests pass by running `pytest`.

6.  **Run Code Quality Checks**:
    *   Format your code: `black reward_kit tests`
    *   Check linting: `flake8 reward_kit tests`
    *   Check types: `mypy reward_kit`
    *   (Once configured) Run pre-commit hooks: `pre-commit run --all-files`

7.  **Update Documentation**:
    *   If your changes affect user-facing features or APIs, update the relevant documentation in the `docs/` directory.
    *   Add examples if applicable.

8.  **Commit Your Changes**:
    *   Write clear and concise commit messages. Reference the issue number if applicable (e.g., `feat: Add awesome new metric (closes #123)`).
    *   `git commit -m "Your descriptive commit message"`

9.  **Push to Your Fork**:
    *   `git push origin your-branch-name`

10. **Submit a Pull Request (PR)**:
    *   Open a pull request from your branch to the `main` branch of the `fireworks-ai/reward-kit` repository.
    *   Provide a clear title and a detailed description of your changes in the PR.
        *   Explain the "what" and "why" of your contribution.
        *   Link to the relevant issue(s) using keywords like `Closes #123` or `Fixes #456`.
    *   Ensure all CI checks pass on your PR.
    *   Be responsive to any feedback or questions from the maintainers during the code review process.

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
