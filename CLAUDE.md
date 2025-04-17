# CLAUDE.md - Reward Kit Development Guide

## Commands
- Install: `pip install -e .`
- Run tests: `pytest`
- Run single test: `pytest tests/test_file.py::test_function`
- Type check: `mypy reward_kit`
- Lint code: `flake8 reward_kit`
- Format code: `black reward_kit`

## Code Style Guidelines
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