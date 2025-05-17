# CLAUDE.md - Reward Kit Development Guide

## Commands
- activate venv: `source .venv/bin/activate`
- Install: `pip install -e .`
- Run tests: `pytest`
- Run single test: `pytest tests/test_file.py::test_function`
- Type check: `mypy reward_kit`
- Lint code: `flake8 reward_kit`
- Format code: `black reward_kit`
- Format code (pre-commit): `pre-commit run --all-files`

## Pre-commit Configuration Notes

The pre-commit hooks in this repository have been configured with relaxed settings to prioritize development
velocity over strict style enforcement. The main changes are:

1. **Line Length:** Increased to 2000 characters (from default 88/79)
2. **Code Complexity:** Increased to 100 (from default 10/12)
3. **Disabled Error Codes:**
   - Import-related: E402 (imports not at top), F401 (unused imports)
   - Style-related: W503 (line break before operator), E203 (whitespace before colon), E731 (lambda assignment)
   - Unused variables: F841
   - Redundant imports: F811
   - Missing whitespace: E226
   - Single-line functions: E704
   - Comparison style: E713, E712
   - Comma spacing: E231
   - F-string warnings: F541

### MyPy Configuration

The mypy type checker has also been relaxed by disabling the following error codes:
- import-not-found
- truthy-function
- no-redef
- assignment
- union-attr
- misc
- name-defined
- index
- arg-type
- operator
- var-annotated
- return-value
- call-arg
- attr-defined

### Future Considerations

In the future, it may be worth revisiting these settings to improve code quality:

1. Enable unused import warnings (F401)
2. Enable unused variable warnings (F841)
3. Decrease maximum line length to a more reasonable value (120-150)
4. Decrease maximum complexity to a more reasonable value (20-30)

For now, these relaxed settings help avoid pre-commit failures while the codebase is under active development.

### Type Checking Notes
When working with mypy, be aware of these common issues:

1. **Module import issues**:
   - For missing imports, we've configured mypy.ini to ignore these errors
   - For new external dependencies, make sure to add proper type stubs when available

2. **Docker, OpenAI and other third-party libraries**:
   - We've added proper fallback classes with type annotations when these libraries aren't available
   - Use TYPE_CHECKING for imports only needed by type checkers

3. **String vs dict type issues**:
   - Several functions may return string or dict depending on the context
   - Use type guards with isinstance() checks when needed

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
