## Development and Publishing Guide

### Development Setup

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
   ```

4. Run tests:
   ```bash
   pytest
   ```

### Publishing to PyPI

We use a Makefile to simplify the publishing process:
                                                                                                                                                                                                                                                               "✳ GitHub Issues" 21:31 01-May-25
1. Update the version in `setup.py`

2. Run quality checks:
   ```bash
   make lint typecheck test
   ```

3. Build and publish:
   ```bash
   make release
   ```

Alternatively, you can run individual steps:
```bash
make clean     # Clean build artifacts
make build     # Build the package
make upload    # Upload to PyPI
```

For help on available commands:
```bash
make help
```