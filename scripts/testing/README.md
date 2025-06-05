# Testing Scripts

This directory contains helper scripts for running integration tests, particularly for MCP agent functionality that requires Docker containers.

## Scripts

### `run_filesystem_rl_example_test.sh`

Runs the MCP Agent Filesystem RL example end-to-end test. This script:
- Starts the RewardKit Intermediary Server
- Runs the filesystem RL example evaluation using the reward-kit CLI
- Shuts down the server gracefully

**Usage:**
```bash
# Must be run from repository root
cd /path/to/reward-kit
./scripts/testing/run_filesystem_rl_example_test.sh
```

**Requirements:**
- Virtual environment set up at `./.venv`
- MCP agent config file `mcp_agent_config.yaml` in the repository root
- Docker available (for running MCP server containers)

### `run_mcp_test.sh`

Runs the MCP Intermediary Server and executes the RL filesystem scenario test client.

**Usage:**
```bash
# Must be run from repository root
cd /path/to/reward-kit
./scripts/testing/run_mcp_test.sh
```

**Requirements:**
- Virtual environment set up at `./.venv`
- MCP agent config file `mcp_agent_config.yaml` in the repository root
- Docker available (for running MCP server containers)

## Migration to pytest

These scripts have been largely replaced by proper pytest integration tests in `tests/mcp_agent/`. The pytest tests provide:

- Better integration with CI/CD pipelines
- Proper test isolation and cleanup
- Standardized test discovery and reporting
- Conditional skipping based on Docker availability

## Running Tests Through pytest

For better developer experience, use pytest instead of these scripts:

```bash
# Run all MCP agent tests (requires Docker)
pytest tests/mcp_agent/ -v

# Run specific integration tests
pytest tests/mcp_agent/test_mcp_integration.py -v

# Run Docker tests only
pytest -m docker -v

# Skip Docker tests
pytest -m "not docker" -v
```

## Legacy Script Usage

These scripts are maintained for backwards compatibility and debugging purposes. They may be useful for:

- Manual testing during development
- Debugging server startup issues
- Running tests outside of pytest framework
- CI/CD environments that prefer shell scripts over pytest

**Note:** These scripts should be run from the repository root directory.
