# Frozen Lake MCP Complete Example

A comprehensive, well-organized example demonstrating Model Context Protocol (MCP) integration with the FrozenLake environment. This example showcases both local development and remote deployment patterns for MCP-based agent evaluation.

## 🎯 Overview

This example demonstrates how to:
- Serve a Gymnasium FrozenLake environment via MCP
- Test locally with various MCP clients and rollout patterns
- Deploy and test remotely using Google Cloud Run
- Implement robust seed testing and evaluation pipelines
- Use the reward-kit north star API (`rk.make()` and `rk.rollout()`)

## 📁 Project Structure

```
frozen_lake_mcp_complete/
├── README.md                   # This comprehensive guide
├── mcp_server/                 # MCP Server Implementation
│   ├── frozen_lake_mcp_server.py    # 🏭 Production server (single-session, NOT for concurrent rollouts)
│   ├── frozen_lake_adapter.py       # Environment adapter
│   └── simulation_server.py         # 🚀 Simulation server (multi-session, BEST for concurrent rollouts)
├── local_testing/              # Local Development & Testing
│   ├── mcp_rollout_client.py        # MCP rollout client
│   ├── rollout_client.py            # Alternative rollout client
│   ├── test_north_star.py           # North star API tests
│   ├── test_adapter_seeding.py      # Adapter seeding tests
│   ├── test_seed_fix.py             # Seed verification tests
│   ├── test_seed_verification.py    # Additional seed tests
│   └── run_all_robustness_tests.py  # Comprehensive test suite
├── remote_testing/             # Remote Deployment Testing
│   └── test_remote_north_star.py    # Remote north star tests
├── shared_data/                # Shared Data & Configurations
│   ├── rollouts.jsonl               # Environment configurations
│   ├── clean_trajectories.jsonl     # Sample trajectory data
│   └── analyze_seed_test.py         # Trajectory analysis utilities
└── docs/                       # Documentation
    ├── mcp_server_readme.md         # MCP server documentation
    └── remote_testing_readme.md     # Remote testing guide
```

## 🏭 Server Types Explained

### `frozen_lake_mcp_server.py` - Production Server
- **Purpose**: Single-session production deployment
- **Use Case**: Individual client connections, demos
- **Concurrency**: ❌ NOT suitable for multiple concurrent rollouts
- **Session Management**: Global state (one game per server instance)

### `simulation_server.py` - Simulation Server
- **Purpose**: Multi-session simulation environment
- **Use Case**: ✅ **PREFERRED for concurrent rollouts and testing**
- **Concurrency**: ✅ Handles multiple parallel sessions properly
- **Session Management**: Per-client isolated sessions with proper seeding

**For Remote Deployment**: Use `simulation_server.py` for best rollout performance!

## 🚀 Quick Start

### Prerequisites

Follow the [development setup guide](../../development/CONTRIBUTING.md) to set up your environment:

```bash
# Activate virtual environment
source .venv/bin/activate

# Ensure dependencies are installed
.venv/bin/pip install -e ".[dev]"

# Set up authentication (see CONTRIBUTING.md)
export FIREWORKS_API_KEY="your_dev_fireworks_api_key"
export FIREWORKS_ACCOUNT_ID="your_account_id"
```

### 1. Local Testing

**First, start the MCP server (in one terminal):**
```bash
cd examples/frozen_lake_mcp_complete/mcp_server
../../../.venv/bin/python simulation_server.py
# Server will start on http://localhost:8000/mcp
# Keep this terminal running for tests
```

**Then run local tests (in another terminal):**
```bash
cd examples/frozen_lake_mcp_complete/local_testing

# Test basic north star API (requires server running!)
../../../.venv/bin/python test_north_star.py

# Test with MCP rollout client (requires server running!)
../../../.venv/bin/python mcp_rollout_client.py --test single --seed 42

# Run comprehensive robustness tests
../../../.venv/bin/python run_all_robustness_tests.py

# Test adapter functionality (doesn't require server)
../../../.venv/bin/python test_adapter_seeding.py
```

### 2. Remote Testing

**⚠️ Note**: Remote testing requires a deployed MCP server. See `REMOTE_DEPLOYMENT_HANDOFF.md` for deployment instructions.

**Test against deployed remote server:**
```bash
cd examples/frozen_lake_mcp_complete/remote_testing
../../../.venv/bin/python test_remote_north_star.py
```

## 🔧 Key Components

### MCP Server (`mcp_server/`)

- **`frozen_lake_mcp_server.py`**: Production-ready FastMCP server using stateless HTTP pattern
- **`frozen_lake_adapter.py`**: Bridges FrozenLakeEnv with reward-kit's MCP framework
- **`simulation_server.py`**: Stateful server managing multiple seeded environment instances

### Local Testing (`local_testing/`)

- **`test_north_star.py`**: Validates the core `rk.make()` and `rk.rollout()` API
- **`mcp_rollout_client.py`**: Demonstrates proper MCP Python SDK connection patterns
- **Seed Testing**: Multiple files ensuring deterministic behavior across different seeds
- **`run_all_robustness_tests.py`**: Comprehensive test suite for development validation

### Remote Testing (`remote_testing/`)

- **`test_remote_north_star.py`**: Tests against deployed Cloud Run instance
- Validates production deployment without local dependencies

### Shared Data (`shared_data/`)

- **`rollouts.jsonl`**: Environment configurations with different seeds and prompts (used by tests)
- **`clean_trajectories.jsonl`**: Sample trajectory data from previous runs
- **`analyze_seed_test.py`**: Trajectory analysis utility (requires specific trajectory format)

## 🎮 Game Environment

**FrozenLake 4x4 Grid:**
```
[S] F  F  F
 F  H  F  H
 F  F  F  H
 H  F  F  [G]
```

- **S**: Start position
- **F**: Frozen (safe) tiles
- **H**: Holes (deadly)
- **G**: Goal
- **Actions**: `LEFT`, `RIGHT`, `UP`, `DOWN`

## 🧪 Testing Workflow

### Development Testing
```bash
# 1. Start local server (in background or separate terminal)
cd mcp_server && ../../../.venv/bin/python frozen_lake_mcp_server.py &

# 2. Wait for server to start, then run validation
sleep 2 && cd ../local_testing && ../../../.venv/bin/python test_north_star.py

# 3. Run comprehensive tests (requires server)
../../../.venv/bin/python run_all_robustness_tests.py

# 4. Test specific seeds (requires server)
../../../.venv/bin/python test_seed_verification.py

# 5. Test adapter (no server needed)
../../../.venv/bin/python test_adapter_seeding.py
```

### Production Validation
```bash
# Test remote deployment
cd remote_testing && ../../../.venv/bin/python test_remote_north_star.py
```

## 📊 Expected Results

### Local Testing Output
```
🎯 Testing Local North Star Interface
✅ Policy created successfully
✅ Local MCP environments created successfully
✅ Generated 3 trajectories from LOCAL server
🏆 Local north star interface test completed successfully!
```

### Remote Testing Output
```
🌐 REMOTE ROLLOUT TEST SUITE
✅ Remote MCP session initialized successfully
✅ Found 1 tools on remote server: lake_move
✅ Remote north star interface test completed successfully!
```

## 🔍 Troubleshooting

### Common Issues

1. **MCP Connection Errors**:
   - Ensure server is running on correct port
   - Check for port conflicts
   - Verify MCP endpoint URL format

2. **Seed Inconsistencies**:
   - Run seed verification tests
   - Check adapter seeding implementation
   - Validate environment state management

3. **Remote Testing Failures**:
   - Verify Cloud Run deployment status
   - Check remote server logs
   - Ensure correct remote URL

### Debugging Tools

```bash
# Check server logs (start server in verbose mode)
cd mcp_server && ../../../.venv/bin/python frozen_lake_mcp_server.py --verbose

# Analyze trajectories (needs generated trajectory data)
cd shared_data && ../../../.venv/bin/python analyze_seed_test.py
# Note: Requires trajectory data in expected format

# Run single rollout for debugging (requires server running)
cd local_testing && ../../../.venv/bin/python mcp_rollout_client.py --test single --seed 42

# Test adapter without server
cd local_testing && ../../../.venv/bin/python test_adapter_seeding.py
```

## 🌟 North Star API

This example validates the reward-kit north star interface:

```python
import reward_kit as rk

# Create policy
policy = rk.FireworksPolicy(
    model_id="accounts/fireworks/models/qwen3-235b-a22b",
    temperature=0.2
)

# Local testing
envs = rk.make("http://localhost:8000/mcp", dataset=dataset)
trajectories = await rk.rollout(envs, policy=policy, steps=8)

# Remote testing
envs = rk.make("https://remote-server.com/mcp", dataset=dataset)
trajectories = await rk.rollout(envs, policy=policy, steps=8)
```

## 📚 Additional Resources

- **[MCP Server Documentation](docs/mcp_server_readme.md)**: Detailed server implementation guide
- **[Remote Testing Guide](docs/remote_testing_readme.md)**: Remote deployment and testing
- **[CONTRIBUTING.md](../../development/CONTRIBUTING.md)**: Development setup and standards

## 🤝 Contributing

When modifying this example:

1. **Follow [CONTRIBUTING.md](../../development/CONTRIBUTING.md)** standards
2. **Test locally first** using the local testing suite
3. **Validate remote deployment** with remote testing
4. **Update documentation** for any structural changes
5. **Run comprehensive tests** before submitting changes

```bash
# Code quality checks
.venv/bin/black examples/frozen_lake_mcp_complete
.venv/bin/flake8 examples/frozen_lake_mcp_complete
.venv/bin/mypy examples/frozen_lake_mcp_complete
```

This example serves as a reference implementation for MCP-based environments in reward-kit and demonstrates best practices for local development, testing, and remote deployment.
