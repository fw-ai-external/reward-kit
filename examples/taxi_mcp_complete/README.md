# Taxi MCP Complete Example

A comprehensive Model Context Protocol (MCP) implementation for the **Taxi-v3** gymnasium environment. This example demonstrates how to create a fully functional MCP server for reinforcement learning environments using the reward-kit framework, including local development, testing, and deployment patterns.

## 🎯 What is the Taxi Problem?

The **Taxi** environment is a classic reinforcement learning problem where:
- A taxi must navigate a 5x5 grid world with walls and designated locations
- Pick up a passenger from one of 4 locations (Red, Green, Yellow, Blue)  
- Drop off the passenger at their destination
- Avoid illegal pickup/dropoff attempts that result in penalties

**Goal**: Successfully complete passenger trips while minimizing steps and avoiding penalties.

## 🏗️ Project Structure

```
taxi_mcp_complete/
├── README.md                           # This comprehensive guide
├── mcp_server/                         # MCP Server Implementation
│   ├── taxi_adapter.py                 # Taxi environment adapter
│   ├── simulation_server.py            # 🚀 Multi-session simulation server (PREFERRED)
│   ├── run_simulation_server.py        # Server startup script
│   ├── test_server.py                  # Basic server test
│   └── requirements.txt                # Server dependencies
├── local_testing/                      # Local Development & Testing
│   ├── test_north_star.py              # North star API tests
│   ├── test_taxi_integration.py        # Adapter integration tests
│   ├── recording_trajectories.jsonl    # Recorded trajectory data
│   └── clean_openai_format.jsonl       # Clean format for training
└── shared_data/                        # Shared Data & Configurations
    ├── taxi_rollouts.jsonl             # Environment configurations and prompts
    └── taxi_rollouts copy.jsonl        # Backup configurations
```

## 🎮 Game Environment

**Taxi 5x5 Grid World:**
```
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+
```

- **R/G/Y/B**: Pickup/dropoff locations (Red, Green, Yellow, Blue)
- **|, +, -**: Walls that block movement
- **:** Empty navigable spaces
- **t**: Empty taxi (needs to pick up passenger)
- **T**: Taxi with passenger (ready for dropoff)
- **r/g/y/b**: Current destination (lowercase)

### State Space & Actions

- **500 discrete states** encoding taxi position, passenger location, and destination
- **6 actions**: SOUTH (0), NORTH (1), EAST (2), WEST (3), PICKUP (4), DROPOFF (5)

### Rewards
- **-1**: Each step (time penalty)
- **+20**: Successful dropoff
- **-10**: Illegal pickup/dropoff attempt

## 🚀 Quick Start

### Prerequisites

Ensure you have the reward-kit development environment set up:

```bash
# Activate virtual environment
source .venv/bin/activate

# Ensure dependencies are installed
.venv/bin/pip install -e ".[dev]"

# Set up authentication
export FIREWORKS_API_KEY="your_dev_fireworks_api_key"
export FIREWORKS_ACCOUNT_ID="your_account_id"
```

### Option 1: North Star API Testing (Recommended)

Test the complete reward-kit north star interface with record-and-playback:

```bash
# Terminal 1: Start the MCP server
cd examples/taxi_mcp_complete/mcp_server
../../../.venv/bin/python simulation_server.py
# Server starts on http://localhost:8000/mcp/

# Terminal 2: Run north star tests
cd examples/taxi_mcp_complete/local_testing
../../../.venv/bin/python test_north_star.py
```

### Option 2: Direct Adapter Testing

Test the taxi adapter directly without running a server:

```bash
cd examples/taxi_mcp_complete/local_testing
../../../.venv/bin/python test_taxi_integration.py
```

### Option 3: Manual Server Testing

Start the server manually and interact with it:

```bash
cd examples/taxi_mcp_complete/mcp_server
../../../.venv/bin/python simulation_server.py --port 8000 --host 0.0.0.0
```

## 🧪 Testing Infrastructure

### North Star API Tests (`test_north_star.py`)

Validates the complete reward-kit north star interface:

```python
import reward_kit as rk

# Load dataset with taxi scenarios
dataset = load_jsonl("../shared_data/taxi_rollouts.jsonl") 

# Create policy with record-and-playback
policy = rk.FireworksPolicy(
    model_id="accounts/fireworks/models/qwen3-235b-a22b",
    temperature=0.2
)

# Create environments and run rollouts
envs = rk.make("http://localhost:8000/mcp/", dataset=dataset)
trajectories = await rk.rollout(envs, policy=policy, steps=25)
```

**Features:**
- ✅ Automatic record-and-playback for fast iteration
- ✅ Multiple taxi scenarios with different seeds
- ✅ Clean OpenAI format output for training
- ✅ Performance metrics and success tracking

### Integration Tests (`test_taxi_integration.py`)

Comprehensive adapter testing:

- ✅ Environment creation with different configurations
- ✅ State decoding and description generation  
- ✅ Action parsing and validation
- ✅ Basic gameplay sequences
- ✅ Reward calculation verification

### Running Tests

```bash
# Run north star tests (requires server)
cd local_testing && ../../../.venv/bin/python test_north_star.py

# Run adapter integration tests (no server needed)
cd local_testing && ../../../.venv/bin/python test_taxi_integration.py

# Run with pytest
python -m pytest local_testing/ -v
```

## 🔧 Configuration Options

The taxi environment supports various configuration parameters:

```python
config = {
    "is_raining": False,      # If True, movement success rate is 80%
    "fickle_passenger": False # If True, passenger may change destinations
}
```

## 🔍 Understanding Game States

The taxi adapter provides helpful state decoding utilities:

```python
from mcp_server.taxi_adapter import TaxiAdapter

adapter = TaxiAdapter()

# Decode a state
state = 328
decoded = adapter.decode_state(state)
# Returns: {
#   "taxi_row": 1,
#   "taxi_col": 3,
#   "passenger_location": 2,  # 0-3: locations, 4: in taxi
#   "destination": 0          # 0: Red, 1: Green, 2: Yellow, 3: Blue
# }

# Get human-readable description
description = adapter.get_state_description(state)
# Returns: "Taxi at T (1, 3), Passenger at Yellow, Destination: r (Red), must pickup passenger"
```

## 📊 Expected Results

### North Star Test Output

**Recording Mode (First Run):**
```
🌟 Testing Simplified North Star Interface - Taxi Environment
📝 === RECORDING MODE ===
🎬 Setting REWARD_KIT_PLAYBACK_FILE=recording_trajectories.jsonl
✅ Policy created in live mode
✅ MCP environments created successfully
✅ Completed 3 trajectories in 45.23s
🚕 Trajectories completed: 3
✅ Successful: 2/3
🏆 Recording phase completed successfully!
```

**Playback Mode (Subsequent Runs):**
```
🌟 Testing Simplified North Star Interface - Taxi Environment  
🎬 === PLAYBACK MODE ===
📂 Using existing file: recording_trajectories.jsonl
✅ Policy created in playback mode
✅ MCP environments created successfully
✅ Completed 3 trajectories in 1.45s
⚡ Playback speedup: ~31x faster than recording
🏆 Playback phase completed successfully!
```

### Integration Test Output
```
✅ Adapter created successfully
✅ Environment created with default config
✅ State decoding works correctly
✅ Action parsing works correctly
✅ Basic gameplay sequence successful
✅ All adapter integration tests passed!
```

## 🏢 Architecture Details

### MCP Server Features

- **Multi-Session Support**: Handles multiple concurrent taxi sessions
- **Proper MCP Implementation**: Uses StreamableHTTP transport  
- **State Persistence**: Sessions maintain state across tool calls
- **Robust Error Handling**: Validates actions and provides clear feedback
- **Configurable Environment**: Supports deterministic and stochastic modes

### Taxi Adapter Interface

The `TaxiAdapter` implements the `EnvironmentAdapter` interface:

```python
# Core environment methods
create_environment(config) -> TaxiEnv
create_environment_with_seed(config, seed) -> Tuple[TaxiEnv, int, Dict]
reset_environment(env, seed) -> Tuple[int, Dict]
step_environment(env, action) -> Tuple[int, float, bool, bool, Dict]

# Taxi-specific methods  
decode_state(state: int) -> Dict[str, Any]
get_state_description(state: int) -> str
parse_action(action_str: str) -> int

# Metadata methods
get_action_space_description() -> Dict[str, Any]
get_observation_space_description() -> Dict[str, Any]
get_default_config() -> Dict[str, Any]
```

## 📋 Testing Decision Guide

**Choose North Star API Testing if:**
- Testing complete reward-kit integration
- Developing LLM policies for taxi navigation
- Need record-and-playback capabilities
- Want realistic evaluation scenarios

**Choose Direct Adapter Testing if:**
- Developing/debugging adapter logic
- Running automated tests in CI/CD
- Want fast, reliable test execution
- Testing specific adapter methods

**Choose Manual Server Testing if:**
- Testing MCP protocol integration
- Developing custom MCP clients
- Testing server deployment scenarios
- Need interactive debugging

## 🐛 Troubleshooting

### Common Issues

**Server Connection Errors:**
```bash
# Check if server is running
curl http://localhost:8000/mcp/

# Check server logs
cd mcp_server && ../../../.venv/bin/python simulation_server.py --verbose
```

**Environment Creation Fails:**
```bash
# Verify gymnasium installation
python -c "import gymnasium as gym; print(gym.make('Taxi-v3'))"

# Install taxi dependencies
pip install gymnasium[toy_text]
```

**State Decoding Issues:**
- Taxi states range from 0-499
- Verify state bounds before decoding
- Check environment reset returns valid initial state

**Import Errors:**
```bash
# Ensure reward-kit is installed in development mode
pip install -e .

# Check adapter imports
python -c "from mcp_server.taxi_adapter import TaxiAdapter; print('OK')"
```

### Debug Mode

Enable detailed logging:

```bash
# Start server with debug logging
python simulation_server.py --log-level DEBUG

# Run tests with verbose output
python test_north_star.py --verbose
```

## 🔗 Related Examples

- **`frozen_lake_mcp_complete/`**: Similar MCP implementation for FrozenLake
- **`apps_coding_example/`**: Code execution evaluation example
- **`math_example/`**: Mathematical reasoning evaluation

## 📚 Learning Resources

- [Taxi Environment Documentation](https://gymnasium.farama.org/environments/toy_text/taxi/)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [Reward-Kit Documentation](../../docs/)
- [MCP North Star Design](../../development/mcp_north_star.md)

## 🤝 Contributing

When modifying this example:

1. **Follow [CONTRIBUTING.md](../../development/CONTRIBUTING.md)** standards
2. **Test locally first** using the comprehensive test suite
3. **Update documentation** for structural changes
4. **Run code quality checks** before submitting

```bash
# Code quality checks
.venv/bin/black examples/taxi_mcp_complete
.venv/bin/flake8 examples/taxi_mcp_complete  
.venv/bin/mypy examples/taxi_mcp_complete
```

---

This example demonstrates a production-ready MCP server implementation suitable for cloud deployment and integration with LLM applications requiring taxi navigation capabilities. It showcases the reward-kit north star API with record-and-playback for efficient development and evaluation workflows. 