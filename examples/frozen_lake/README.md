# Frozen Lake HTTP Rollout Evaluation

This example demonstrates how to evaluate LLM agents on the Frozen Lake game using reward-kit's HTTP rollout evaluation framework. The agent must navigate from start (S) to goal (G) while avoiding holes (H).

## Overview

This implementation bridges reward-kit's agent evaluation framework with HTTP-based game environments, showcasing:
- **Separated server/client architecture** for clear responsibility boundaries
- **HTTP rollout protocol** for standardized environment communication  
- **Real-time agent evaluation** on interactive environments
- **String-based actions** for clearer agent decision-making
- **Comprehensive logging** and trajectory analysis

## Architecture

```
┌─────────────────┐    HTTP     ┌──────────────────┐
│ Client Side     │ ◄─────────► │ Server Side      │
│ (reward-kit)    │  Rollout    │ (Game Env)       │
│                 │             │                  │
│ • Agent Eval    │             │ • Game Logic     │
│ • Reward Func   │             │ • State Mgmt     │
│ • Trajectory    │             │ • HTTP API       │
└─────────────────┘             └──────────────────┘
```

## Project Structure

```
frozen_lake/
├── README.md                    # This overview document
├── server/                      # 🎮 Game Environment (Server Side)
│   ├── README.md               # Server setup & API documentation
│   └── http_rollout_server.py  # FastAPI game server implementation
├── client/                     # 🤖 Agent Evaluation (Client Side)  
│   ├── README.md               # Client setup & evaluation guide
│   ├── task_def.yaml           # Fireworks model evaluation config
│   ├── task_def_openai.yaml    # OpenAI model evaluation config
│   ├── reward.py               # Performance scoring function
│   ├── run_evaluation.sh       # Fireworks model test script
│   ├── run_evaluation_openai.sh # OpenAI model test script
│   └── evaluation_logs/        # Generated evaluation results
│       ├── full_evaluation_*.log      # Complete system logs
│       ├── agent_trajectory_*.log     # Agent decision traces
│       └── openai_evaluation_*.log    # OpenAI-specific logs
├── gymnasium_frozen_lake_server.py  # Gymnasium-based game implementation
└── deprecated/                 # Legacy implementations
    ├── frozen_lake_server.py   # Original hand-rolled implementation
    └── gymnasium_adapter.py    # Abstract adapter (future extensibility)
```

## Separation of Concerns

### 🎮 **Server Side** (`server/`)
**Who uses this:** Game environment developers, infrastructure teams

**Responsibilities:**
- Implements the game logic and rules
- Provides HTTP API endpoints (`/start_episode`, `/step`, `/end_episode`)
- Manages game state and episode lifecycle
- Returns structured observations and rewards
- Can be deployed independently and reused across evaluations

### 🤖 **Client Side** (`client/`)  
**Who uses this:** ML researchers, agent evaluation teams

**Responsibilities:**
- Configures agent evaluation parameters
- Defines reward functions and success criteria
- Handles LLM model integration (Fireworks, OpenAI, etc.)
- Processes agent actions and responses
- Generates evaluation metrics and analysis

### 🔗 **HTTP Rollout Protocol**
Clean and simple standardized communication interface:
1. **Episode Start:** `POST /start_episode` → Initialize game
2. **Action Execution:** `POST /step` → Execute agent action
3. **Episode End:** `POST /end_episode` → Cleanup
4. **Health Check:** `GET /health` → Status check

## Gymnasium Integration

This implementation is built on the official **Gymnasium FrozenLake-v1** environment, providing:

### 🏗️ **Standard RL Compatibility**
- Uses the official Gymnasium interface for consistent behavior
- Compatible with other RL frameworks and tools
- Supports both deterministic (`is_slippery=False`) and stochastic (`is_slippery=True`) modes
- Official action space (0=Left, 1=Down, 2=Right, 3=Up) with string conversion layer

### 🎛️ **Configurable Environments**
- **Map sizes**: 4x4 (default) and 8x8 grids
- **Slipperiness**: Toggle deterministic vs stochastic behavior
- **Render modes**: Support for different visualization options
- **Custom maps**: Can be extended to support custom grid layouts

### 🔌 **Extensible Architecture**
Clean and focused implementation that can be extended for other Gymnasium environments in the future.

### 🧹 **Clean API Design**
- Focused on core functionality with minimal endpoints
- Initial prompt embedded in start_episode response
- Environment information available through Gymnasium metadata

## Game Rules

**Objective:** Navigate from S to G without falling into holes (H)

**Game Board:**
```
[S] F  F  F 
 F  H  F  H 
 F  F  F  H 
 H  F  F  G 
```

**Legend:**
- `S` = Start position
- `F` = Frozen (safe to step on) 
- `H` = Hole (game over if you step here)
- `G` = Goal (win condition)
- `[X]` = Current position

**Actions:** `"left"`, `"right"`, `"up"`, `"down"`

## Setup

### Prerequisites

1. **reward-kit installed** with agent evaluation support
2. **Fireworks API key** for LLM model access
3. **Python 3.8+** with required dependencies

### API Key Configuration

For **Fireworks models** (qwen3):
```bash
export FIREWORKS_API_KEY="your_fireworks_api_key_here"
export MODEL_AGENT="accounts/fireworks/models/qwen3-235b-a22b"
```

For **OpenAI models** (GPT-4):
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export MODEL_AGENT="openai/gpt-4.1-2025-04-14"
```

## Getting Started

### For Game Environment Developers (Server Side)
If you're implementing a game environment for HTTP rollout evaluation:

1. **Review the API specification** in `server/README.md`
2. **Study the reference implementation** in `server/http_rollout_server.py`
3. **Implement the required endpoints** for your game
4. **Test with the provided client** to verify compatibility

### For Agent Evaluation Teams (Client Side)  
If you're evaluating agents on an existing HTTP rollout environment:

1. **Set up your LLM API credentials** (Fireworks, OpenAI, etc.)
2. **Configure the evaluation parameters** in `client/task_def.yaml`
3. **Customize reward functions** in `client/reward.py` if needed
4. **Run the evaluation** using `client/run_evaluation.sh`

### Quick Start - Model Comparison
Run both Fireworks and OpenAI models to compare performance:

**Test Fireworks qwen3 model:**
```bash
cd client/
export FIREWORKS_API_KEY="your_key"
./run_evaluation.sh
```

**Test OpenAI GPT-4.1 model:**
```bash
cd client/
export OPENAI_API_KEY="your_key" 
./run_evaluation_openai.sh
```

### Full End-to-End Setup
To run the complete example from scratch:

```bash
# 1. Start the game server (in one terminal)
cd server/
python http_rollout_server.py

# 2. Run agent evaluation (in another terminal)  
cd client/
# For Fireworks:
export FIREWORKS_API_KEY="your_key"
./run_evaluation.sh

# OR for OpenAI:
export OPENAI_API_KEY="your_key"
./run_evaluation_openai.sh
```

## Key Features

### 🎯 **String-Based Actions**
Clear action commands (`"left"`, `"right"`, `"up"`, `"down"`) instead of confusing numeric codes.

### 🏋️ **Gymnasium Integration**
Built on the official Gymnasium FrozenLake-v1 environment for standard RL compatibility and extensibility.

### 🔄 **Initial State Injection**  
Agent automatically receives the game board and rules at the start of each episode.

### ⚡ **Analysis Paralysis Prevention**
Ultra-concise prompts prevent the agent from getting stuck in long reasoning loops.

### 📊 **Comprehensive Logging**
Detailed trajectory analysis including agent reasoning, tool calls, and game state changes.

### 🔧 **Modular Architecture**
Server and client can be developed, deployed, and maintained independently.

### 🌟 **Extensible Environment Support**
Abstraction layer supports multiple Gymnasium environments beyond FrozenLake.

## Model Performance Results

We've tested multiple LLM models with different prompting strategies. Here are the comprehensive results:

### 🏆 Winner: Fireworks qwen3-235b-a22b (Enhanced Prompt)
```bash
Strategy: down→down→right→right→down→right
Steps: 6 moves
Result: ✅ SUCCESS - Reached goal at (3,3)!
Path: (0,0) → (1,0) → (2,0) → (2,1) → (2,2) → (3,2) → (3,3)
Score: 1.0
```

### Other Model Results
| Model | Strategy | Steps | Result | Score |
|-------|----------|-------|--------|-------|
| qwen3 (original) | right→right→right→down | 4 | ❌ Failed at [1,3] | 0.0 |
| GPT-4-1106 | down→down→right→right→right | 5 | ❌ Failed at [2,3] | 0.0 |
| GPT-4.1-2025 | right→right→right→down | 4 | ❌ Failed at [1,3] | 0.0 |

### Key Insights
- **Prompt engineering matters**: Enhanced autonomous gameplay instructions improved qwen3 from failure to complete victory
- **Strategic planning**: The winning model showed sophisticated path planning in reasoning traces
- **Model-specific responses**: Different models responded differently to the same prompt improvements

## Example Output

### Successful Evaluation (qwen3 Enhanced)
```bash
🎮 FROZEN LAKE HTTP ROLLOUT EVALUATION
========================================

📊 AGENT TRAJECTORY SUMMARY:
• Total tool calls made: 6
• Result: ✅ SUCCESS - Agent reached the goal!

🎮 DETAILED TRAJECTORY:
STEP 1: down - Move to (1,0) on F cell
STEP 2: down - Move to (2,0) on F cell  
STEP 3: right - Move to (2,1) on F cell
STEP 4: right - Move to (2,2) on F cell
STEP 5: down - Move to (3,2) on F cell
STEP 6: right - Move to (3,3) on G cell - VICTORY!

📄 Final Score: 1.0 - Successfully reached the goal in Frozen Lake

🎉 Congratulations! You've successfully navigated the Frozen Lake!
```

### Failed Evaluation Example
```bash
📊 AGENT TRAJECTORY SUMMARY:
• Total tool calls made: 4
• Result: ❌ FAILURE - Agent fell into hole

🎮 DETAILED TRAJECTORY:
STEP 1: right - Move to (0,1) on F cell
STEP 2: right - Move to (0,2) on F cell
STEP 3: right - Move to (0,3) on F cell
STEP 4: down - Move to (1,3) on H cell - GAME OVER

📄 Final Score: 0.0 - Failed to reach the goal (fell into hole)
```

## Reproducing the Evaluation Results

### Step 1: Setup Environment
```bash
# Install reward-kit and dependencies
cd /path/to/reward-kit/examples/frozen_lake

# Set up API keys
export FIREWORKS_API_KEY="your_fireworks_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```

### Step 2: Test the Winning Model (qwen3 Enhanced)
```bash
cd client/
./run_evaluation.sh
```

**Expected Output:**
- 6 tool calls made
- Success path: down→down→right→right→down→right  
- Final score: 1.0
- Logs saved to `evaluation_logs/full_evaluation_*.log`

### Step 3: Compare with OpenAI GPT-4.1
```bash
cd client/
./run_evaluation_openai.sh
```

**Expected Output:**
- 4 tool calls made
- Failed path: right→right→right→down
- Final score: 0.0
- Logs saved to `evaluation_logs/openai_evaluation_*.log`

### Step 4: Analyze Results
```bash
# View detailed agent reasoning
cat evaluation_logs/agent_trajectory_*.log

# Compare model strategies
ls evaluation_logs/
```

### Step 5: Test Different Prompts
To test the original (weaker) prompt, modify the server's `/initial_prompt` endpoint in `server/http_rollout_server.py` to use the original shorter prompt without autonomous instructions.

## Success Metrics

- **Score 1.0:** Agent successfully navigated to goal
- **Score 0.0:** Agent failed (fell in hole or got stuck)
- **Tool calls:** Number of actions taken by the agent
- **Trajectory quality:** Analysis of decision-making efficiency
- **Strategy effectiveness:** Path planning and hole avoidance

## Customization & Extension

### Server Side Customization
See `server/README.md` for:
- Custom game board layouts
- Different game mechanics  
- API endpoint modifications
- Deployment configurations

### Client Side Customization
See `client/README.md` for:
- Different LLM models (OpenAI, Fireworks, etc.)
- Custom reward functions
- Batch evaluation setups
- Prompt engineering strategies

## HTTP Rollout Protocol

This example implements a standardized communication protocol:

1. **Episode Start:** `POST /start_episode` → Initialize game
2. **Action Execution:** `POST /step` → Execute agent action
3. **Episode End:** `POST /end_episode` → Cleanup

This protocol can be adapted for other interactive environments beyond Frozen Lake.

## Troubleshooting

### Common Issues
- **Connection errors:** Ensure server is running at configured URL
- **API key issues:** Verify LLM model access credentials  
- **Action format errors:** System enforces string-based actions
- **Analysis paralysis:** Ultra-concise prompts prevent reasoning loops

### Debug Resources
- Check `client/evaluation_logs/` for detailed execution traces
- Review individual README files for component-specific guidance
- Verify HTTP communication in server logs

## ✨ Migration Summary: Hand-Rolled → Gymnasium

This project successfully migrated from a custom FrozenLake implementation to the **official Gymnasium FrozenLake-v1 environment**.

### 🎯 **Key Achievements**
- ✅ **Standard RL Compatibility**: Now uses official Gymnasium interface
- ✅ **Simplified Architecture**: Clean 4-endpoint HTTP API (was 6+ endpoints)
- ✅ **Backward Compatibility**: All existing evaluation scripts work unchanged
- ✅ **Better Reliability**: Battle-tested Gymnasium code vs custom implementation
- ✅ **Future Ready**: Foundation for supporting more Gymnasium environments

### 🔄 **What Changed**
- `frozen_lake_server.py` → `gymnasium_frozen_lake_server.py` (Gymnasium-based)
- HTTP endpoints simplified from 6 to 4 core endpoints
- Initial prompt now embedded in start_episode (not separate endpoint)
- Environment info available through Gymnasium metadata (not separate endpoint)

### 📦 **What Stayed the Same**
- All evaluation scripts work without changes
- Same string-based action interface (`"left"`, `"right"`, etc.)
- Same HTTP rollout protocol structure
- Same reward functions and scoring

## Use Cases

This example serves as a foundation for:
- **Game environment developers:** Reference implementation for HTTP rollout API with Gymnasium
- **Agent evaluation teams:** Production-ready evaluation framework with standard RL environments
- **Research:** Comparative studies across different Gymnasium environments and agents
- **Infrastructure:** Template for deploying scalable evaluation systems with official RL environments

## Next Steps

1. **Start with the basics:** Run the example end-to-end
2. **Customize for your needs:** Modify game rules or evaluation criteria  
3. **Scale up:** Deploy server/client architecture for production use
4. **Extend:** Apply the pattern to other interactive environments

For detailed implementation guidance, see the respective README files in `server/` and `client/` directories.