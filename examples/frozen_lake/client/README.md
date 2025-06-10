# Frozen Lake Evaluation Client

This is the **client-side implementation** for evaluating LLM agents on the Frozen Lake game using reward-kit's agent evaluation framework.

## Overview

This client connects to a Frozen Lake game server via HTTP and evaluates how well an LLM agent can navigate the game. The evaluation includes:
- Agent decision-making analysis
- Success/failure metrics
- Detailed trajectory logging
- Performance scoring

## Prerequisites

### API Access
You need access to an LLM model. This example uses Fireworks AI:

```bash
export FIREWORKS_API_KEY="your_fireworks_api_key"
export MODEL_AGENT="fireworks/accounts/fireworks/models/qwen3-235b-a22b"
```

### Game Server
A Frozen Lake game server must be running (see `../server/` directory).

### Dependencies
- reward-kit with agent evaluation support
- Python 3.8+

## Configuration Files

### `task_def.yaml`
Main task definition that configures the evaluation:

```yaml
name: "frozen_lake_http_rollout"
resource_type: "http_rollout"
base_resource_config:
  base_url: "http://localhost:8080"  # Game server URL
  timeout: 30.0
reward_function_path: "examples.frozen_lake.reward.frozen_lake_reward"
poc_max_turns: 20
```

**Key Settings:**
- `base_url`: Where the game server is running
- `timeout`: Maximum time per HTTP request
- `poc_max_turns`: Maximum conversation turns
- Agent receives ultra-concise prompt to avoid analysis paralysis

### `reward.py`
Defines how agent performance is scored:

```python
@reward_function
def frozen_lake_reward(messages: List[Message], **kwargs) -> EvaluateResult:
    # Analyzes final conversation for success/failure indicators
    # Returns score 1.0 for reaching goal, 0.0 for failure
```

**Success Indicators:** "you win", "you reached the goal", "congratulations"  
**Failure Indicators:** "you lose", "game over", "you fell", "hole"

### `config.yaml` (Optional)
Additional configuration for batch evaluation or custom LLM settings.

### `initial_prompt.jsonl` (Optional)
Alternative prompt dataset if not using task_def.yaml messages.

## Running Evaluation

### Quick Start
```bash
# 1. Ensure game server is running
# 2. Set environment variables
export FIREWORKS_API_KEY="your_key"
export MODEL_AGENT="fireworks/accounts/fireworks/models/qwen3-235b-a22b"

# 3. Run evaluation
./run_evaluation.sh
```

### Manual Execution
```bash
python -m reward_kit.cli agent-eval --task-def task_def.yaml
```

## Understanding Results

### Success Metrics
- **Score 1.0**: Agent successfully navigated to goal
- **Score 0.0**: Agent failed (fell in hole or other failure)
- **Tool calls**: Number of actions taken
- **Reasoning quality**: Analysis of decision-making process

### Output Files
The evaluation generates detailed logs:

```
evaluation_logs/
├── full_evaluation_*.log     # Complete system logs
├── agent_trajectory_*.log    # Agent decision process  
└── trajectory_analysis_*.txt # Human-readable summary
```

### Example Successful Run
```
📊 SUMMARY:
• Total tool calls: 6
• Game state changes: 5
• Final result: SUCCESS - Reached goal!

🎮 DETAILED TRAJECTORY:
STEP 1: STEP - Arguments: {'action': 'right'}
STEP 2: STEP - Arguments: {'action': 'right'} 
STEP 3: STEP - Arguments: {'action': 'down'}
STEP 4: STEP - Arguments: {'action': 'down'}
STEP 5: STEP - Arguments: {'action': 'down'}
STEP 6: STEP - Arguments: {'action': 'right'}
```

## Agent Behavior

### Action Format
The agent uses **string-based actions** for clarity:
- `"left"` - Move left
- `"right"` - Move right  
- `"up"` - Move up
- `"down"` - Move down

### Initial State Injection
The agent automatically receives the game board at the start:
```
Environment: You are at position (0, 0) on a S cell.

Game Board:
[S] F  F  F 
 F  H  F  H 
 F  F  F  H 
 H  F  F  G 

Starting Position: [0, 0]
```

### Prompt Engineering
The task uses an ultra-concise prompt to prevent analysis paralysis:
```
🎮 FROZEN LAKE GAME - IMMEDIATE ACTION REQUIRED

Objective: Get from S to G without hitting H
RULES: S=start, F=safe, H=hole(death), G=goal(win)
ACTION: Use step tool with: "left", "right", "up", or "down"

⚡ NO LONG THINKING - Make your move NOW!
```

## Customization

### Different Models
Test various LLM models:
```bash
# OpenAI models
export MODEL_AGENT="openai/gpt-4o-mini"

# Other Fireworks models
export MODEL_AGENT="fireworks/accounts/fireworks/models/llama-v3p3-8b-instruct"
```

### Custom Reward Functions
Create specialized evaluation criteria:
```python
@reward_function
def path_efficiency_reward(messages: List[Message], **kwargs) -> EvaluateResult:
    # Score based on path length, safety margin, etc.
    return EvaluateResult(score=efficiency_score, reason=analysis)
```

### Batch Evaluation
Run multiple episodes or test different configurations by modifying `config.yaml`.

## Troubleshooting

### Common Issues

**Agent Analysis Paralysis**
- Symptom: Agent generates long reasoning but no actions
- Solution: Ultra-concise prompt already implemented

**Connection Errors**
- Symptom: "Connection refused" errors
- Solution: Ensure game server is running at configured URL

**Action Format Errors**
- Symptom: Agent uses numbers instead of strings
- Solution: Tool specification enforces string enum

**Missing Initial State**
- Symptom: Agent doesn't see game board
- Solution: HttpRolloutResource auto-injects initial state

### Debug Logging
Check evaluation logs for detailed HTTP communication and agent reasoning.

## Integration Notes

This client demonstrates how to:
- Connect reward-kit to HTTP-based environments
- Handle real-time agent evaluation
- Process string-based tool calling
- Implement comprehensive logging and analysis

The patterns here can be adapted for other HTTP rollout-compatible environments beyond Frozen Lake.