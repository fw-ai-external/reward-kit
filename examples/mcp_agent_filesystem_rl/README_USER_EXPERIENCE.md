# MCP Agent RL Example - User Experience

This example shows how to evaluate LLM agents that use MCP (Model Context Protocol) tools in isolated environments.

## Quick Start

```bash
# 1. Define your task and evaluation
# 2. Run evaluation with multiple rollouts
reward-kit mcp-agent run --config config.yaml

# Results saved to outputs/
```

## What You Need to Provide

### 1. Dataset (dataset.jsonl)

Each line defines a task with setup instructions:

```jsonl
{
  "id": "move_file_task",
  "prompt": "Move important_document.txt from /data/source/ to /data/archive/",
  "rollout_count": 4,
  "setup": {
    "mcp_server": "docker:mcp/filesystem",
    "template_files": {
      "/data/source/important_document.txt": "Important content here",
      "/data/archive/.gitkeep": ""
    }
  },
  "expected_outcome": {
    "files_in_source": [],
    "files_in_archive": ["important_document.txt"]
  }
}
```

### 2. Setup Function (setup.py)

How to prepare the MCP environment for each rollout:

```python
def setup_environment(sample):
    """Set up MCP environment for this task."""
    setup_config = sample['setup']

    if setup_config['mcp_server'] == 'docker:mcp/filesystem':
        # Create temp directory with template files
        temp_dir = create_temp_directory()
        for file_path, content in setup_config['template_files'].items():
            write_file(temp_dir + file_path, content)

        # Start Docker container
        container = docker.run(
            'mcp/filesystem',
            volumes={temp_dir: '/data'},
            remove=True
        )

        return MCPEnvironment(
            container=container,
            tools=['list_directory', 'read_file', 'write_file', 'move_file'],
            temp_dir=temp_dir
        )
```

### 3. State Capture Function (capture.py)

What to extract after the LLM runs:

```python
def capture_final_state(mcp_env, llm_response):
    """Capture the final state of the environment."""

    # Extract file listings from key directories
    state = {}
    for directory in ['/data/source', '/data/archive']:
        try:
            result = mcp_env.call_tool('list_directory', {'path': directory})
            state[f'files_in_{directory.split("/")[-1]}'] = parse_files(result)
        except:
            state[f'files_in_{directory.split("/")[-1]}'] = []

    return state
```

### 4. Reward Function (reward.py)

Pure evaluation logic - no MCP calls needed:

```python
from reward_kit import reward_function

@reward_function
def evaluate_file_move(
    expected_outcome: dict,
    actual_outcome: dict,
    llm_response: str
) -> EvaluateResult:
    """Evaluate if the file was moved correctly."""

    # Simple comparison
    success = (expected_outcome == actual_outcome)

    if success:
        score = 1.0
        reason = "File moved successfully"
    else:
        score = 0.0
        reason = f"Expected {expected_outcome}, got {actual_outcome}"

    return EvaluateResult(
        score=score,
        reason=reason,
        metrics={
            "exact_match": MetricResult(
                score=1.0 if success else 0.0,
                reason="State matches expected exactly"
            )
        }
    )
```

### 5. Cleanup Function (cleanup.py)

How to clean up after each rollout:

```python
def cleanup_environment(mcp_env):
    """Clean up the environment."""
    if hasattr(mcp_env, 'container'):
        mcp_env.container.stop()

    if hasattr(mcp_env, 'temp_dir'):
        shutil.rmtree(mcp_env.temp_dir)
```

### 6. Configuration (config.yaml)

Tie everything together:

```yaml
# LLM settings
model: "accounts/fireworks/models/llama-v3p1-8b-instruct"
temperature: 0.7
max_tokens: 512

# Your data and functions
dataset: "dataset.jsonl"
functions:
  setup: "setup.setup_environment"
  capture: "capture.capture_final_state"
  reward: "reward.evaluate_file_move"
  cleanup: "cleanup.cleanup_environment"

# System prompt
system_prompt: "You have access to filesystem tools. Use them to complete tasks accurately."
```

## Running the Evaluation

```bash
# Single command runs everything
reward-kit mcp-agent run --config config.yaml
```

This will:
1. Load your dataset
2. For each task, run N rollouts (as specified in dataset)
3. For each rollout:
   - Call your setup function → fresh MCP environment
   - Send prompt to LLM with available tools
   - Execute any tool calls the LLM makes
   - Call your capture function → extract final state
   - Call your reward function → compare expected vs actual
   - Call your cleanup function → remove environment
4. Save aggregated results

## Results

```jsonl
{
  "sample_id": "move_file_task",
  "rollout_idx": 0,
  "llm_response": "I'll move the file using move_file...",
  "actual_outcome": {"files_in_source": [], "files_in_archive": ["important_document.txt"]},
  "evaluation_score": 1.0,
  "evaluation_reason": "File moved successfully"
}
```

## Example File Structure

```
my_filesystem_eval/
├── config.yaml           # Main configuration
├── dataset.jsonl         # Tasks and expected outcomes
├── setup.py              # Environment setup logic
├── capture.py            # State extraction logic
├── reward.py             # Evaluation logic
├── cleanup.py            # Cleanup logic
└── results/              # Generated results
```

## What You Control

- **MCP Server**: Use any Docker image, local process, or cloud service
- **Environment Setup**: Files, databases, APIs - whatever your task needs
- **Tool Selection**: Which tools the LLM has access to
- **State Capture**: What data to extract for evaluation
- **Evaluation Logic**: How to score success/failure
- **Rollout Count**: How many independent runs per task

## What the Framework Handles

- Loading your functions and configuration
- LLM interaction and tool call execution
- Calling your functions in the right order
- Running multiple rollouts in parallel
- Aggregating results across all runs
- Error handling and resource cleanup

## Benefits

- **Simple**: Just write 4 functions, framework does the rest
- **Flexible**: Use any MCP server, any evaluation criteria
- **Isolated**: Each rollout gets a fresh environment
- **Scalable**: Run many rollouts in parallel
- **Debuggable**: Clear separation between setup, execution, and evaluation

You focus on defining your task and evaluation logic. The framework handles the orchestration.
