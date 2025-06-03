# MCP Agent RL Example (Minimal Framework Approach)

This example demonstrates a minimal framework approach where users define everything and the framework just orchestrates the basic flow.

## Architecture Principle

**Framework does the minimum**: Provision MCP → Execute LLM → Capture state → Call reward function
**User defines everything else**: MCP setup, rollout count, environment initialization, evaluation logic

## What the Framework Provides

```python
# reward_kit/mcp_agent/runner.py - Minimal orchestration only
class MCPAgentRunner:
    def run_evaluation(self, config):
        for sample in load_dataset(config.dataset):
            for rollout_idx in range(sample.get('rollout_count', 1)):
                # 1. Call user's setup function
                mcp_session = user_setup_function(sample)

                # 2. Execute LLM with tools
                response = llm_client.generate(sample['prompt'], tools=mcp_session.tools)

                # 3. Call user's state capture function
                final_state = user_state_capture_function(mcp_session, response)

                # 4. Call user's reward function
                result = user_reward_function(sample['expected'], final_state, response)

                # 5. Call user's cleanup function
                user_cleanup_function(mcp_session)
```

## What Users Define

### 1. Dataset with Setup Instructions

```jsonl
{
  "id": "fs_move_001",
  "prompt": "Move important_document.txt from /data/source_files/ to /data/archive/",
  "rollout_count": 4,
  "expected_final_state": {
    "/data/source_files": [],
    "/data/archive": ["important_document.txt"]
  },
  "setup": {
    "mcp_server": "docker:mcp/filesystem",
    "template_files": {
      "/data/source_files/important_document.txt": "This is an important document.",
      "/data/archive/.gitkeep": ""
    },
    "tools": ["list_directory", "read_file", "write_file", "move_file"]
  }
}
```

### 2. User Setup Function

```python
# user_functions.py
def setup_mcp_environment(sample_config):
    """User-defined function to set up MCP environment for this sample."""
    setup = sample_config['setup']

    if setup['mcp_server'].startswith('docker:'):
        # User's Docker setup logic
        container = docker.run(
            setup['mcp_server'].split(':')[1],
            volumes={create_temp_dir_with_files(setup['template_files']): '/data'}
        )
        return MCPSession(container, setup['tools'])

    elif setup['mcp_server'] == 'local':
        # User's local process setup logic
        return MCPSession(local_process, setup['tools'])
```

### 3. User State Capture Function

```python
def capture_final_state(mcp_session, llm_response):
    """User-defined function to capture environment state after LLM interaction."""
    final_state = {}

    # User decides what state to capture and how
    for directory in ['/data/source_files', '/data/archive']:
        try:
            result = mcp_session.call_tool('list_directory', {'path': directory})
            final_state[directory] = parse_file_list(result)
        except:
            final_state[directory] = []

    return final_state
```

### 4. User Reward Function (Pure)

```python
@reward_function
def evaluate_filesystem_task(
    expected_final_state: Dict[str, List[str]],
    actual_final_state: Dict[str, List[str]],
    llm_response: str,
    **kwargs
) -> EvaluateResult:
    """Pure evaluation - no MCP calls, just state comparison."""

    score = 1.0 if expected_final_state == actual_final_state else 0.0

    return EvaluateResult(
        score=score,
        reason=f"Expected: {expected_final_state}, Got: {actual_final_state}",
        metrics={}
    )
```

### 5. User Cleanup Function

```python
def cleanup_mcp_environment(mcp_session):
    """User-defined cleanup logic."""
    if hasattr(mcp_session, 'container'):
        mcp_session.container.stop()
        mcp_session.container.remove()
    # Clean up temp directories, etc.
```

## Configuration

```yaml
# config.yaml - User defines everything
model: "accounts/fireworks/models/qwen3-235b-a22b"
dataset: "dataset.jsonl"

# User-defined functions
user_functions:
  setup: "user_functions.setup_mcp_environment"
  state_capture: "user_functions.capture_final_state"
  reward: "user_functions.evaluate_filesystem_task"
  cleanup: "user_functions.cleanup_mcp_environment"

# LLM settings
generation:
  temperature: 1.0
  max_tokens: 512
```

## CLI Usage

```bash
# Framework just orchestrates, user defines everything else
reward-kit mcp-agent run --config config.yaml
```

## Framework Implementation (Minimal)

```python
# reward_kit/cli_commands/mcp_agent_cmd.py
def mcp_agent_command(args):
    config = load_config(args.config)

    # Load user-defined functions
    setup_fn = load_function(config.user_functions.setup)
    capture_fn = load_function(config.user_functions.state_capture)
    reward_fn = load_function(config.user_functions.reward)
    cleanup_fn = load_function(config.user_functions.cleanup)

    dataset = load_dataset(config.dataset)
    llm_client = create_llm_client(config)

    results = []

    for sample in dataset:
        rollout_count = sample.get('rollout_count', 1)

        for rollout_idx in range(rollout_count):
            mcp_session = None
            try:
                # 1. User sets up environment
                mcp_session = setup_fn(sample)

                # 2. Framework executes LLM
                response = llm_client.generate(
                    sample['prompt'],
                    tools=mcp_session.available_tools
                )

                # 3. User captures state
                final_state = capture_fn(mcp_session, response)

                # 4. User evaluates
                result = reward_fn(
                    expected_final_state=sample['expected_final_state'],
                    actual_final_state=final_state,
                    llm_response=response
                )

                results.append({
                    'sample_id': sample['id'],
                    'rollout_idx': rollout_idx,
                    'score': result.score,
                    'reason': result.reason
                })

            finally:
                # 5. User cleans up
                if mcp_session:
                    cleanup_fn(mcp_session)

    save_results(results)
```

## What This Achieves

### Framework Responsibilities (Minimal)
- Load configuration and user functions
- Execute LLM with tools
- Call user functions in the right order
- Aggregate and save results

### User Responsibilities (Everything Else)
- Define MCP server setup (Docker, local, cloud, etc.)
- Define number of rollouts per sample
- Define environment initialization
- Define state capture logic
- Define evaluation criteria
- Define cleanup procedures

### Benefits
- **Flexible**: Users can use any MCP server, any setup
- **Simple Framework**: Just orchestration, no assumptions
- **Testable**: Each user function can be tested independently
- **Reusable**: Same pattern works for any MCP-based task

The framework doesn't know about Docker, filesystem tools, or specific evaluation criteria. It just calls user functions in sequence and aggregates results.
