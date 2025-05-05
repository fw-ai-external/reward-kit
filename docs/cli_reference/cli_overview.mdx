# Command Line Interface Reference

The Reward Kit provides a command-line interface (CLI) for common operations like previewing evaluations, deploying reward functions, and running agent evaluations.

## Installation

When you install the Reward Kit, the CLI is automatically installed:

```bash
pip install reward-kit
```

You can verify the installation by running:

```bash
reward-kit --help
```

## Authentication Setup

Before using the CLI, set up your authentication credentials:

```bash
# Set your API key
export FIREWORKS_API_KEY=your_api_key

# Optional: Set the API base URL (for development environments)
export FIREWORKS_API_BASE=https://api.fireworks.ai
```

## Command Overview

The Reward Kit CLI supports the following main commands:

- `preview`: Preview an evaluation with sample data
- `deploy`: Deploy a reward function as an evaluator
- `agent-eval`: Run agent evaluations on task bundles 
- `list`: List existing evaluators (coming soon)
- `delete`: Delete an evaluator (coming soon)

## Preview Command

The `preview` command allows you to test an evaluation with sample data before deployment.

### Syntax

```bash
reward-kit preview [options]
```

### Options

- `--metrics-folders`: Specify metrics to use in the format "name=path"
- `--samples`: Path to a JSONL file containing sample conversations
- `--max-samples`: Maximum number of samples to process (optional)
- `--output`: Path to save preview results (optional)
- `--verbose`: Enable verbose output (optional)

### Examples

```bash
# Basic usage
reward-kit preview --metrics-folders "clarity=./my_metrics/clarity" --samples ./samples.jsonl

# Multiple metrics
reward-kit preview --metrics-folders "clarity=./my_metrics/clarity" "accuracy=./my_metrics/accuracy" --samples ./samples.jsonl

# Limit sample count
reward-kit preview --metrics-folders "clarity=./my_metrics/clarity" --samples ./samples.jsonl --max-samples 5

# Save results to file
reward-kit preview --metrics-folders "clarity=./my_metrics/clarity" --samples ./samples.jsonl --output ./results.json
```

### Sample File Format

The samples file should be a JSONL (JSON Lines) file with each line containing a conversation in the following format:

```json
{"messages": [{"role": "user", "content": "What is machine learning?"}, {"role": "assistant", "content": "Machine learning is a method of data analysis..."}]}
```

## Deploy Command

The `deploy` command deploys a reward function as an evaluator on the Fireworks platform.

### Syntax

```bash
reward-kit deploy [options]
```

### Options

- `--id`: ID for the deployed evaluator (required)
- `--metrics-folders`: Specify metrics to use in the format "name=path" (required)
- `--display-name`: Human-readable name for the evaluator (optional)
- `--description`: Description of the evaluator (optional)
- `--force`: Overwrite if an evaluator with the same ID already exists (optional)
- `--providers`: List of model providers to use (optional)
- `--verbose`: Enable verbose output (optional)

### Examples

```bash
# Basic deployment
reward-kit deploy --id my-evaluator --metrics-folders "clarity=./my_metrics/clarity"

# With display name and description
reward-kit deploy --id my-evaluator \
  --metrics-folders "clarity=./my_metrics/clarity" \
  --display-name "Clarity Evaluator" \
  --description "Evaluates responses based on clarity"

# Force overwrite existing evaluator
reward-kit deploy --id my-evaluator \
  --metrics-folders "clarity=./my_metrics/clarity" \
  --force

# Multiple metrics
reward-kit deploy --id comprehensive-evaluator \
  --metrics-folders "clarity=./my_metrics/clarity" "accuracy=./my_metrics/accuracy" \
  --display-name "Comprehensive Evaluator"
```

## Common Workflows

### Iterative Development Workflow

A typical development workflow might look like:

1. Create a reward function
2. Preview it with sample data
3. Refine the function based on preview results
4. Deploy when satisfied

```bash
# Step 1: Create a reward function (manually in ./my_metrics/clarity)

# Step 2: Preview with samples
reward-kit preview --metrics-folders "clarity=./my_metrics/clarity" --samples ./samples.jsonl

# Step 3: Refine the function (manually)

# Step 4: Preview again
reward-kit preview --metrics-folders "clarity=./my_metrics/clarity" --samples ./samples.jsonl

# Step 5: Deploy when satisfied
reward-kit deploy --id clarity-evaluator \
  --metrics-folders "clarity=./my_metrics/clarity" \
  --display-name "Clarity Evaluator" \
  --description "Evaluates response clarity" \
  --force
```

### Comparing Multiple Metrics

You can preview multiple metrics to compare their performance:

```bash
# Preview with multiple metrics
reward-kit preview \
  --metrics-folders \
  "metric1=./my_metrics/metric1" \
  "metric2=./my_metrics/metric2" \
  "metric3=./my_metrics/metric3" \
  --samples ./samples.jsonl
```

### Deployment with Custom Providers

You can deploy with specific model providers:

```bash
# Deploy with custom provider
reward-kit deploy --id my-evaluator \
  --metrics-folders "clarity=./my_metrics/clarity" \
  --providers '[{"providerType":"anthropic","modelId":"claude-3-sonnet-20240229"}]'
```

## Agent-Eval Command

The `agent-eval` command enables you to run agent evaluations using task bundles.

### Syntax

```bash
reward-kit agent-eval [options]
```

### Options

#### Task Specification:
- `--task-dir`: Path to task bundle directory containing reward.py, tools.py, etc.
- `--dataset` or `-d`: Path to JSONL file containing task specifications.

#### Output and Models:
- `--output-dir` or `-o`: Directory to store evaluation runs (default: "./runs").
- `--model`: Override MODEL_AGENT environment variable.
- `--sim-model`: Override MODEL_SIM environment variable for simulated user.

#### Testing and Debugging:
- `--no-sim-user`: Disable simulated user (use static initial messages only).
- `--test-mode`: Run in test mode without requiring API keys.
- `--mock-response`: Use a mock agent response (works with --test-mode).
- `--debug`: Enable detailed debug logging.
- `--validate-only`: Validate task bundle structure without running evaluation.
- `--export-tools`: Export tool specifications to directory for manual testing.

#### Advanced Options:
- `--task-ids`: Comma-separated list of task IDs to run.
- `--max-tasks`: Maximum number of tasks to evaluate.
- `--registries`: Custom tool registries in format 'name=path'.
- `--registry-override`: Override all toolset paths with this registry path.
- `--evaluator`: Custom evaluator module path (overrides default).

### Examples

```bash
# Run agent evaluation with default settings
export MODEL_AGENT=openai/gpt-4o-mini
reward-kit agent-eval --task-dir examples/flight_task

# Use a specific dataset file
reward-kit agent-eval --dataset examples/flight_task/task.jsonl

# Run in test mode (no API keys required)
reward-kit agent-eval --task-dir examples/flight_task --test-mode --mock-response

# Validate task bundle structure without running
reward-kit agent-eval --task-dir examples/flight_task --validate-only

# Use a custom model and limit to specific tasks
reward-kit agent-eval --task-dir examples/flight_task \
  --model anthropic/claude-3-opus-20240229 \
  --task-ids flight.booking.001,flight.booking.002

# Export tool specifications for manual testing
reward-kit agent-eval --task-dir examples/flight_task --export-tools ./tool_specs
```

### Task Bundle Structure

A task bundle is a directory containing the following files:
- `reward.py`: Reward function with @reward_function decorator
- `tools.py`: Tool registry with tool definitions
- `task.jsonl`: Dataset rows with task specifications
- `seed.sql` (optional): Initial database state

See the [Agent Evaluation](../developer_guide/agent_evaluation.md) guide for more details.

## Environment Variables

The CLI recognizes the following environment variables:

- `FIREWORKS_API_KEY`: Your Fireworks API key (required for deployment operations)
- `FIREWORKS_API_BASE`: Base URL for the Fireworks API (defaults to `https://api.fireworks.ai`)
- `FIREWORKS_ACCOUNT_ID`: Your Fireworks account ID (optional, can be configured in auth.ini)
- `MODEL_AGENT`: Default agent model to use (e.g., "openai/gpt-4o-mini")
- `MODEL_SIM`: Default simulation model to use (e.g., "openai/gpt-3.5-turbo")

## Troubleshooting

### Common Issues

1. **Authentication Errors**:
   ```
   Error: Authentication failed. Check your API key.
   ```
   Solution: Ensure `FIREWORKS_API_KEY` is correctly set.

2. **Metrics Folder Not Found**:
   ```
   Error: Metrics folder not found: ./my_metrics/clarity
   ```
   Solution: Check that the path exists and contains a valid `main.py` file.

3. **Invalid Sample File**:
   ```
   Error: Failed to parse sample file. Ensure it's a valid JSONL file.
   ```
   Solution: Verify the sample file is in the correct JSONL format.

4. **Deployment Permission Issues**:
   ```
   Error: Permission denied. Your API key doesn't have deployment permissions.
   ```
   Solution: Use a production API key with deployment permissions or request additional permissions.

5. **Task Bundle Validation Errors**:
   ```
   Error: Missing required files in task bundle: tools.py, reward.py
   ```
   Solution: Ensure your task bundle has all required files.

6. **Model API Key Not Set**:
   ```
   Warning: MODEL_AGENT environment variable is not set
   ```
   Solution: Set the MODEL_AGENT environment variable or use the --model parameter.

7. **Import Errors with Task Bundle**:
   ```
   Error: Failed to import tool registry from example.task.tools
   ```
   Solution: Check that the Python path is correct and the module can be imported.

### Getting Help

For additional help, use the `--help` flag with any command:

```bash
reward-kit --help
reward-kit preview --help
reward-kit deploy --help
reward-kit agent-eval --help
```

## Next Steps

- Explore the [Developer Guide](../developer_guide/getting_started.md) for conceptual understanding
- Try the [Creating Your First Reward Function](../tutorials/creating_your_first_reward_function.md) tutorial
- Learn about [Agent Evaluation](../developer_guide/agent_evaluation.md) to create your own task bundles
- See [Examples](../examples/basic_reward_function.md) for practical implementations