# BFCL Agent Example

This directory contains a standalone example of how to use the Berkeley Function Call Leaderboard (BFCL) evaluation framework with LLM agents. It's designed to be a self-contained example that developers can learn from and adapt to their own needs.

## Directory Structure

```
bfcl_agent_example/
├── envs/                  # Environment implementations
│   ├── __init__.py
│   ├── gorilla_file_system.py  # File system environment
│   └── posting_api.py     # Twitter API environment
├── framework.py           # Core framework components
├── tasks/                 # Task definitions
│   └── file_management_task.yaml  # Example task
├── bfcl_runner.py         # Command-line runner
└── README.md              # This file
```

## Core Components

1. **Environments** (`envs/`): Implementations of simulated environments that agents can interact with.

2. **Framework** (`framework.py`): Contains the reusable components:
   - `BFCLResource`: Manages environments and tool execution
   - `BFCLOrchestrator`: Handles agent-environment interaction
   - Helper functions for running evaluations

3. **Tasks** (`tasks/`): YAML definitions of evaluation tasks, specifying:
   - Initial environment state
   - User messages
   - Ground truth function calls
   - Evaluation criteria

## Running an Evaluation

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=sk-...

# Run an evaluation task
python bfcl_runner.py --task tasks/file_management_task.yaml --model openai/gpt-4
```

Additional options:
- `--verbose`: Enable verbose logging
- `--output <file>`: Save results to a JSON file

## Creating Your Own Tasks

To create your own evaluation tasks:

1. Define environment classes in `envs/` if needed, or use existing ones
2. Create a YAML task definition in `tasks/` with:
   ```yaml
   name: my_task
   description: "Task description"

   base_resource_config:
     involved_classes:
       - MyEnvironmentClass
     initial_config:
       MyEnvironmentClass:
         # Initial state configuration

   messages:
     - role: user
       content: "First user instruction"
     - role: user
       content: "Second user instruction"

   evaluation_criteria:
     ground_truth_function_calls:
       - - function1(arg1='value1')
         - function2(arg2='value2')
       - - function3(arg3='value3')
   ```

3. Run your task with the `bfcl_runner.py` script

## Extending the Framework

The framework is designed to be extended:

- Create custom environment classes for specific domains
- Extend `BFCLResource` to add custom serialization logic
- Extend `BFCLOrchestrator` to customize evaluation methods

## Framework Design Principles

This example follows key design principles:

1. **Minimal Code Duplication**: The framework handles common functionality, so developers only need to focus on environment and task definitions.

2. **Separation of Concerns**:
   - Environments define available tools and state
   - Tasks define the evaluation scenario
   - Framework handles orchestration and evaluation

3. **Extensibility**: The framework can be extended with custom logic at any level.

4. **Standalone**: This example can run independently without other dependencies.

## Learning from This Example

This example demonstrates:

1. How to define simulated environments for agent evaluation
2. How to structure multi-turn conversations with tool use
3. How to evaluate agent performance against ground truth
4. How to implement a reusable framework for agent evaluation

You can adapt this framework for your own agent evaluation tasks by creating custom environments and task definitions.
