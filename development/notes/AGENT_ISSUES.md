# Agent Evaluation Framework Design

## Current State and Vision

The agent evaluation framework in Reward Kit has evolved from a SQL-centric "Task Bundle" design to a more versatile "ForkableResource" architecture. This document outlines the current state of the framework, core design principles, implementation status, and next steps to improve developer experience.

## 1. Core Design Principles

### 1.1 Resource Abstraction with `ForkableResource`

The framework revolves around the `ForkableResource` interface, which provides a unified approach to:

- **Environment Management**: Any stateful environment (SQL database, filesystem, Docker container, etc.) can be represented
- **Forking/Checkpointing**: Create identical, independent copies of an environment's state
- **Tool Exposure**: Dynamically expose available tools based on the environment's state

```python
class ForkableResource(ABC):
    async def setup(self, config: Dict) -> None: ...
    async def fork(self) -> 'ForkableResource': ...
    async def checkpoint(self) -> Any: ...
    async def restore(self, state_data: Any) -> None: ...
    async def step(self, action_name: str, action_params: Dict) -> Any: ...
    async def get_observation(self) -> Any: ...
    async def get_tools_spec(self) -> List[Dict]: ...
    async def close(self) -> None: ...
```

### 1.2 Orchestrator-Based Task Management

The `Orchestrator` class manages the lifecycle of an agent evaluation task:

- Loading task definitions
- Setting up resources
- Managing agent-environment interactions
- Collecting and evaluating results

### 1.3 Resource Types

Current implementations include:

- `SQLResource`: For database-backed environments
- `FileSystemResource`: For file/directory operations
- `PythonStateResource`: For in-memory Python objects
- `DockerResource`: For containerized environments
- `BFCLSimAPIResource`: For Berkeley Function Call Leaderboard simulated APIs

### 1.4 YAML-Based Task Definitions

Tasks are defined in structured YAML files that specify:

- Resource type and configuration
- Tools and reward function
- Evaluation criteria
- Initial messages and context

```yaml
name: "Flight Booking Task"
description: "Agent needs to book a flight given user constraints."
resource_type: "SQLResource"
base_resource_config:
  db_type: "sqlite"
  schema_file: "schema.sql"
  seed_data_file: "seed.sql"
tools_module_path: "flight_task/sql_tools.py"
reward_function_path: "flight_task/reward.py"
```

## 2. Implementation Status

### 2.1 Core Components (Complete)

- ✅ `ForkableResource` abstract base class
- ✅ `Orchestrator` class for task management
- ✅ Basic resource implementations: `SQLResource`, `FileSystemResource`, `PythonStateResource`, `DockerResource`
- ✅ BFCL integration via `BFCLSimAPIResource`
- ✅ CLI integration with `agent-eval-v2` command
- ✅ Task definition model via Pydantic

### 2.2 Current Capabilities

- ✅ End-to-end evaluation workflow for agent tasks
- ✅ Multi-turn conversations with tool usage
- ✅ OpenAI API integration (function calling)
- ✅ Metrics collection for evaluation
- ✅ BFCL task execution and evaluation

### 2.3 Known Issues

- 🔴 Unit test failures for V2 stack (Orchestrator, Resources)
- 🔴 Issues with tool discovery in `Orchestrator._get_available_tools()`
- 🔴 BFCL reward function needs alignment with original BFCL evaluation logic
- 🔴 Developer experience needs improvement (per-task orchestration)

## 3. Developer Experience Improvements Needed

### 3.1 Per-Task Orchestration (High Priority)

Current limitation: Orchestrator, tasks, and BFCLSimAPIResource are not cleanly separated per task, making it difficult for developers to create and run evaluations.

**Required Changes:**
- Refactor to make Orchestrator, tasks, and resources per-task instances
- Enable CLI to specify and run tasks end-to-end
- Simplify task definition and component discovery

### 3.2 Improved CLI Workflow

- Create helper commands to generate task templates
- Add interactive mode for task development
- Provide clear, actionable error messages
- Implement task validation before execution

### 3.3 Documentation and Examples

- Create comprehensive guides for creating custom resources
- Provide example tasks for each resource type
- Document task definition schema with examples
- Create tutorials for common evaluation patterns

## 4. BFCL Integration Status

The Berkeley Function Call Leaderboard (BFCL) integration is operational but requires refinement:

### 4.1 Current Status

- ✅ `BFCLSimAPIResource` wraps BFCL environments and infers tool schemas
- ✅ Dataset conversion from original BFCL to YAML task definitions
- ✅ Multi-turn task execution with OpenAI models
- ✅ State serialization and tool execution

### 4.2 Reward Function Improvements Needed

The current `bfcl_reward.py` needs enhancement to match original BFCL evaluation logic:

- 🔴 Implement per-turn state and response checking
- 🔴 Refine state comparison for complex objects like file systems
- 🔴 Align scoring logic with original BFCL binary pass/fail approach

## 5. Next Steps

### 5.1 Immediate Priorities

1. **Fix Unit Tests for V2 Stack**
   - Resolve issues in `tests/test_agent_v2_orchestrator.py`
   - Fix tool discovery and AsyncMock handling
   - Ensure proper async interface across resources

2. **Refine BFCL Reward Logic**
   - Implement per-turn simulation and checking
   - Match original `state_checker` and `response_checker` behavior
   - Update scoring to align with BFCL evaluation standards

3. **Improve Developer Experience**
   - Refactor for per-task orchestration
   - Enhance CLI workflow
   - Create comprehensive documentation

### 5.2 Medium-Term Goals

1. **Expand Resource Types**
   - Add more specialized resources (e.g., web service interactions)
   - Implement VM resource for more complex environments
   - Create composition capabilities for multiple resources

2. **Enhanced Metrics Collection**
   - Implement richer metrics for agent evaluation
   - Create visualization capabilities for evaluation results
   - Support for comparative evaluation across models

3. **Testing Infrastructure**
   - Mock mode for testing without API keys
   - Test suites for custom resource types
   - Automated CI for task validation

## 6. Usage Example (Current State)

```bash
# Set environment variables
export MODEL_AGENT=openai/gpt-4.1-2025-04-14
export PYTHONPATH="references/verifiers:$PYTHONPATH"

# Run a BFCL task
reward-kit agent-eval-v2 --task-def evaluations/bfcl/tasks/multi_turn_base_0.yaml
```

## 7. Vision for Improved Developer Experience

The goal is to make agent evaluation simple and accessible:

```bash
# Create a new task
reward-kit create-task my_file_system_task --resource-type FileSystemResource

# Run the task with a specified model
reward-kit run-task my_file_system_task --model openai/gpt-4-turbo

# View results
reward-kit show-results my_file_system_task
```

Each task should have its own Orchestrator instance and resource configuration, making development and testing straightforward.

---

This document represents the current state and future direction of the Agent Evaluation Framework. The focus on developer experience and ease of use will guide future improvements to enable researchers and developers to efficiently create, run, and analyze agent evaluations.
