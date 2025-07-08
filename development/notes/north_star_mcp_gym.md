# Democratizing RL Environments: Fireworks' Reward Protocol & MCP-Gym

### The RL Environment Fragmentation Crisis

The LLM community's advances in open-sourcing models and SFT datasets stands in stark contrast to reinforcement learning's persistent bottleneck: **environmental fragmentation**. This isn't due to lacking innovation – high-quality RL environments across different tasks – but because they operate in disconnected silos:

- **Integration Burden:** 70% of RL effort is wasted adapting environments to training stacks
- **Benchmark Inconsistency:** No standardized interfaces = no comparable results
- **Production-Reality Chasm:** Training environments diverge from live MCP deployments, where most the application developers live
- **Capability Lock-In:** Researchers rebuild foundational plumbing instead of advancing intelligence

**This is RL's "integration tax"**

### Creating RL's Universal Adapter

Fireworks is bridging this divide with **MCP-Gym** – implementing our open **Reward Protocol** to converge:

- Training Infrastructure
- Production MCP Standards
- High-Quality Environments

Like PyTorch abstracted GPU programming, MCP-Gym creates universal compatibility between your:

1. Specialized environments (Gymnasium/web/computer use/customer support agents)
2. Production systems (existing MCP toolchains)
3. RL training stacks (Fireworks RFT + community tools)

### Our Solution: Reward Protocol via MCP-Gym

Fireworks introduces a universal abstraction layer for RL environments using **MCP-Gym** - an implementation of our open **Reward Protocol** that standardizes agent-environment interaction like PyTorch standardized GPU programming.

**How MCP-Gym Works:**

```mermaid
**sequenceDiagram
    participant Agent
    participant mcp-gym as MCP-Gym (Environment)
    participant Protocol as MCP-Gym (Reward Protocol)

    Note over Agent,mcp-gym: DATA PLANE (States/Actions)
    Agent->>mcp-gym: JSON Tool Call (state + action)
    mcp-gym-->>Agent: JSON Tool Response (new state)

    Note over mcp-gym,Protocol: CONTROL PLANE (Rewards/Termination)
    Agent-->>Protocol: Update Environment State
    Protocol-->>Agent: Reward Signal**

```

1. **Strict Plane Separation**
    - **Data Plane:** JSON tool calls/responses via MCP (state transitions/actions)
    - **Control Plane:** Rewards/termination signals via persistent SSE channels
2. **Environment Implementation**
    - Single-process MCP server per environment
    - State encoded in initial prompt (`environment_description` + `user_intent`)
    - Tool calls modify environment state
    - Optional reward functions trigger control plane signals
3. **Fireworks Infrastructure**
    - Isolated sessions via Docker/Conda
    - Rollouts: OpenAI JSONL format with rewards
    - Training: reinforcement learning with verifiable rewards with trajectory optimization
    - Hosted environments in Fireworks Cloud

### Why This Matters

1. **For Model Builders**

    Train once → deploy anywhere with MCP support

    No more adapter hell: Reuse environments across projects

    Benchmark agents consistently with hosted environments

2. **For Application Developers**

    Plug in environments without pipeline changes:


```python
# Switch between chip design/diagnostics/robotics
tools = [{"type": "sse", "server_url": "<http://mcp-gym/env/medical-lab>"}]

```

Production-to-simulation parity via identical interfaces

Share custom environments like Python packages

### Founders' Mandate: Democratization Via Standardization

> As PyTorch core contributors at Meta, Lin and Dmytro witnessed how standardization unlocks ecosystem progress. We're applying that same principle to RL environments:
>
> - **No walled gardens:** Apache 2.0 licensed protocol
> - **No captive audiences:** Host MCP-Gym anywhere
> - **Zero reinvention:** Sharable environments, reproducible benchmarks

Power corrupts when concentrated. Fragmented RL environments create artificial scarcity. True agent advancement requires foundations built together.

### Join the Open RL Movement

Fireworks enables today:

✅ **Seamless tool integration** via MCP

✅ **Production-aligned RL models**

✅ **Environment-agnostic training**

**Next frontier:** Universal MCP-Gym environments. We invite you to:

1. Contribute to the [Reward Protocol Specification](http://fireworks.ai/reward-protocol)
2. Host environments with MCP-Gym
3. Run benchmarks with hosted environments

The cost of rebuilding RL plumbing ends now. Let's create the abstraction layer agentic AI deserves.

[Build with RFT](https://fireworks.ai/) | [Create MCP Tools](https://fireworks.ai/docs) | [Join Our Discord](https://discord.gg/fireworks)

# Minimal code example

## Current Implementation Architecture

The MCP-Gym framework has been implemented with the following components:

### 1. Framework Base Class (`reward_kit/mcp/mcpgym.py`)

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, TypeVar, Generic
from fastmcp import Context
from reward_kit.mcp.base import GymProductionServer

EnvType = TypeVar('EnvType')

class McpGym(GymProductionServer, ABC, Generic[EnvType]):
    """
    Base class for MCP-Gym environments.
    Inherits from GymProductionServer which provides FastMCP infrastructure.
    """

    def __init__(self, name: str = "mcp-gym", version: str = "1.0.0"):
        super().__init__(name, version)
        self.env_adapter = None
        self.is_initialized = False

    @abstractmethod
    def create_adapter(self) -> 'EnvironmentAdapter[EnvType]':
        """Create the environment adapter for this MCP-Gym instance"""
        pass

    async def initialize_environment(self, seed: Optional[int] = None) -> None:
        """Initialize the environment with optional seed"""
        if not self.is_initialized:
            self.env_adapter = self.create_adapter()
            await self.env_adapter.reset(seed=seed)
            self.is_initialized = True
```

### 2. Environment Adapter Pattern (`examples/frozen_lake_mcp/frozen_lake_adapter.py`)

```python
from typing import Any, Dict, Optional, Tuple
from gymnasium.envs.toy_text import FrozenLakeEnv
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from reward_kit.mcp.adapters import EnvironmentAdapter

class FrozenLakeAdapter(EnvironmentAdapter[FrozenLakeEnv]):
    """Adapter for FrozenLake Gymnasium environment"""

    ACTION_NAMES = ["LEFT", "DOWN", "RIGHT", "UP"]

    def create_with_seed(self, seed: Optional[int] = None) -> Tuple[FrozenLakeEnv, int, Dict[str, Any]]:
        desc = generate_random_map(size=4, p=0.8, seed=seed)
        env = FrozenLakeEnv(desc=desc, is_slippery=False, render_mode="ansi")
        obs, info = env.reset(seed=seed)
        return env, obs, info

    def format_observation(self, obs: int, env: FrozenLakeEnv) -> Dict[str, Any]:
        return {
            "position": obs,
            "grid": env.render()
        }

    def step(self, env: FrozenLakeEnv, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        return env.step(action)

    def close(self, env: FrozenLakeEnv) -> None:
        env.close()
```

### 3. MCP Server Implementation (`examples/frozen_lake_mcp/frozen_lake_mcp.py`)

```python
from typing import Any, Dict
from fastmcp import Context
from reward_kit.mcp.mcpgym import McpGym
from .frozen_lake_adapter import FrozenLakeAdapter

class FrozenLakeMcp(McpGym):
    """FrozenLake MCP server implementation"""

    def create_adapter(self) -> FrozenLakeAdapter:
        return FrozenLakeAdapter()

    @self.mcp.tool(
        name="lake_move",
        description="Move on the frozen lake. Actions: LEFT, DOWN, RIGHT, UP"
    )
    def lake_move(self, action: str, ctx: Context) -> Dict[str, Any]:
        action_str = action.strip().upper()
        if action_str not in self.env_adapter.ACTION_NAMES:
            raise ValueError(f"Invalid action '{action_str}'. Valid: {self.env_adapter.ACTION_NAMES}")

        action_idx = self.env_adapter.ACTION_NAMES.index(action_str)
        obs, reward, terminated, truncated, info = self.env_adapter.step(self.env_adapter.env, action_idx)

        # Format the observation for the LLM
        formatted_obs = self.env_adapter.format_observation(obs, self.env_adapter.env)

        return {
            **formatted_obs,
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }
```

### 4. Server Launcher (`examples/frozen_lake_mcp/server.py`)

```python
import asyncio
from frozen_lake_mcp import FrozenLakeMcp

async def main():
    server = FrozenLakeMcp()
    await server.initialize_environment()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
```

### 5. System Integration

Environment descriptions and user intents are loaded from `shared_data/rollouts.jsonl`:

```json
{
  "system_prompt": "You are playing FrozenLake, a grid-based navigation game...",
  "user_intent": "Navigate safely to reach the goal 'G' while avoiding holes 'H'",
  "config": {"seed": 42, "size": 4, "p": 0.8}
}
```

## Usage

Start the server:
```bash
cd examples/frozen_lake_mcp
python server.py --port 9004
```

Run rollouts:
```python
import reward_kit as rk

# Connect to MCP server
envs = rk.connect_mcp_environments([
    {"url": "http://localhost:9004", "count": 2}
])

# Run rollouts with policy
rollouts = await rk.rollout(envs, policy, steps=20)
```

## Key Features Implemented

✅ **Proper MCP Server Architecture**: FastMCP inheritance chain
✅ **Environment Adapter Pattern**: Clean separation between gym env and MCP server
✅ **Tool Registration**: `@self.mcp.tool()` decorator system
✅ **System Prompt Loading**: Environment descriptions from `rollouts.jsonl`
✅ **Record/Replay System**: Complete trajectory recording and 1000x speedup playback
✅ **Production Server Integration**: Compatible with CondaServerProcessManager
✅ **Comprehensive Testing**: End-to-end testing with persistent trajectory storage
