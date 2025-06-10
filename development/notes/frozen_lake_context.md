# Frozen Lake Implementation Context

This document provides context for the existing implementation of the Frozen Lake MCP agent.

## Current Implementation Status

### Completed Components

1. **MCP Server Configuration**: Created `mcp_agent_config.yaml` that defines the Frozen Lake game backend
2. **Frozen Lake Server**: Implemented `frozen_lake_server.py` that runs the game and exposes movement actions as tools
3. **Reward Function**: Created a reward function that evaluates whether the agent successfully reached the goal
4. **Test Script**: Set up `run_frozen_lake_rl_example_test.sh` to run the example end-to-end
5. **Simulation Mode**: Added a fallback simulation mode for testing when the MCP server is not responding

### Components In Progress

1. **MCP Server Connection**: The direct connection to the MCP server is not working reliably yet
2. **LLM Integration**: Need to integrate with a real LLM to generate the navigation decisions
3. **Test Cases**: Need to add more test cases to verify different scenarios (success, falling in holes, etc.)

## Setup Instructions

1. **Requirements**:
   - reward-kit repository
   - Python 3.10+ with virtual environment
   - Gymnasium package
   - Docker (for running the MCP server)

2. **File Structure**:
   - `/examples/mcp_agent_frozen_lake/`: Main directory for the Frozen Lake example
   - `/examples/mcp_agent_frozen_lake/frozen_lake_server.py`: MCP server implementation of the Frozen Lake game
   - `/examples/mcp_agent_frozen_lake/reward.py`: Reward function for evaluating the agent's performance
   - `/examples/mcp_agent_frozen_lake/conf/config.yaml`: Configuration for running the evaluation
   - `/examples/mcp_agent_frozen_lake/dummy_input.jsonl`: Example input for testing
   - `/run_frozen_lake_rl_example_test.sh`: Script to run the example end-to-end

3. **Running the Example**:
   ```bash
   bash run_frozen_lake_rl_example_test.sh
   ```

## Implementation Details

### Frozen Lake Environment
The Frozen Lake environment is a simple grid world where the agent must navigate from a starting tile (S) to a goal tile (G) without falling into any holes (H). The agent can move on frozen tiles (F).

The environment's behavior is significantly affected by the `is_slippery` parameter. When `is_slippery` is `True`, the agent only has a 1/3 probability of moving in the intended direction. The other 2/3 of the time, it will move in a perpendicular direction. This introduces a significant element of stochasticity to the environment.

The standard 4x4 map is defined as:
```
S F F F
F H F H
F F F H
H F F G
```
The environment can also be initialized with an 8x8 map or a randomly generated map.

### MCP Server Implementation
The MCP server implementation in `frozen_lake_server.py` wraps the Gymnasium `FrozenLakeEnv` and exposes the game's action space as a set of tools. The key tools are:
- `move_left`, `move_down`, `move_right`, `move_up`: These correspond to the four actions in the environment's action space.
- `get_initial_state`: This resets the environment and returns the initial observation.

The `step` function in the environment returns a tuple of `(observation, reward, terminated, truncated, info)`. The MCP server's tool implementations should handle these return values appropriately, returning the new state to the agent and indicating whether the episode has ended.

### Reward Function and `reward-kit` Integration
The reward function in `reward.py` is designed to be used with the `reward-kit` framework. It inspects the conversation history to determine if the agent reached the goal, returning a score of `1.0` for success and `0.0` otherwise.

This reward function is hosted by a `RewardServer` (from `reward_kit/server.py`), which exposes it as an HTTP endpoint. The `RLDataAligner` (from `reward_kit/rl_processing.py`) then aligns the rewards from this server with the step-by-step data from the environment rollouts. This is a crucial step in preparing the data for the reinforcement learning algorithm.

### Test Script
The test script `run_frozen_lake_rl_example_test.sh`:
1. Starts the MCP server
2. Runs the evaluation using the reward-kit CLI
3. Evaluates the results using the reward function
4. Shuts down the server

### Simulated Interaction
When direct interaction with the MCP server is not possible, the code falls back to a simulated interaction that follows a successful path through the lake:
1. Start at position S
2. Move right, down, right, down, right, down to reach G
3. Each move generates a state update in the conversation

## Next Steps
To complete the implementation:

1. **Fix MCP Server Connection**:
   - Debug the connection issues with the MCP server
   - Ensure proper request handling and response formatting

2. **LLM Integration**:
   - Integrate with a real LLM to generate navigation decisions
   - Add proper prompt templates to guide the LLM's decision making
   - Check the trajectory to make sure we are making proper tool calls in OpenAI format

3. **Improve Evaluation**:
   - Add more sophisticated metrics for evaluating performance
   - Implement different reward signals for different outcomes

4. **Documentation**:
   - Complete full documentation with examples
   - Add diagrams explaining the interaction flow

5. **Extensibility**:
   - Show how to modify for different grid layouts
   - Demonstrate how to add new features like variable slipperiness

## Code Examples

### MCP Server (frozen_lake_server.py)
```python
import gymnasium as gym
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("frozen_lake")

# Initialize the Frozen Lake environment
env = gym.make("FrozenLake-v1", is_slippery=False)
env.reset()

def get_state() -> str:
    """Get the current state of the game."""
    return env.render()

@mcp.tool()
async def get_initial_state() -> str:
    """Get the initial state of the game."""
    env.reset()
    return get_state()

@mcp.tool()
async def move_left() -> str:
    """Move the player left."""
    state, reward, terminated, truncated, info = env.step(0)
    if terminated:
        if reward == 1.0:
            return "You reached the goal! 🎉"
        else:
            return "You fell into a hole! 😭"
    return get_state()

# Additional movement tools: move_down, move_right, move_up...

if __name__ == "__main__":
    mcp.run(transport='stdio')
```

### Reward Function (reward.py)
```python
from typing import List, Dict, Any, Optional, Union
from reward_kit.reward_function import reward_function
from reward_kit.models import EvaluateResult, MetricResult

@reward_function
def frozen_lake_reward(
    messages: Union[List[Dict[str, Any]], List[str]],
    ground_truth: Optional[str] = None,
    **kwargs: Any
) -> EvaluateResult:
    """
    A simple reward function for the Frozen Lake game.
    """
    score = 0.0
    reason = "The agent did not reach the goal."
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content", "")
            if "You reached the goal!" in content:
                score = 1.0
                reason = "The agent successfully reached the goal."
                break
    return EvaluateResult(
        score=score,
        reason=reason,
        metrics={
            "success": MetricResult(
                score=score,
                success=(score > 0.5),
                reason=reason
            )
        }
    )
```

## Original Frozen Lake Environment Documentation

Action Space

Discrete(4)

Observation Space

Discrete(16)

import

gymnasium.make("FrozenLake-v1")

Frozen lake involves crossing a frozen lake from start to goal without falling into any holes by walking over the frozen lake. The player may not always move in the intended direction due to the slippery nature of the frozen lake.

Description
The game starts with the player at location [0,0] of the frozen lake grid world with the goal located at far extent of the world e.g. [3,3] for the 4x4 environment.

Holes in the ice are distributed in set locations when using a pre-determined map or in random locations when a random map is generated.

The player makes moves until they reach the goal or fall in a hole.

The lake is slippery (unless disabled) so the player may move perpendicular to the intended direction sometimes (see is_slippery).

Randomly generated worlds will always have a path to the goal.

Elf and stool from https://franuka.itch.io/rpg-snow-tileset. All other assets by Mel Tillery http://www.cyaneus.com/.

Action Space
The action shape is (1,) in the range {0, 3} indicating which direction to move the player.

0: Move left

1: Move down

2: Move right

3: Move up

Observation Space
The observation is a value representing the player’s current position as current_row * ncols + current_col (where both the row and col start at 0).

For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15. The number of possible observations is dependent on the size of the map.

The observation is returned as an int().

Starting State
The episode starts with the player in state [0] (location [0, 0]).

Rewards
Reward schedule:

Reach goal: +1

Reach hole: 0

Reach frozen: 0

Episode End
The episode ends if the following happens:

Termination:

The player moves into a hole.

The player reaches the goal at max(nrow) * max(ncol) - 1 (location [max(nrow)-1, max(ncol)-1]).

Truncation (when using the time_limit wrapper):

The length of the episode is 100 for 4x4 environment, 200 for FrozenLake8x8-v1 environment.

Information
step() and reset() return a dict with the following keys:

p - transition probability for the state.

See is_slippery for transition probability information.

Arguments
import gymnasium as gym
gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
desc=None: Used to specify maps non-preloaded maps.

Specify a custom map.

    desc=["SFFF", "FHFH", "FFFH", "HFFG"].
The tile letters denote:

“S” for Start tile

“G” for Goal tile

“F” for frozen tile

“H” for a tile with a hole

A random generated map can be specified by calling the function generate_random_map.

from gymnasium.envs.toy_text.frozen_lake import generate_random_map

gym.make('FrozenLake-v1', desc=generate_random_map(size=8))
map_name="4x4": ID to use any of the preloaded maps.

    "4x4":[
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
        ]

    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ]
If desc=None then map_name will be used. If both desc and map_name are None a random 8x8 map with 80% of locations frozen will be generated.

is_slippery=True: If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions.

For example, if action is left and is_slippery is True, then:

P(move left)=1/3

P(move up)=1/3

P(move down)=1/3

Version History
v1: Bug fixes to rewards

v0: Initial version release
