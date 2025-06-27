# FrozenLake via MCP (Model Context Protocol)

This example demonstrates how to serve a Gymnasium environment (`FrozenLake-v1`) over the Model Context Protocol (MCP) and interact with it using a rollout client.

It contains the definitive, working patterns for building and interacting with MCP servers, which were used to implement the `rk.make()` and `rk.rollout()` APIs.

## Key Files

1.  `frozen_lake_mcp_server.py` (in `mcp_server/`)
    *   **Purpose**: A stateless, production-ready MCP server using `FastMCP`.
    *   **Architecture**: Uses the `stateless_http=True` pattern, which is the recommended approach for production shims. It manages a single global environment state, automatically resetting on game completion.
    *   **Key Feature**: Demonstrates the correct, minimalist approach to exposing a game or environment as an MCP server.

2.  `mcp_rollout_client.py` (in `local_testing/`)
    *   **Purpose**: A client script to run episodes ("rollouts") against the MCP server.
    *   **Architecture**: Implements the **official MCP Python SDK pattern** for establishing a connection using `streamablehttp_client` and `ClientSession`.
    *   **Key Feature**: This is the canonical example of how to correctly connect to and interact with any MCP server. It was the key to resolving all connection issues.

3.  `frozen_lake_adapter.py` (in `mcp_server/`)
    *   **Purpose**: The adapter that bridges the `FrozenLakeEnv` with the `reward-kit`'s generic MCP simulation server framework. It translates MCP calls into environment-specific actions.

4.  `simulation_server.py` (in `mcp_server/`)
    *   **Purpose**: A stateful simulation server that can manage multiple, independent, seeded environment instances simultaneously.
    *   **Architecture**: Uses the `SimulationServerBase` framework, which handles all session management internally and validates its tool signatures against the production server to prevent drift.

## How to Run the Example

### 1. Start the Production Server

In your terminal, run the production server:

```bash
cd examples/frozen_lake_mcp_complete/mcp_server
.venv/bin/python frozen_lake_mcp_server.py
```

The server will start and listen on `http://localhost:8000/mcp`.

### 2. Run the Rollout Client

In a second terminal, run the client to execute a single episode or a batch of episodes against the server.

**Run a single episode (with a specific seed):**
```bash
cd examples/frozen_lake_mcp_complete/local_testing
.venv/bin/python mcp_rollout_client.py --test single --seed 42
```
This will run one episode with seed 42, showing each step.

**Run a batch of episodes:**
```bash
.venv/bin/python mcp_rollout_client.py --test batch --count 10
```
This will run 10 episodes and print summary statistics, such as success rate and average steps.

## North Star Integration

The patterns established in these files directly led to the successful implementation of the `reward-kit` north-star API:

```python
# The working client/server pattern here enabled this API
import reward_kit as rk

envs = rk.make("http://localhost:8000/mcp", n=10, seeds=range(10))
trajectories = await rk.rollout(envs, policy=MyPolicy(), steps=50)
```
