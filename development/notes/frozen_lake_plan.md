# Frozen Lake Example Plan

This document outlines the plan for creating a Frozen Lake example using `reward-kit`.

### **Part 1: Standalone Frozen Lake Test Server**

The first step is to create a simple, standalone web server that wraps the Frozen Lake environment. This will allow us to test the game logic and environment interactions in isolation before integrating with the `reward-kit` ecosystem.

1.  **Create the Server File:**
    *   Create a new file named `frozen_lake_server.py` inside a new `examples/frozen_lake` directory.
2.  **Implement the Flask Server:**
    *   The server will use Flask to expose a few simple RESTful endpoints.
    *   It will import the `FrozenLakeEnv` from `gymnasium.envs.toy_text.frozen_lake`.
3.  **Define the API Endpoints:**
    *   `POST /reset`:
        *   Creates a new instance of the `FrozenLakeEnv`.
        *   Calls `env.reset()` to initialize the game.
        *   Stores the environment instance in a global dictionary, keyed by a unique `episode_id`.
        *   Returns the initial `observation` and the `episode_id`.
    *   `POST /step`:
        *   Accepts an `episode_id` and an `action` in the request body.
        *   Retrieves the correct environment instance from the global dictionary.
        *   Calls `env.step(action)` to advance the game.
        *   Returns the `observation`, `reward`, `done`, and `info` from the environment.
    *   `GET /render`:
        *   Accepts an `episode_id`.
        *   Calls `env.render(mode='ansi')` to get a text-based representation of the game state.
        *   Returns the rendered string.
4.  **Create a Test Client:**
    *   Create a separate script, `test_frozen_lake_server.py`, to act as a client.
    *   This script will make requests to the server's endpoints to ensure everything is working as expected. It will:
        *   Call `/reset` to start a game.
        *   Send a series of hardcoded actions to `/step`.
        *   Fetch the rendered state with `/render` after each step.

### **Part 2: HTTP Rollout Integration**

With the standalone server working, the next step is to create a client that interacts with it using the `reward-kit` HTTP rollout protocol. This will simulate an LLM playing the game.

1.  **Create the Rollout Client:**
    *   Create a new file, `frozen_lake_rollout_client.py`, in the `examples/frozen_lake` directory.
2.  **Implement the `RemoteHttpRolloutClient` API:**
    *   This client will implement the three key endpoints defined in `development/notes/http_rollout.md`:
        *   `POST /start_episode`: This will call the `/reset` endpoint of our `frozen_lake_server.py`.
        *   `POST /step`: This will take an action (initially, a random one), and call the `/step` endpoint of our `frozen_lake_server.py`.
        *   `POST /end_episode`: This will clean up the episode data.
3.  **Simulate the LLM:**
    *   Inside the `/step` endpoint implementation, add a function that generates a random, valid action for the Frozen Lake environment. This will stand in for the LLM's decision-making process.
4.  **Test the End-to-End Flow:**
    *   Write a test script that uses the `reward-kit` infrastructure to run a rollout against our `frozen_lake_rollout_client.py`. This will verify that the entire HTTP rollout process is working correctly.

### **Part 3: Evaluating an LLM on Frozen Lake**

The final step is to use the `reward-kit` evaluation pipeline to test how well a specific LLM can play the Frozen Lake game. The `reward-kit` CLI will act as the client, managing the conversation with the LLM and using our HTTP rollout server to interact with the environment.

1.  **Create the Reward Function:**
    *   Create a new file, `examples/frozen_lake/reward.py`.
    *   This function will be decorated with `@reward_function` and will analyze the final conversation history to determine if the agent succeeded. It will return an `EvaluateResult` with a score of 1.0 for success and 0.0 for failure.

2.  **Create the Initial Prompt Dataset:**
    *   Create a new file, `examples/frozen_lake/initial_prompt.jsonl`.
    *   This file will contain a single JSON object with a system message that instructs the LLM on its goal and how to use the available tools (e.g., `step(action: int)`).

3.  **Create the Evaluation Configuration (`config.yaml`):**
    *   Create a new file, `examples/frozen_lake/config.yaml`.
    *   This file will configure the `reward-kit` evaluation run:
        *   `dataset`: Point to the `initial_prompt.jsonl` file.
        *   `generation_config`: Specify the LLM to use (e.g., `accounts/fireworks/models/qwen3-235b-a22b`) and the necessary API credentials.
        *   `rollout_client`: Configure the connection to our `http_rollout_server.py` using the `remote_http` type.
        *   `reward_function`: Point to the `frozen_lake_reward` function.

4.  **Create a Final Runner Script:**
    *   Create a new shell script, `examples/frozen_lake/run_evaluation.sh`.
    *   This script will:
        1.  Start the `frozen_lake_server.py`.
        2.  Start the `http_rollout_server.py`.
        3.  Wait for both servers to be ready.
        4.  Execute the evaluation using the `reward-kit` CLI: `reward-kit --config-path examples/frozen_lake/config.yaml run-evaluation`.
        5.  Clean up and terminate the server processes.
