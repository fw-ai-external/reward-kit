# Remote Evaluation with Secrets and Serveo.net Demo

This demo showcases how reward functions in `reward-kit` can securely call remote APIs. It uses **Serveo.net** (via SSH) to expose a local mock API service to the internet and manages secrets (API keys) through environment variables (simulating `fireworks secret` behavior). This setup prioritizes minimal external dependencies, relying on a standard SSH client.

## Prerequisites

1.  **Python Environment**: Ensure you have a Python environment with `reward-kit` and its development dependencies installed. This typically includes:
    *   `fastapi`
    *   `uvicorn`
    *   `requests` (though not directly used by Serveo, it might be a general dev dependency)
2.  **SSH Client**:
    *   A working **SSH client must be installed and accessible in your system's PATH.** The demo script (`run_demo.py`) uses the `ssh` command to establish a tunnel with Serveo.net.
    *   Most Linux and macOS systems have an SSH client pre-installed. Windows users might need to install one (e.g., OpenSSH included in modern Windows, or Git Bash).
    *   Verify by typing `ssh -V` in your terminal.

## Running the Demo

The easiest way to run the full automated demo is using the Makefile target from the root of the `reward-kit` repository:

```bash
make demo-remote-eval
```

This command will:
1.  Generate a temporary API key for the mock service for this session.
2.  Start a local mock FastAPI server (defined in `mock_api_service.py`). This server will expect the generated API key.
3.  Attempt to start an SSH tunnel using **Serveo.net** to expose the local mock server (running on port 8001 by default) to the internet.
4.  Automatically fetch the public HTTPS URL provided by Serveo.net.
5.  Run several test evaluations using this URL:
    *   Using a reward function with a hardcoded API key (for demonstration of an insecure pattern, though it will use the session's generated key for the actual call).
    *   Using a reward function that securely retrieves the API key from an environment variable.
6.  Automatically stop the mock API server and the Serveo.net SSH tunnel when the script finishes or is interrupted.

Log files for the mock API server and the Serveo SSH client output will be created in the `logs/remote_eval_demo/` directory at the root of the repository (e.g., `mock_api_service.log` and `serveo_ssh.log`).

## How it Works

*   **`run_demo.py`**: The main script that orchestrates the demo.
    *   It generates a unique API key for the session using `generate_api_key.py`.
    *   It starts `mock_api_service.py` as a background process, passing the generated API key and configured port via environment variables (`EXPECTED_API_KEY`, `PORT`).
    *   It uses `development/utils/subprocess_manager.py` to manage these background processes and to start the Serveo.net tunnel.
*   **`development/utils/subprocess_manager.py`**: A utility to start, stop, and manage background processes.
    *   Includes `start_serveo_and_get_url()`: This function executes an `ssh` command to connect to Serveo.net, requests a tunnel, and parses the output to retrieve the public URL.
    *   It handles automatic cleanup of all started processes.
*   **`mock_api_service.py`**: A simple FastAPI application that simulates an external service.
    *   It expects an `X-Secret-API-Key` header for its `/api/validate_data` endpoint.
    *   It reads the `EXPECTED_API_KEY` and `PORT` from environment variables set by `run_demo.py`.
*   **`rewards/remote_validator_reward_hardcoded.py`**: A reward function demonstrating an insecure way to handle an API key by having a placeholder for a hardcoded key. In the demo, `run_demo.py` overrides this by passing the `target_api_key`.
*   **`rewards/remote_validator_reward_secure.py`**: A reward function demonstrating a secure way to handle an API key by fetching it from the `MOCK_SERVICE_API_KEY` environment variable (which `run_demo.py` sets for the test, using the dynamically generated API key).
*   **`development/utils/generate_api_key.py`**: A utility to generate secure random API keys. `run_demo.py` calls this to create a unique key for each demo run.

## Troubleshooting

*   **"ERROR: 'ssh' command not found..."**:
    *   Ensure an SSH client is installed and its location is added to your system's PATH.
    *   Verify by typing `ssh -V` (or `ssh --version`) in your terminal.
*   **"ERROR: Timeout (...) waiting for Serveo URL."** or **"ERROR: Could not retrieve public URL from Serveo.net."**:
    *   Check the Serveo SSH client log file (e.g., `logs/remote_eval_demo/serveo_ssh.log`).
    *   Serveo.net is a public service; its availability or performance can sometimes vary. Try running the demo again after a short wait.
    *   Ensure your internet connection is stable and allows outbound SSH connections on port 22 (or the port Serveo.net uses if it differs).
    *   Firewall or network policies might be blocking the SSH connection to `serveo.net`.
*   **"ERROR: Serveo SSH process terminated unexpectedly"**:
    *   Check the `serveo_ssh.log`. This could be due to network issues, Serveo.net service interruptions, or issues with the local SSH client setup.
*   **API Key Mismatches / Authentication Failures in Reward Function Tests**:
    *   The `run_demo.py` script generates a new API key for each session and configures the `mock_api_service.py` to expect this specific key.
    *   The reward functions are then called with this generated key (either directly for the "hardcoded" example's override, or via an environment variable for the "secure" example).
    *   If you see authentication failures, check the logs from `run_demo.py` to see the generated API key and ensure the mock service logs indicate it's expecting the same. The `X-Secret-API-Key` header sent by the reward functions must match what the mock service expects.
*   **Port Conflicts**: The demo uses port `8001` by default for the local mock API service. If this port is already in use, `mock_api_service.py` might fail to start. The `run_demo.py` script sets this via the `MOCK_API_SERVICE_PORT` variable, which is then passed as a `PORT` environment variable to the mock service.
