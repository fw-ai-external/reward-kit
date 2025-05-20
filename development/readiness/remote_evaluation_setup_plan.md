# Plan: Remote Evaluation with Secrets (Automated Tunneling via Serveo.net)

This document outlines the plan to set up a demonstration for remote evaluation capabilities. The demo will focus on using environment variables and the `fireworks secret` mechanism for managing secrets required by evaluation functions that invoke remote URLs. For achieving a one-command, automated demo, **Serveo.net** will be used for exposing the local mock server, prioritizing minimal external dependencies (assuming a standard SSH client). Initial manual setup steps might still reference Ngrok or other tunneling tools as manual alternatives for developer testing.

## 1. Objectives

*   Demonstrate that evaluation functions can call arbitrary remote URLs.
*   Showcase secure secret management for API keys using `fireworks secret`.
*   Illustrate a less secure method (hardcoding) for comparison.
*   Provide a utility for API key generation.
*   Adapt an existing simple reward function (e.g., math reward) to fit this scenario for local testing.
*   **Key Goal for Automation:** Enable a one-command execution (e.g., `make demo-remote-eval`) that sets up all necessary components, including the public tunnel, using Serveo.net.

## 2. Components

### 2.1. Mock API Service (FastAPI) - **Status: DONE**
*   **Purpose:** Simulate an external service requiring API key authentication.
*   **Technology:** Python with FastAPI.
*   **Endpoint:** A simple endpoint (e.g., `/api/validate_data`).
    *   Accepts: A small JSON payload.
    *   Requires: An API key passed in a custom header (e.g., `X-Secret-API-Key`).
*   **Behavior:** Validates the API key. If valid, processes the payload and returns a success message. If invalid, returns an authentication error.

### 2.2. Tunneling Component (Targeting Serveo.net for Automation)
*   **Purpose:** Expose the local FastAPI service to the public internet, making it accessible as a remote URL.
*   **Automated Tool (Target for `make demo-remote-eval`):** Serveo.net (using SSH). - **Status: Implementation TODO (See Section 6)**
*   **Manual Alternatives (For developer testing of components):** Ngrok, Localtunnel can be used if preferred by the developer during initial component development, but the final automated demo script will integrate Serveo.net.
*   **Setup (Serveo.net):** Relies on a standard SSH client. The demo script will automate the `ssh -R ... serveo.net` command.

### 2.3. Evaluation Function (`remote_validator_reward`) - **Status: DONE**
*   **Purpose:** A Python function, decorated with `@reward_function`, that calls the tunneled FastAPI service.
*   **Logic:**
    1.  Accepts input parameters.
    2.  Retrieves the API key for the mock service.
    3.  Constructs and sends an HTTP request to the mock API's endpoint (via the tunnel URL).
    4.  Includes the API key in the `X-Secret-API-Key` header.
    5.  Evaluates the response from the mock API.
*   **Note:** Variable names for tunnel URL (e.g., `DEFAULT_NGROK_URL`) in existing reward functions may need generalization (e.g., to `DEFAULT_TUNNEL_URL`).

### 2.4. Secret Management Strategies - **Status: Implemented in Reward Functions**
*   **Strategy A: Hardcoded Secret** - **Status: DONE**
*   **Strategy B: `fireworks secret` Integration (Simulated via Env Var)** - **Status: DONE**

## 3. Detailed Implementation Steps

**Current Status of Overall Plan (as of YYYY-MM-DD - *developer to fill in date*):**
*   **Core Components (Steps 1, 2, 4, 5 below):** The API key generator, mock FastAPI service, and the two reward function variants (hardcoded and secure via env var) have been implemented. These components are designed to work with a generic tunnel URL.
*   **Automated Demo Script (`run_demo.py`, `subprocess_manager.py`, `Makefile`):** An initial version targeting Ngrok automation was developed. **This is now being redirected to use Serveo.net.** The existing automation code provides a base but requires significant modification/replacement for Serveo (see Section 6).
*   **Documentation (`README.md` for demo, this plan):** Initial versions exist, need updates to reflect the Serveo strategy.

The following steps describe the creation of the core components. The automation of these components into a one-command demo using Serveo.net is detailed in Section 6.

### Step 1: Develop Utility for API Key Generation - **Status: DONE**
*   Create `generate_api_key.py` in `development/utils/`.
*   Implement a function using `secrets.token_hex()` or `uuid.uuid4()`.
*   Provide a simple CLI interface.

### Step 2: Develop the FastAPI Mock Service - **Status: DONE**
*   Create `mock_api_service.py` in `examples/remote_eval_demo/`.
*   Implement the FastAPI app with the `/api/validate_data` endpoint.
*   Store the "expected" API key (e.g., from Step 1, or a consistent demo key).
*   Document how to run it manually (e.g., `python examples/remote_eval_demo/mock_api_service.py` or `uvicorn ...`).

### Step 3: Tunneling Setup and Configuration - **Status: Manual options documented; Automation via Serveo is TODO (See Section 6)**
*   **For Automated Demo (Target: Serveo.net):** This step will be handled automatically by the `run_demo.py` script using Serveo.net. No manual ngrok/localtunnel setup is required by the end-user running `make demo-remote-eval`. (See Section 6 for automation details).
*   **For Manual Component Testing (Developer-Only):**
    *   If a developer is testing components individually, they can manually use any tunneling tool:
        *   **Serveo.net (manual):** `ssh -R 80:localhost:8001 serveo.net` (copy the HTTPS URL).
        *   **Ngrok (manual):** Install ngrok, then `ngrok http 8001` (copy the HTTPS URL).
        *   **Localtunnel (manual):** `npm install -g localtunnel`, then `lt --port 8001` (copy the URL).
    *   The evaluation functions will need their tunnel URL constant updated with the manually obtained URL for such tests. The automated `run_demo.py` will pass this URL programmatically.

### Step 4: Implement Evaluation Function - Strategy A (Hardcoded Secret) - **Status: DONE**
*   Create `remote_validator_reward_hardcoded.py` in `examples/remote_eval_demo/rewards/`.
*   Implement the reward function, hardcoding the tunnel URL (e.g. `DEFAULT_TUNNEL_URL` - to be updated by `run_demo.py` or manually for tests) and API key.
*   **Note:** Existing file uses `HARDCODED_NGROK_URL`; consider renaming for generality.

### Step 5: Implement Evaluation Function - Strategy B (`fireworks secret` via Env Var) - **Status: DONE**
*   Create `remote_validator_reward_secure.py` in `examples/remote_eval_demo/rewards/`.
*   Modify the function to retrieve the API key using `os.getenv("MOCK_SERVICE_API_KEY")`.
*   Tunnel URL handled similarly to Strategy A.
*   **Note:** Existing file uses `DEFAULT_NGROK_URL`; consider renaming.

### Step 6: Documentation and Demo Script (Initial Local Runner) - **Status: Partially DONE (`run_demo.py` needs Serveo integration, `README.md` needs Serveo focus)**
*   Update `development/DEMO_READINESS.md` with a section for this demo. - **Status: DONE** (Link added)
*   The `run_demo.py` script (for local execution) has been created. It currently includes logic for Ngrok automation which needs to be **replaced with Serveo automation** (see Section 6).
*   The final user-facing documentation in `examples/remote_eval_demo/README.md` needs to be **updated for Serveo**.

## 4. Files to Create/Modify (Reflecting Serveo Automation Goal)

*   `development/readiness/remote_evaluation_setup_plan.md` (This file) - **Status: BEING UPDATED NOW**
*   `development/utils/generate_api_key.py` - **Status: DONE**
*   `examples/remote_eval_demo/mock_api_service.py` - **Status: DONE**
*   `examples/remote_eval_demo/rewards/remote_validator_reward_hardcoded.py` - **Status: DONE** (Consider renaming URL const)
*   `examples/remote_eval_demo/rewards/remote_validator_reward_secure.py` - **Status: DONE** (Consider renaming URL const)
*   `examples/remote_eval_demo/run_demo.py` - **Status: Partially DONE (Ngrok base), NEEDS SERVEO IMPL.**
*   `development/utils/subprocess_manager.py` - **Status: Partially DONE (Ngrok base), NEEDS SERVEO IMPL. & Ngrok code removal/deprecation.**
*   `Makefile` (Update) - **Status: Partially DONE (Ngrok base), NEEDS SERVEO IMPL.**
*   `examples/remote_eval_demo/README.md` (New) - **Status: Partially DONE (Ngrok base), NEEDS SERVEO IMPL.**
*   `development/DEMO_READINESS.md` (Update) - **Status: DONE** (Link added)

## 5. Considerations (General)

*   **Error Handling:** Reward functions and demo scripts should handle network errors, API errors, missing secrets, and tunnel failures gracefully. - **Status: Basic handling in place, review for robustness.**
*   **Idempotency:** Mock API is read-only for the demo endpoint. - **Status: OK**
*   **Clarity:** Demo should distinguish local execution vs. platform. - **Status: OK**
*   **Reward Kit Version:** Ensure compatibility. - **Status: OK**

## 6. Automating Tunneling for One-Command Demo using Serveo.net

**Overall Goal:** Achieve a true one-command demo (e.g., via a `make` target) by automating the use of **Serveo.net** for exposing the local mock server. This choice prioritizes minimizing additional software installations for the user, assuming a standard SSH client is available.

**Current Status of this Section (as of YYYY-MM-DD - *developer to fill in date*):**
*   **Previous Ngrok Automation Attempt (To be Replaced):**
    *   An initial attempt was made to automate tunneling using Ngrok. This involved changes to `development/utils/subprocess_manager.py`, `examples/remote_eval_demo/run_demo.py`, and `Makefile`.
    *   **Decision:** This Ngrok-based automation will be **replaced** by a Serveo.net implementation to better meet the goal of minimal external dependencies. The existing Ngrok-specific code in `subprocess_manager.py` (`start_ngrok_and_get_url`, `get_ngrok_public_url`) should be removed or clearly marked as deprecated/archived if kept for reference.
*   **Serveo.net Implementation (Not Started - THIS IS THE NEXT STEP):**
    *   The specific code changes for Serveo.net automation in `subprocess_manager.py` and `run_demo.py` have **not** been implemented. This is the primary task for the next developer.

This section outlines the implementation path for using Serveo.net.

### 6.1. Implementation Path for Serveo.net

#### 6.1.1. Key Characteristics of Serveo.net
*   **Pros:** Uses SSH, so no *additional* client software installation is typically needed if an SSH client is already present on the user's system. No registration is required for basic public tunnel use.
*   **Cons:** The reliability of the public `serveo.net` service can vary. Automating SSH interactions (especially initial host key verification and parsing output for the URL) from a script can be more complex than tools with dedicated local APIs (like Ngrok).
*   **Command Example:** `ssh -R 80:localhost:<local_port> serveo.net`

#### 6.1.2. Required Changes in `development/utils/subprocess_manager.py` - **Status: TODO**
*   **Task:** Create a new function `start_serveo_and_get_url(local_port: int, log_file_path: str) -> tuple[subprocess.Popen | None, str | None]:`
    1.  **Check SSH Availability:** Verify that the `ssh` command is available on the system's PATH. If not, raise an error or return `(None, None)` with a clear message.
    2.  **Construct SSH Command:**
        *   The command should be similar to: `['ssh', '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null', '-R', f'80:localhost:{local_port}', 'serveo.net']`.
        *   The options `-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null` are crucial for non-interactive execution. Note: `/dev/null` might need platform-specific handling (e.g., `nul` on Windows) or use of a temporary file.
    3.  **Execute SSH Command:** Use `start_process` or `subprocess.Popen` directly, ensuring output (stdout/stderr) is captured for parsing.
    4.  **Parse Output for URL:** Continuously read output, looking for `Forwarding HTTP traffic from https://<subdomain>.serveo.net`. Extract URL using regex.
    5.  **Retry Logic & Timeout:** Implement a loop (e.g., 10-15 seconds) to find the URL. If not found, terminate SSH and return `(None, None)`.
    6.  **Process Management:** Ensure the started `ssh` process is managed for cleanup.
    7.  **Return Value:** `(ssh_process_object, public_url)` or `(None, None)`.
*   **Task:** Remove or clearly deprecate existing Ngrok-specific functions (`start_ngrok_and_get_url`, `get_ngrok_public_url`).

#### 6.1.3. Required Changes in `examples/remote_eval_demo/run_demo.py` - **Status: TODO**
*   **Task:** Modify the script to use Serveo automation:
    1.  Import `start_serveo_and_get_url`.
    2.  Remove Ngrok-related calls and logic.
    3.  Call `start_serveo_and_get_url`, handle success/failure.
    4.  Assign fetched URL to a generic variable (e.g., `TUNNEL_PUBLIC_URL`) and pass to reward functions.
    5.  Remove manual ngrok prompts.
    6.  Ensure `atexit` handler stops the Serveo SSH process.

#### 6.1.4. Required Changes in `Makefile` - **Status: TODO**
*   **Task:** Update `demo-remote-eval` target description:
    *   State it uses Serveo.net.
    *   Prerequisite: "Working SSH client in PATH."
    *   Remove Ngrok-specific notes.

#### 6.1.5. Required Changes in `examples/remote_eval_demo/README.md` - **Status: TODO**
*   **Task:** Overhaul README for Serveo:
    1.  Update title, introduction.
    2.  Prerequisites: Focus on SSH client. Remove Ngrok.
    3.  "Running the Demo", "How it Works": Describe Serveo-based process.
    4.  "Troubleshooting": Address common Serveo/SSH issues.

### 6.2. Testing Focus for Serveo Implementation - **Status: TODO**
*   **Cross-Platform:** Test SSH command and parsing on macOS, Linux, Windows.
*   **First-Time Connection:** Verify non-interactive host key handling.
*   **URL Parsing:** Ensure regex robustness.
*   **Process Cleanup:** Confirm reliable termination of `ssh` tunnel.
*   **Serveo Service Reliability:** Test with potential service flakiness in mind.

This revised plan provides a more direct path for the next engineer to implement the Serveo.net solution for the one-command demo.
