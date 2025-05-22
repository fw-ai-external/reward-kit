import atexit  # For cleanup
import os
import subprocess
import sys
import time

# Adjust the Python path to include the root of the reward-kit project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

try:
    # Import from subprocess_manager
    from development.utils.subprocess_manager import (
        start_process,
        start_serveo_and_get_url,
        stop_all_processes,
        stop_process,
    )
    from examples.remote_eval_demo.rewards.remote_validator_reward_hardcoded import (
        remote_validator_reward_hardcoded,
    )
    from examples.remote_eval_demo.rewards.remote_validator_reward_secure import (
        remote_validator_reward_secure,
    )
    from reward_kit.models import Message
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(
        "Please ensure that reward-kit is installed or the PYTHONPATH is set correctly."
    )
    print(f"Attempted to add project root: {project_root} to sys.path")
    sys.exit(1)

# --- Configuration for the Demo ---
MOCK_API_SERVICE_PORT = 8001  # Port for the local mock API service
LOG_DIR = os.path.join(project_root, "logs", "remote_eval_demo")
MOCK_API_LOG_FILE = os.path.join(LOG_DIR, "mock_api_service.log")
SERVEO_LOG_FILE = os.path.join(LOG_DIR, "serveo_ssh.log")
MOCK_API_SCRIPT_PATH = os.path.join(
    project_root, "examples", "remote_eval_demo", "mock_api_service.py"
)
API_KEY_GENERATOR_SCRIPT_PATH = os.path.join(
    project_root, "development", "utils", "generate_api_key.py"
)

# This will be dynamically fetched from Serveo
TUNNEL_PUBLIC_URL = None

# API key for the mock service. This will be generated.
MOCK_API_KEY_VALUE = None

# Environment variable name that the secure reward function will look for
SECURE_REWARD_API_KEY_ENV_VAR = "MOCK_SERVICE_API_KEY"


def generate_api_key() -> str | None:
    """Generates an API key using the utility script."""
    try:
        print(f"Generating a new API key using {API_KEY_GENERATOR_SCRIPT_PATH}...")
        result = subprocess.run(
            [sys.executable, API_KEY_GENERATOR_SCRIPT_PATH],
            capture_output=True,
            text=True,
            check=True,
        )
        key = result.stdout.strip()
        if not key:
            print("ERROR: API key generation script produced an empty key.")
            return None
        print(f"Generated API Key: {key}")
        return key
    except FileNotFoundError:
        print(
            f"ERROR: API key generator script not found at {API_KEY_GENERATOR_SCRIPT_PATH}"
        )
        return None
    except subprocess.CalledProcessError as e:
        print(f"ERROR: API key generation script failed: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return None


def print_header(title):
    print("\n" + "=" * 60)
    print(f"DEMO: {title}")
    print("=" * 60)


def run_evaluation(reward_fn_name, reward_fn, messages, **kwargs):
    print(f"\n--- Running: {reward_fn_name} ---")
    try:
        result = reward_fn(messages=messages, **kwargs)
        print(f"  Score: {result.score}")
        print(f"  Reason: {result.reason}")
        print(f"  Is Valid: {result.is_score_valid}")
        if result.metrics:
            for k, v in result.metrics.items():
                print(
                    f"  Metric '{k}': Score={v.score}, Valid={v.is_score_valid}, Reason={v.reason}"
                )
    except Exception as e:
        print(f"  ERROR during evaluation: {e}")
    print("--- End ---")


if __name__ == "__main__":
    atexit.register(stop_all_processes)  # Ensure cleanup on exit

    print_header("Automated Remote Evaluation Demo with Serveo.net")

    # Create log directory
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"Logs will be stored in: {LOG_DIR}")

    # 1. Generate API Key
    MOCK_API_KEY_VALUE = generate_api_key()
    if not MOCK_API_KEY_VALUE:
        print("Exiting demo due to API key generation failure.")
        sys.exit(1)

    # 2. Start Mock API Service
    print_header("Starting Mock API Service")
    mock_api_command = [sys.executable, MOCK_API_SCRIPT_PATH]
    # Pass the generated API key to the mock service via an environment variable
    mock_api_env = os.environ.copy()
    mock_api_env["EXPECTED_API_KEY"] = MOCK_API_KEY_VALUE
    mock_api_env["PORT"] = str(
        MOCK_API_SERVICE_PORT
    )  # Ensure mock service uses the configured port

    # Need to modify mock_api_service.py to accept EXPECTED_API_KEY and PORT from env
    # For now, assuming mock_api_service.py is adapted or uses a default key if env var not found.
    # The plan implies mock_api_service.py stores the "expected" API key.
    # For a fully automated demo, it's better if mock_api_service.py reads this from an env var
    # that this script sets.

    # For now, let's assume mock_api_service.py has been updated to use EXPECTED_API_KEY from env.
    # If not, the hardcoded key in mock_api_service.py must match the generated one, which is not feasible.
    # The plan says: "Store the "expected" API key (e.g., from Step 1, or a consistent demo key)."
    # This implies the mock service needs to know the key.

    # We will proceed by setting the EXPECTED_API_KEY env var for the mock service.
    # The mock_api_service.py will need to be checked/updated to use this.

    mock_api_process = start_process(
        mock_api_command, MOCK_API_LOG_FILE, cwd=project_root, env=mock_api_env
    )
    if not mock_api_process or mock_api_process.poll() is not None:
        print(
            f"ERROR: Failed to start Mock API Service. Check log: {MOCK_API_LOG_FILE}"
        )
        sys.exit(1)
    print(
        f"Mock API Service started with PID {mock_api_process.pid}. Waiting for it to initialize..."
    )
    time.sleep(5)  # Give the server a moment to start

    # 3. Start Serveo Tunnel
    print_header("Starting Serveo.net Tunnel")
    serveo_process, TUNNEL_PUBLIC_URL = start_serveo_and_get_url(
        MOCK_API_SERVICE_PORT, SERVEO_LOG_FILE
    )
    if not TUNNEL_PUBLIC_URL or not serveo_process:
        print("ERROR: Failed to start Serveo tunnel or get public URL.")
        # stop_all_processes() will be called by atexit, which will stop the mock_api_process if it started.
        sys.exit(1)

    print(f"Serveo Tunnel established: {TUNNEL_PUBLIC_URL}")
    print(f"Serveo SSH client PID: {serveo_process.pid}")
    print(f"Using Mock API Key for tests: {MOCK_API_KEY_VALUE}")
    print(
        f"Secure reward function will look for env var: {SECURE_REWARD_API_KEY_ENV_VAR}"
    )

    sample_messages = [
        Message(role="user", content="Test query for remote validation."),
        Message(
            role="assistant", content="Assistant response to be validated remotely."
        ),
    ]

    # --- Test 1: Hardcoded Reward Function (Correct Key) ---
    print_header("Test 1: Hardcoded Reward Function (Correct Key)")
    run_evaluation(
        "remote_validator_reward_hardcoded (correct key)",
        remote_validator_reward_hardcoded,
        sample_messages,
        target_service_url=TUNNEL_PUBLIC_URL,
        target_api_key=MOCK_API_KEY_VALUE,
    )

    # --- Test 2: Hardcoded Reward Function (Incorrect Key) ---
    print_header("Test 2: Hardcoded Reward Function (Incorrect Key)")
    run_evaluation(
        "remote_validator_reward_hardcoded (incorrect key)",
        remote_validator_reward_hardcoded,
        sample_messages,
        target_service_url=TUNNEL_PUBLIC_URL,
        target_api_key="THIS_IS_A_WRONG_KEY",
    )

    # --- Test 3: Secure Reward Function (Correct Key via Env Var) ---
    print_header("Test 3: Secure Reward Function (Correct Key via Env Var)")
    print(
        f"Setting environment variable for secure test: {SECURE_REWARD_API_KEY_ENV_VAR}={MOCK_API_KEY_VALUE}"
    )
    os.environ[SECURE_REWARD_API_KEY_ENV_VAR] = MOCK_API_KEY_VALUE

    run_evaluation(
        "remote_validator_reward_secure (correct key from env)",
        remote_validator_reward_secure,
        sample_messages,
        target_service_url=TUNNEL_PUBLIC_URL,
    )
    del os.environ[SECURE_REWARD_API_KEY_ENV_VAR]
    print(f"Cleared environment variable: {SECURE_REWARD_API_KEY_ENV_VAR}")

    # --- Test 4: Secure Reward Function (API Key Env Var Not Set) ---
    print_header("Test 4: Secure Reward Function (API Key Env Var Not Set)")
    if os.getenv(SECURE_REWARD_API_KEY_ENV_VAR):
        print(
            f"Warning: {SECURE_REWARD_API_KEY_ENV_VAR} was unexpectedly set. Deleting for test."
        )
        del os.environ[SECURE_REWARD_API_KEY_ENV_VAR]

    run_evaluation(
        "remote_validator_reward_secure (API key env var not set)",
        remote_validator_reward_secure,
        sample_messages,
        target_service_url=TUNNEL_PUBLIC_URL,
    )

    print("\n" + "=" * 60)
    print("DEMO SCRIPT FINISHED")
    print(
        "Review the outputs above. Mock API and Serveo tunnel were managed automatically."
    )
    print("=" * 60)
