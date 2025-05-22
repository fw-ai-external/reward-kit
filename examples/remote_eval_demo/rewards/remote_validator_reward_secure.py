from typing import Dict, List, Any, Union, Optional
import requests
import os  # For accessing environment variables

# Assuming reward_kit is installed and these are the correct import paths
from reward_kit.reward_function import reward_function
from reward_kit.models import Message, EvaluateResult, MetricResult

# --- Configuration (Secure - via Environment Variable) ---
# IMPORTANT: This is a placeholder for a default URL if the tunnel URL isn't provided.
# The demo (run_demo.py) will override this with the actual tunnel URL via 'target_service_url' kwarg.
DEFAULT_TARGET_SERVICE_URL = "YOUR_TUNNEL_URL_HERE"
# The API key will be fetched from this environment variable.
# run_demo.py sets this environment variable before calling.
API_KEY_ENV_VAR = "MOCK_SERVICE_API_KEY"  # This is the var name the function looks for.


@reward_function
def remote_validator_reward_secure(
    messages: Union[List[Dict[str, Any]], List[Message]],
    ground_truth: Optional[str] = None,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Evaluates by calling a remote API using an API key
    retrieved from an environment variable (simulating `fireworks secret`).

    Args:
        messages: List of conversation messages. The last assistant message's content
                  is used as part of the payload if available.
        ground_truth: Optional expected correct answer.
        **kwargs: Additional arguments.
                  'target_service_url': Overrides DEFAULT_TARGET_SERVICE_URL if provided. Expected from run_demo.py.
                  'api_key_env_var': Overrides API_KEY_ENV_VAR if provided (less common for this secure version).

    Returns:
        EvaluateResult with evaluation score and metrics based on API response.
    """
    target_service_url = kwargs.get("target_service_url", DEFAULT_TARGET_SERVICE_URL)
    # Allow overriding the env var name itself, though typically it's fixed for a given reward function.
    api_key_env_to_use = kwargs.get("api_key_env_var", API_KEY_ENV_VAR)

    if target_service_url == "YOUR_TUNNEL_URL_HERE":
        return EvaluateResult(
            score=0.0,
            reason=f"Placeholder DEFAULT_TARGET_SERVICE_URL is not replaced. Please pass 'target_service_url' in kwargs.",
            is_score_valid=False,
            metrics={},
        )

    api_key = os.getenv(api_key_env_to_use)

    if not api_key:
        return EvaluateResult(
            score=0.0,
            reason=f"API Key not found in environment variable '{api_key_env_to_use}'. "
            f"Ensure it's set (e.g., via 'fireworks secret set {api_key_env_to_use} your_key' or 'export {api_key_env_to_use}=your_key' for local tests).",
            is_score_valid=False,  # Cannot proceed without API key
            metrics={},
        )

    api_endpoint = f"{target_service_url.rstrip('/')}/api/validate_data"

    payload_data = {"info": "Default payload for secure call"}
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, Message):
            if last_message.role == "assistant" and last_message.content:
                payload_data = {
                    "info": "From assistant message (secure)",
                    "content": last_message.content,
                }
        elif isinstance(last_message, dict):
            if last_message.get("role") == "assistant" and last_message.get("content"):
                payload_data = {
                    "info": "From assistant message (secure)",
                    "content": last_message.get("content"),
                }

    item_payload = {"name": "SecureTestItem", "data": payload_data}
    headers = {"X-Secret-API-Key": api_key, "Content-Type": "application/json"}

    try:
        response = requests.post(
            api_endpoint, json=item_payload, headers=headers, timeout=10
        )

        if response.status_code == 200:
            response_data = response.json()
            return EvaluateResult(
                score=1.0,
                reason=f"API call successful (secure): {response_data.get('message', 'OK')}",
                is_score_valid=True,
                metrics={  # Only add api_response_code metric for successful calls
                    "api_response_code": MetricResult(
                        score=1.0,
                        is_score_valid=True,
                        reason=f"HTTP Status Code {response.status_code}",
                    )
                },
            )
        elif response.status_code == 401:
            return EvaluateResult(
                score=0.0,
                reason=f"API Authentication Error (401) (secure): {response.json().get('detail', 'Unauthorized - Header missing?')}",
                is_score_valid=True,
                metrics={},  # No specific metric score, error is in reason/main score
            )
        elif response.status_code == 403:
            return EvaluateResult(
                score=0.0,
                reason=f"API Authorization Error (403) (secure): {response.json().get('detail', 'Invalid API Key')}",
                is_score_valid=True,
                metrics={},  # No specific metric score, error is in reason/main score
            )
        else:
            return EvaluateResult(
                score=0.0,
                reason=f"API call failed with status {response.status_code} (secure): {response.text}",
                is_score_valid=True,
                metrics={},  # No specific metric score, error is in reason/main score
            )
    except requests.exceptions.RequestException as e:
        return EvaluateResult(
            score=0.0,
            reason=f"Network or request error (secure): {str(e)}",
            is_score_valid=False,
            metrics={},
        )
    except Exception as e:
        return EvaluateResult(
            score=0.0,
            reason=f"An unexpected error occurred (secure): {str(e)}",
            is_score_valid=False,
            metrics={},
        )


if __name__ == "__main__":
    # Example of how to test this function locally
    # 1. Start mock_api_service.py
    # 2. Start your tunnel (e.g., ngrok http 8001 or ssh -R 80:localhost:8001 serveo.net)
    #    and update DEFAULT_TARGET_SERVICE_URL above or pass 'target_service_url' as a kwarg.
    # 3. Set the MOCK_SERVICE_API_KEY environment variable:
    #    export MOCK_SERVICE_API_KEY="your_actual_api_key_for_mock_service"
    #    (e.g., the one mock_api_service.py expects)
    # 4. Run this script.

    print("Testing remote_validator_reward_secure function...")

    manual_test_url = "http://localhost:8001"

    effective_url_for_direct_test = DEFAULT_TARGET_SERVICE_URL
    if DEFAULT_TARGET_SERVICE_URL == "YOUR_TUNNEL_URL_HERE":
        print(
            f"\nNote: DEFAULT_TARGET_SERVICE_URL is a placeholder. For direct testing, either update it"
        )
        print(
            f"or ensure your mock service is reachable at '{manual_test_url}'. Using '{manual_test_url}'."
        )
        effective_url_for_direct_test = manual_test_url

    actual_api_key_for_test = os.getenv(API_KEY_ENV_VAR)

    if (
        effective_url_for_direct_test == "YOUR_TUNNEL_URL_HERE"
        and DEFAULT_TARGET_SERVICE_URL == "YOUR_TUNNEL_URL_HERE"
    ):
        print(
            f"\nERROR: Please update 'DEFAULT_TARGET_SERVICE_URL' or 'manual_test_url' in this script with your actual tunnel URL for direct testing."
        )
    elif not actual_api_key_for_test:
        print(
            f"\nERROR: Environment variable '{API_KEY_ENV_VAR}' is not set. Please set it to your mock service API key for testing."
        )
        print(f'  Example: export {API_KEY_ENV_VAR}="your_key_here"')
    else:
        print(
            f"Using Target Service URL for direct test: {effective_url_for_direct_test}"
        )
        print(
            f"Using API Key from env var '{API_KEY_ENV_VAR}': {'*' * (len(actual_api_key_for_test) - 4) + actual_api_key_for_test[-4:] if actual_api_key_for_test else 'Not Set'}"
        )

        sample_messages = [
            Message(role="user", content="Hello secure world"),
            Message(role="assistant", content="This is a secure test response."),
        ]

        result = remote_validator_reward_secure(
            messages=sample_messages, target_service_url=effective_url_for_direct_test
        )

        print(f"\nTest result:")
        print(f"  Score: {result.score}")
        print(f"  Reason: {result.reason}")
        if result.metrics:
            for k, v_metric in result.metrics.items():  # Renamed v to v_metric
                print(
                    f"  Metric '{k}': Score={v_metric.score}, Valid={v_metric.is_score_valid}, Reason={v_metric.reason}"
                )

    print(f"\nNote: For these tests to pass, the mock FastAPI service must be running,")
    print(
        f"your tunnel must be correctly forwarding, and the '{API_KEY_ENV_VAR}' environment variable must be set correctly."
    )
