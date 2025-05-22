from typing import Dict, List, Any, Union, Optional
import requests  # Using requests library for HTTP calls

# Assuming reward_kit is installed and these are the correct import paths
# Adjust if necessary based on the actual package structure
from reward_kit.reward_function import reward_function
from reward_kit.models import Message, EvaluateResult, MetricResult

# --- Configuration (Hardcoded for Strategy A) ---
# IMPORTANT: This is a placeholder for a default URL if the tunnel URL isn't provided.
# The demo (run_demo.py) will override this with the actual tunnel URL via 'target_service_url' kwarg.
DEFAULT_TARGET_SERVICE_URL = "YOUR_TUNNEL_URL_HERE"
# This key demonstrates hardcoding. The demo (run_demo.py) overrides this with `target_api_key` kwarg.
HARDCODED_API_KEY = "d1bcc497c95659be6fbdcad869fa86390cefba53dc140b284d1508efdae81dd6"


@reward_function
def remote_validator_reward_hardcoded(
    messages: Union[List[Dict[str, Any]], List[Message]],
    ground_truth: Optional[str] = None,
    # kwargs is used to pass the actual target_service_url and target_api_key from run_demo.py
    **kwargs: Any,
) -> EvaluateResult:
    """
    Evaluates by calling a remote API using a hardcoded API key (by default) and a target URL.

    This function demonstrates an insecure way of handling secrets (API key) and configurations (URL).
    The `run_demo.py` script will typically override both `target_service_url` and `target_api_key` via kwargs.

    Args:
        messages: List of conversation messages. The last assistant message's content
                  is used as part of the payload if available.
        ground_truth: Optional expected correct answer (not used in this basic remote call).
        **kwargs: Additional arguments.
                  'target_service_url': Overrides DEFAULT_TARGET_SERVICE_URL. This is expected from run_demo.py.
                  'target_api_key': Overrides HARDCODED_API_KEY. This is expected from run_demo.py.

    Returns:
        EvaluateResult with evaluation score and metrics based on API response.
    """
    # Use kwargs to get the actual URL and API key, falling back to defaults if not provided.
    # run_demo.py is expected to provide these.
    target_service_url = kwargs.get("target_service_url", DEFAULT_TARGET_SERVICE_URL)
    api_key = kwargs.get("target_api_key", HARDCODED_API_KEY)

    if target_service_url == "YOUR_TUNNEL_URL_HERE":
        # This check is mostly for when testing this file directly without run_demo.py
        return EvaluateResult(
            score=0.0,
            reason="Placeholder DEFAULT_TARGET_SERVICE_URL is not replaced and 'target_service_url' not provided via kwargs.",
            is_score_valid=False,
            metrics={},
        )

    api_endpoint = f"{target_service_url.rstrip('/')}/api/validate_data"

    # Construct a payload from messages (e.g., use the last assistant message)
    payload_data = {"info": "Default payload"}
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, Message):  # If using Message Pydantic model
            if last_message.role == "assistant" and last_message.content:
                payload_data = {
                    "info": "From assistant message",
                    "content": last_message.content,
                }
        elif isinstance(last_message, dict):  # If using raw dicts
            if last_message.get("role") == "assistant" and last_message.get("content"):
                payload_data = {
                    "info": "From assistant message",
                    "content": last_message.get("content"),
                }

    item_payload = {"name": "TestItem", "data": payload_data}
    headers = {"X-Secret-API-Key": api_key, "Content-Type": "application/json"}

    try:
        response = requests.post(
            api_endpoint, json=item_payload, headers=headers, timeout=10
        )

        if response.status_code == 200:
            response_data = response.json()
            return EvaluateResult(
                score=1.0,
                reason=f"API call successful: {response_data.get('message', 'OK')}",
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
                reason=f"API Authentication Error (401): {response.json().get('detail', 'Unauthorized - Header missing?')}",
                is_score_valid=True,
                metrics={},  # No specific metric score, error is in reason/main score
            )
        elif response.status_code == 403:
            return EvaluateResult(
                score=0.0,
                reason=f"API Authorization Error (403): {response.json().get('detail', 'Invalid API Key')}",
                is_score_valid=True,
                metrics={},  # No specific metric score, error is in reason/main score
            )
        else:
            return EvaluateResult(
                score=0.0,
                reason=f"API call failed with status {response.status_code}: {response.text}",
                is_score_valid=True,
                metrics={},  # No specific metric score, error is in reason/main score
            )
    except requests.exceptions.RequestException as e:
        return EvaluateResult(
            score=0.0,
            reason=f"Network or request error: {str(e)}",
            is_score_valid=False,
            metrics={},
        )
    except Exception as e:
        return EvaluateResult(
            score=0.0,
            reason=f"An unexpected error occurred: {str(e)}",
            is_score_valid=False,
            metrics={},
        )


if __name__ == "__main__":
    # Example of how to test this function locally (requires mock API and a tunnel like Serveo/Ngrok running)
    # 1. Start mock_api_service.py (e.g., python examples/remote_eval_demo/mock_api_service.py)
    # 2. Start your tunnel (e.g., ngrok http 8001 or ssh -R 80:localhost:8001 serveo.net)
    #    and update DEFAULT_TARGET_SERVICE_URL above or pass 'target_service_url' as a kwarg.
    # 3. Run this script (e.g., python examples/remote_eval_demo/rewards/remote_validator_reward_hardcoded.py)

    print("Testing remote_validator_reward_hardcoded function...")

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

    print(f"Using Target Service URL for direct test: {effective_url_for_direct_test}")
    print(f"Using API Key for direct test: {HARDCODED_API_KEY}")

    sample_messages_valid = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="This is a test response for validation."),
    ]
    result_valid = remote_validator_reward_hardcoded(
        messages=sample_messages_valid,
        target_service_url=effective_url_for_direct_test,
        target_api_key=HARDCODED_API_KEY,
    )
    print(f"\nTest with (presumably) valid key:")
    print(f"  Score: {result_valid.score}")
    print(f"  Reason: {result_valid.reason}")
    if result_valid.metrics:
        for k, v_metric in result_valid.metrics.items():  # Renamed v to v_metric
            print(
                f"  Metric '{k}': Score={v_metric.score}, Valid={v_metric.is_score_valid}, Reason={v_metric.reason}"
            )

    sample_messages_invalid_key = [
        Message(role="assistant", content="Test with wrong key")
    ]
    result_invalid_key = remote_validator_reward_hardcoded(
        messages=sample_messages_invalid_key,
        target_service_url=effective_url_for_direct_test,
        target_api_key="WRONG_KEY_SHOULD_FAIL",
    )
    print(f"\nTest with invalid key (WRONG_KEY_SHOULD_FAIL):")
    print(f"  Score: {result_invalid_key.score}")
    print(f"  Reason: {result_invalid_key.reason}")
    if result_invalid_key.metrics:  # Added metrics check
        for k, v_metric in result_invalid_key.metrics.items():
            print(
                f"  Metric '{k}': Score={v_metric.score}, Valid={v_metric.is_score_valid}, Reason={v_metric.reason}"
            )

    print(f"\nTest with a bad Tunnel URL (simulating network error):")
    result_bad_url = remote_validator_reward_hardcoded(
        messages=sample_messages_valid, target_service_url="http://localhost:12345"
    )
    print(f"  Score: {result_bad_url.score}")
    print(f"  Reason: {result_bad_url.reason}")
    if result_bad_url.metrics:  # Added metrics check
        for k, v_metric in result_bad_url.metrics.items():
            print(
                f"  Metric '{k}': Score={v_metric.score}, Valid={v_metric.is_score_valid}, Reason={v_metric.reason}"
            )

    print(
        "\nNote: For these tests to pass, the mock FastAPI service must be running and your tunnel (if used) must be correctly forwarding to it."
    )
