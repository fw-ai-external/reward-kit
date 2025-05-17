import json
import os

import requests

# Print API key info (masked)
api_key = os.environ.get("FIREWORKS_API_KEY", "")
if api_key:
    print(f"API key found: {api_key[:4]}...{api_key[-4:]}")
else:
    print("No API key found in environment")

# Read account ID from settings
account_id = os.environ.get("FIREWORKS_ACCOUNT_ID", "")
if not account_id:
    print("No account ID found in environment, trying to read from settings file")
    import pathlib

    settings_path = pathlib.Path.home() / ".fireworks" / "settings.ini"
    if settings_path.exists():
        try:
            with open(settings_path, "r") as f:
                for line in f:
                    if "account_id" in line and "=" in line:
                        account_id = line.split("=", 1)[1].strip()
                        break
        except Exception as e:
            print(f"Error reading settings file: {str(e)}")

if account_id:
    print(f"Using account ID: {account_id}")
else:
    print("No account ID found")

# Test API connection
try:
    # Try listing models to verify API connectivity
    headers = {"Authorization": f"Bearer {api_key}"}
    base_url = "https://api.fireworks.ai/v1"

    # Check if models endpoint works (to verify API connection)
    models_url = f"{base_url}/models?limit=1"
    print(f"Testing models endpoint: {models_url}")
    response = requests.get(models_url, headers=headers)
    print(f"Response: {response.status_code} - {response.reason}")
    if response.status_code == 200:
        print("Successfully connected to Fireworks API")
    else:
        print(f"Error response: {response.text}")

    if account_id:
        # Check if the evaluations endpoint is available
        eval_url = f"{base_url}/accounts/{account_id}/evaluations"
        print(f"Testing evaluations endpoint: {eval_url}")
        response = requests.get(eval_url, headers=headers)
        print(f"Response: {response.status_code} - {response.reason}")
        if response.status_code != 200:
            print(f"Error response: {response.text}")

        # Check if there's an evaluators endpoint
        evaluators_url = f"{base_url}/accounts/{account_id}/evaluators"
        print(f"Testing evaluators endpoint: {evaluators_url}")
        response = requests.get(evaluators_url, headers=headers)
        print(f"Response: {response.status_code} - {response.reason}")
        if response.status_code != 200:
            print(f"Error response: {response.text}")

        # Look for alternate endpoints
        for endpoint in ["evaluation", "evaluator"]:
            url = f"{base_url}/accounts/{account_id}/{endpoint}"
            print(f"Testing alternate endpoint: {url}")
            response = requests.get(url, headers=headers)
            print(f"Response: {response.status_code} - {response.reason}")

except Exception as e:
    print(f"Error connecting to Fireworks API: {str(e)}")
