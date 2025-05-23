import os
from typing import Dict, Literal, Optional

import yaml
from pydantic import BaseModel, ValidationError

CONFIG_FILE_NAME = "rewardkit.yaml"

# --- Pydantic Models for Configuration Structure ---

# Unused model, can be removed. The Dict[str, str] in GCPCloudRunConfig handles this.
# class GCPCloudRunSecretsConfig(BaseModel):
#     # Maps ENV_VAR_NAME_IN_FUNCTION_CONTAINER: gcp_secret_manager_id
#     # Example: "OPENAI_API_KEY": "projects/my-gcp-project/secrets/openai-api-key/versions/latest"
#     pass


class GCPCloudRunConfig(BaseModel):
    project_id: Optional[str] = None
    region: Optional[str] = None
    service_name_template: Optional[str] = "rewardeval-{evaluator_id}"
    default_auth_mode: Optional[Literal["api-key", "iam", "mtls-client-auth"]] = (
        "api-key"
    )
    secrets: Optional[Dict[str, str]] = {}  # Maps ENV_VAR_NAME to GCP Secret Manager ID


# Unused models, can be removed.
# class AWSLambdaSecretsConfig(BaseModel):
#     # Maps ENV_VAR_NAME_IN_FUNCTION_CONTAINER: aws_secrets_manager_arn
#     # Example: "OPENAI_API_KEY": "arn:aws:secretsmanager:us-east-1:123456789012:secret:my-openai-key-xxxxxx"
#     pass


class AWSLambdaConfig(BaseModel):
    region: Optional[str] = None
    function_name_template: Optional[str] = "rewardeval-{evaluator_id}"
    default_auth_mode: Optional[Literal["api-key", "iam", "mtls-client-auth"]] = (
        "api-key"
    )
    secrets: Optional[Dict[str, str]] = {}  # Maps ENV_VAR_NAME to AWS Secret ARN


class RewardKitConfig(BaseModel):
    default_deployment_target: Optional[
        Literal["gcp-cloud-run", "aws-lambda", "fireworks", "local"]
    ] = "fireworks"
    gcp_cloud_run: Optional[GCPCloudRunConfig] = GCPCloudRunConfig()
    aws_lambda: Optional[AWSLambdaConfig] = AWSLambdaConfig()
    evaluator_endpoint_keys: Optional[Dict[str, str]] = (
        {}
    )  # Stores generated API keys for self-hosted evaluator endpoints


# --- Global variable to hold the loaded configuration ---
_loaded_config: Optional[RewardKitConfig] = None
_config_file_path: Optional[str] = None


def find_config_file(start_path: Optional[str] = None) -> Optional[str]:
    """
    Finds the rewardkit.yaml file by searching upwards from start_path (or CWD).
    """
    if start_path is None:
        start_path = os.getcwd()

    current_path = os.path.abspath(start_path)
    while True:
        potential_path = os.path.join(current_path, CONFIG_FILE_NAME)
        if os.path.isfile(potential_path):
            return potential_path

        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:  # Reached root
            return None
        current_path = parent_path


def load_config(config_path: Optional[str] = None) -> RewardKitConfig:
    """
    Loads the rewardkit.yaml configuration.
    If already loaded, returns the cached version unless a new path is provided.
    If no path is provided, it tries to find rewardkit.yaml in CWD or parent directories.
    """
    global _loaded_config, _config_file_path

    if config_path:  # If a specific path is given, always try to load from it
        pass
    elif (
        _loaded_config and not config_path
    ):  # Already loaded and no new path, return cached
        return _loaded_config
    else:  # Not loaded or no specific path, try to find it
        config_path = find_config_file()

    if not config_path:
        # print(f"Info: No {CONFIG_FILE_NAME} found. Using default configuration.")
        _loaded_config = RewardKitConfig()  # Return default config if no file found
        _config_file_path = None
        return _loaded_config

    if (
        _loaded_config and config_path == _config_file_path
    ):  # Already loaded this specific file
        return _loaded_config

    try:
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        if raw_config is None:  # Empty YAML file
            _loaded_config = RewardKitConfig()
        else:
            _loaded_config = RewardKitConfig(**raw_config)

        _config_file_path = config_path
        # print(f"Successfully loaded configuration from: {config_path}")
        return _loaded_config
    except FileNotFoundError:
        # print(f"Warning: Configuration file not found at {config_path}. Using default configuration.")
        _loaded_config = RewardKitConfig()
        _config_file_path = None
        return _loaded_config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML from {config_path}: {e}")
        # Decide: raise error or return default? For now, return default and warn.
        _loaded_config = RewardKitConfig()
        _config_file_path = (
            config_path  # So it doesn't try to reload this broken file again
        )
        return _loaded_config
    except ValidationError as e:
        print(f"Error validating configuration from {config_path}: {e}")
        _loaded_config = RewardKitConfig()
        _config_file_path = config_path
        return _loaded_config
    except Exception as e:
        print(
            f"An unexpected error occurred while loading configuration from {config_path}: {e}"
        )
        _loaded_config = RewardKitConfig()
        _config_file_path = config_path
        return _loaded_config


def get_config() -> RewardKitConfig:
    """
    Returns the loaded configuration. Loads it if not already loaded.
    """
    if _loaded_config is None:
        return load_config()
    return _loaded_config


# Example usage (can be removed or kept for testing module directly)
if __name__ == "__main__":
    # Create a dummy rewardkit.yaml for testing
    dummy_config_content = """
default_deployment_target: gcp-cloud-run

gcp_cloud_run:
  project_id: "test-gcp-project"
  region: "europe-west1"
  default_auth_mode: "iam"
  secrets:
    DB_PASSWORD: "projects/test-gcp-project/secrets/db-password/versions/latest"

aws_lambda:
  region: "eu-north-1"

evaluator_endpoint_keys:
  my_test_eval: "dummy_key_for_test_eval"
"""
    dummy_file_path = os.path.join(os.getcwd(), CONFIG_FILE_NAME)
    with open(dummy_file_path, "w") as f:
        f.write(dummy_config_content)

    print(f"Created dummy {CONFIG_FILE_NAME} for testing.")

    config = get_config()
    print("\nLoaded configuration:")
    print(f"  Default Target: {config.default_deployment_target}")
    if config.gcp_cloud_run:
        print(f"  GCP Project ID: {config.gcp_cloud_run.project_id}")
        print(f"  GCP Region: {config.gcp_cloud_run.region}")
        print(f"  GCP Auth Mode: {config.gcp_cloud_run.default_auth_mode}")
        print(f"  GCP Secrets: {config.gcp_cloud_run.secrets}")
    if config.aws_lambda:
        print(f"  AWS Region: {config.aws_lambda.region}")
    print(f"  Endpoint Keys: {config.evaluator_endpoint_keys}")

    # Test finding config
    print("\nTesting find_config_file():")
    found_path = find_config_file()
    print(f"  Found at: {found_path}")

    # Clean up dummy file
    os.remove(dummy_file_path)
    print(f"\nRemoved dummy {CONFIG_FILE_NAME}.")

    # Test loading with no file
    _loaded_config = None  # Reset cache
    _config_file_path = None
    print("\nTesting get_config() with no file present:")
    config_no_file = get_config()
    print(f"  Default Target (no file): {config_no_file.default_deployment_target}")
    print(f"  GCP Config (no file): {config_no_file.gcp_cloud_run}")
