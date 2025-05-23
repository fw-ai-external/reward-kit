import os
from typing import Dict, Literal, Optional

import yaml
from pydantic import BaseModel, ValidationError

CONFIG_FILE_NAME = "rewardkit.yaml"

# --- Pydantic Models for Configuration Structure ---


class GCPCloudRunConfig(BaseModel):
    project_id: Optional[str] = (
        None  # Default will be applied in deploy_command if not set
    )
    region: Optional[str] = None  # Default will be applied in deploy_command if not set
    artifact_registry_repository: Optional[str] = (
        None  # Default will be applied in deploy_command
    )
    service_name_template: Optional[str] = "rewardeval-{evaluator_id}"
    default_auth_mode: Optional[Literal["api-key", "iam", "mtls-client-auth"]] = (
        "api-key"  # Default auth mode if using GCP target and not specified
    )
    secrets: Optional[Dict[str, str]] = {}  # Maps ENV_VAR_NAME to GCP Secret Manager ID


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

        return _loaded_config
    except FileNotFoundError:

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
