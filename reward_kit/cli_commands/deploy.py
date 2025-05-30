"""
CLI command for creating and deploying an evaluator,
or registering a pre-deployed remote evaluator.
"""

import json
import os  # For os.getcwd()
import secrets  # For API key generation
import sys  # For sys.exit
from pathlib import Path  # For path operations
from typing import Any, Dict

import yaml  # For saving config if save_config helper doesn't exist

from reward_kit.auth import get_fireworks_account_id  # Added
from reward_kit.config import GCPCloudRunConfig, RewardKitConfig
from reward_kit.config import _config_file_path as global_loaded_config_path
from reward_kit.config import get_config
from reward_kit.evaluation import create_evaluation
from reward_kit.gcp_tools import (
    build_and_push_docker_image,
    deploy_to_cloud_run,
    ensure_artifact_registry_repo_exists,
    ensure_gcp_secret,
)
from reward_kit.packaging import generate_dockerfile_content
from reward_kit.platform_api import create_or_update_fireworks_secret  # Added
from reward_kit.platform_api import (  # For catching errors from create_evaluation
    PlatformAPIError,
)

from .common import check_environment


# Helper to save config (can be moved to config.py later)
def _save_config(config_data: RewardKitConfig, path: str):
    # Basic save, ideally config.py would provide a robust method
    try:
        with open(path, "w") as f:
            yaml.dump(config_data.model_dump(exclude_none=True), f, sort_keys=False)
        print(f"Config updated and saved to {path}")
    except Exception as e:
        print(f"Warning: Failed to save updated config to {path}: {e}")


def deploy_command(args):
    """Create and deploy an evaluator or register a remote one."""

    # Check environment variables
    if not check_environment():
        return 1

    if not args.id:  # ID is always required
        print("Error: Evaluator ID (--id) is required.")
        return 1

    # Process HuggingFace key mapping if provided
    huggingface_message_key_map = None
    if args.huggingface_key_map:
        try:
            huggingface_message_key_map = json.loads(args.huggingface_key_map)
        except json.JSONDecodeError:
            print("Error: Invalid JSON format for --huggingface-key-map")
            return 1

    if args.target == "gcp-cloud-run":
        print(f"Deploying evaluator '{args.id}' to GCP Cloud Run...")

        current_config = get_config()
        gcp_config_from_yaml = (
            current_config.gcp_cloud_run if current_config.gcp_cloud_run else None
        )

        # Resolve function_ref (must be from CLI for GCP)
        function_ref = args.function_ref
        if not function_ref:
            print("Error: --function-ref is required for GCP Cloud Run deployment.")
            return 1

        # Resolve GCP project_id: CLI > config.yaml > Error
        gcp_project_id = args.gcp_project
        if not gcp_project_id and gcp_config_from_yaml:
            gcp_project_id = gcp_config_from_yaml.project_id
        if not gcp_project_id:
            print(
                "Error: GCP Project ID must be provided via --gcp-project argument or in rewardkit.yaml."
            )
            return 1

        # Resolve GCP region: CLI > config.yaml > Error
        gcp_region = args.gcp_region
        if not gcp_region and gcp_config_from_yaml:
            gcp_region = gcp_config_from_yaml.region
        if not gcp_region:
            print(
                "Error: GCP Region must be provided via --gcp-region argument or in rewardkit.yaml."
            )
            return 1

        # Resolve GCP AR repo name: CLI > config.yaml > Fallback default
        gcp_ar_repo_name = args.gcp_ar_repo
        if not gcp_ar_repo_name and gcp_config_from_yaml:
            gcp_ar_repo_name = gcp_config_from_yaml.artifact_registry_repository
        if not gcp_ar_repo_name:
            gcp_ar_repo_name = "reward-kit-evaluators"  # Final fallback

        print(
            f"Using GCP Project: {gcp_project_id}, Region: {gcp_region}, AR Repo: {gcp_ar_repo_name}"
        )

        # 1. Ensure Artifact Registry repo exists
        if not ensure_artifact_registry_repo_exists(
            project_id=gcp_project_id, region=gcp_region, repo_name=gcp_ar_repo_name
        ):
            print(
                f"Failed to ensure Artifact Registry repository '{gcp_ar_repo_name}' exists. Aborting."
            )
            return 1

        # 2. Generate Dockerfile content
        dockerfile_content = generate_dockerfile_content(
            function_ref=args.function_ref,
            python_version=(
                f"{args.runtime[6]}.{args.runtime[7:]}"
                if args.runtime.startswith("python") and len(args.runtime) > 7
                else args.runtime.replace("python", "")
            ),
            reward_kit_install_source=".",
            user_requirements_path=None,
            service_port=8080,
        )
        if not dockerfile_content:
            print("Failed to generate Dockerfile content. Aborting.")
            return 1

        # 3. Build and push Docker image
        image_tag = "latest"
        image_name_tag = f"{gcp_region}-docker.pkg.dev/{gcp_project_id}/{gcp_ar_repo_name}/{args.id}:{image_tag}"

        build_context_dir = os.getcwd()

        if not build_and_push_docker_image(
            image_name_tag=image_name_tag,
            dockerfile_content=dockerfile_content,
            build_context_dir=build_context_dir,
            gcp_project_id=gcp_project_id,
        ):
            print(f"Failed to build and push Docker image {image_name_tag}. Aborting.")
            return 1

        print(f"Successfully built and pushed Docker image: {image_name_tag}")

        # Handle GCP Auth Mode
        gcp_env_vars: Dict[str, str] = {}
        parsed_gcp_secrets: Dict[str, Any] = (
            {}
        )  # Initialize for potential RK_ENDPOINT_API_KEY
        allow_unauthenticated_gcp = True  # Default for 'open' or 'api-key' at GCP infra level (app layer handles api-key)

        # Determine auth_mode: CLI explicit > config.yaml > 'api-key' (hardcoded default)
        resolved_auth_mode = "api-key"  # Start with the ultimate default
        if gcp_config_from_yaml and gcp_config_from_yaml.default_auth_mode:
            resolved_auth_mode = gcp_config_from_yaml.default_auth_mode
        if args.gcp_auth_mode is not None:  # CLI argument takes highest precedence
            resolved_auth_mode = args.gcp_auth_mode

        print(f"Using GCP Auth Mode: {resolved_auth_mode}")

        if resolved_auth_mode == "api-key":
            print(f"Configuring GCP Cloud Run service for API key authentication.")
            evaluator_id = args.id
            api_key = None
            # Use the imported global variable that stores the path of the loaded config file
            config_path = global_loaded_config_path

            if (
                current_config.evaluator_endpoint_keys
                and evaluator_id in current_config.evaluator_endpoint_keys
            ):
                api_key = current_config.evaluator_endpoint_keys[evaluator_id]
                print(
                    f"Using existing API key for '{evaluator_id}' from configuration."
                )
            else:
                api_key = secrets.token_hex(32)
                print(f"Generated new API key for '{evaluator_id}'.")
                if not current_config.evaluator_endpoint_keys:
                    current_config.evaluator_endpoint_keys = {}
                current_config.evaluator_endpoint_keys[evaluator_id] = api_key
                if config_path:
                    _save_config(current_config, config_path)
                else:
                    print(
                        f"Warning: No rewardkit.yaml found to save API key for '{evaluator_id}'. Key will be ephemeral for this deployment."
                    )

            # Instead of setting as direct env var, store in GCP Secret Manager and mount
            # gcp_env_vars["RK_ENDPOINT_API_KEY"] = api_key

            # Store the API key in GCP Secret Manager
            # Use a consistent secret ID, e.g., based on evaluator ID
            # Ensure args.id is sanitized if used directly in secret_id to avoid issues with special chars.
            # For gcloud secret IDs: 1-255 characters, [a-zA-Z0-9-_].
            gcp_sanitized_eval_id = "".join(
                filter(lambda char: char.isalnum() or char in ["-", "_"], args.id)
            )
            if not gcp_sanitized_eval_id:  # Fallback if ID is all special chars
                gcp_sanitized_eval_id = "rewardkit-evaluator"  # For GCP secret ID

            secret_id_for_auth_key = (
                f"rk-eval-{gcp_sanitized_eval_id}-authkey"  # GCP Secret ID
            )

            print(
                f"Storing API key for '{evaluator_id}' in GCP Secret Manager as '{secret_id_for_auth_key}'..."
            )
            secret_labels = {
                "managed-by": "reward-kit",
                "evaluator-id": evaluator_id,
            }  # Use original args.id for label

            # Ensure gcp_project is available (already resolved to gcp_project_id)
            api_key_secret_version_id = ensure_gcp_secret(
                project_id=gcp_project_id,
                secret_id=secret_id_for_auth_key,
                secret_value=api_key,
                labels=secret_labels,
                # region=args.gcp_region # Pass region if secrets need to be regional
            )

            if not api_key_secret_version_id:
                print(
                    f"Error: Failed to store API key in GCP Secret Manager for evaluator '{evaluator_id}'. Aborting."
                )
                return 1

            print(
                f"API key successfully stored. Secret version: {api_key_secret_version_id}"
            )

            # Prepare to mount this secret for the GCP Cloud Run service
            if parsed_gcp_secrets is None:  # Should have been initialized to {} earlier
                parsed_gcp_secrets = {}
            parsed_gcp_secrets["RK_ENDPOINT_API_KEY"] = api_key_secret_version_id

            # Now, also store this api_key on the Fireworks platform for the shim to use
            fireworks_account_id_for_secret = get_fireworks_account_id()
            if not fireworks_account_id_for_secret:
                print(
                    "Error: Fireworks Account ID not found, cannot store shim API key on Fireworks platform."
                )
                # Decide if this is fatal or just a warning. For now, let's make it a warning and proceed.
                # The shim won't be able to authenticate if the key isn't on the platform.
                print(
                    "Warning: The deployed service on GCP will be API key protected, but the Fireworks shim will not be able to provide the key."
                )
            else:
                # Construct a valid keyName for the secret on Fireworks platform
                # Rules: lowercase a-z, 0-9, and hyphen (-). Max length not specified, assume reasonable.
                fw_eval_id_sanitized = args.id.lower()
                fw_eval_id_sanitized = "".join(
                    filter(
                        lambda char: char.isalnum() or char == "-", fw_eval_id_sanitized
                    )
                )
                fw_eval_id_sanitized = "-".join(
                    filter(None, fw_eval_id_sanitized.split("-"))
                )  # Consolidate multiple hyphens
                if not fw_eval_id_sanitized:
                    fw_eval_id_sanitized = "evaluator"  # Fallback

                # Ensure overall keyName is not excessively long, e.g. by truncating fw_eval_id_sanitized if necessary
                # Max length for secret IDs is often around 63-255. Let's aim for < 60 for the ID part.
                max_len_eval_id_part = 40
                fw_eval_id_sanitized = fw_eval_id_sanitized[:max_len_eval_id_part]

                fw_secret_key_name = f"rkeval-{fw_eval_id_sanitized}-shim-key"

                print(
                    f"Registering API key on Fireworks platform as secret '{fw_secret_key_name}' for account '{fireworks_account_id_for_secret}'..."
                )
                if create_or_update_fireworks_secret(
                    account_id=fireworks_account_id_for_secret,
                    key_name=fw_secret_key_name,
                    secret_value=api_key,  # The same key used for the GCP service
                ):
                    print(
                        f"Successfully registered/updated secret '{fw_secret_key_name}' on Fireworks platform."
                    )
                else:
                    print(
                        f"Warning: Failed to register/update secret '{fw_secret_key_name}' on Fireworks platform. The shim may not authenticate correctly."
                    )

            # allow_unauthenticated_gcp remains True, as auth is handled by the app.

        # 4. Deploy to Cloud Run
        cloud_run_service_url = deploy_to_cloud_run(
            service_name=args.id,
            image_name_tag=image_name_tag,
            gcp_project_id=gcp_project_id,
            gcp_region=gcp_region,
            allow_unauthenticated=allow_unauthenticated_gcp,
            env_vars=gcp_env_vars if gcp_env_vars else None,  # Pass any other env_vars
            # service_account=args.service_account, # TODO: gcp_tools needs to support this
            secrets_to_mount=parsed_gcp_secrets,  # This now includes RK_ENDPOINT_API_KEY if auth_mode is api-key
        )

        if not cloud_run_service_url:
            print(f"Failed to deploy to Cloud Run or retrieve service URL. Aborting.")
            return 1

        print(
            f"Successfully deployed to Cloud Run. Service URL: {cloud_run_service_url}"
        )

        # 5. Register the Cloud Run URL with Fireworks AI platform
        try:
            print(
                f"Registering Cloud Run service URL '{cloud_run_service_url}' with Fireworks AI for evaluator '{args.id}'..."
            )
            evaluator = create_evaluation(
                evaluator_id=args.id,
                remote_url=cloud_run_service_url,
                display_name=args.display_name or args.id,
                description=args.description
                or f"GCP Cloud Run deployed evaluator: {args.id}",
                force=args.force,
                huggingface_dataset=args.huggingface_dataset,
                huggingface_split=args.huggingface_split,
                huggingface_message_key_map=huggingface_message_key_map,
                huggingface_prompt_key=args.huggingface_prompt_key,
                huggingface_response_key=args.huggingface_response_key,
            )

            evaluator_name = evaluator.get("name", args.id)
            print(
                f"Successfully registered GCP Cloud Run service as evaluator '{evaluator_name}' on Fireworks AI."
            )
            return 0
        except PlatformAPIError as e:
            print(
                f"Error registering GCP Cloud Run service URL with Fireworks AI: {str(e)}"
            )
            return 1
        except Exception as e:
            print(
                f"An unexpected error occurred during Fireworks AI registration: {str(e)}"
            )
            return 1

    elif args.remote_url:
        # Deploying an evaluator that proxies to a remote URL (standard Fireworks deployment of a shim).
        print(
            f"Deploying evaluator '{args.id}' configured to proxy to remote URL: {args.remote_url}"
        )
        if not (
            args.remote_url.startswith("http://")
            or args.remote_url.startswith("https://")
        ):
            print(
                f"Error: Invalid --remote-url '{args.remote_url}'. Must start with http:// or https://"
            )
            return 1

        if args.metrics_folders:
            print(
                "Info: --metrics-folders are not packaged when deploying with --remote-url. "
                "A shim will be generated to call the remote URL."
            )

        try:
            evaluator = create_evaluation(
                evaluator_id=args.id,
                remote_url=args.remote_url,
                display_name=args.display_name or args.id,
                description=args.description
                or f"Remote proxy evaluator for {args.id} at {args.remote_url}",
                force=args.force,
                huggingface_dataset=args.huggingface_dataset,
                huggingface_split=args.huggingface_split,
                huggingface_message_key_map=huggingface_message_key_map,
                huggingface_prompt_key=args.huggingface_prompt_key,
                huggingface_response_key=args.huggingface_response_key,
            )
            evaluator_name = evaluator.get("name", args.id)
            print(
                f"Successfully created/updated evaluator '{evaluator_name}' to proxy to {args.remote_url}"
            )
            return 0
        except PlatformAPIError as e:
            print(f"Error deploying remote proxy evaluator '{args.id}': {str(e)}")
            return 1
        except Exception as e:
            print(
                f"An unexpected error occurred while deploying remote proxy evaluator '{args.id}': {str(e)}"
            )
            return 1

    else:
        # Original behavior: Deploying by packaging local metrics_folders to Fireworks AI directly
        if not args.metrics_folders:
            print(
                "Error: --metrics-folders are required if not using --remote-url and target is not gcp-cloud-run."
            )
            return 1

        for folder_spec in args.metrics_folders:
            if "=" not in folder_spec:
                print(
                    f"Error: Metric folder format should be 'name=path', got '{folder_spec}'"
                )
                return 1

        try:
            evaluator = create_evaluation(
                evaluator_id=args.id,
                metric_folders=args.metrics_folders,
                display_name=args.display_name or args.id,
                description=args.description or f"Evaluator: {args.id}",
                force=args.force,
                huggingface_dataset=args.huggingface_dataset,
                huggingface_split=args.huggingface_split,
                huggingface_message_key_map=huggingface_message_key_map,
                huggingface_prompt_key=args.huggingface_prompt_key,
                huggingface_response_key=args.huggingface_response_key,
            )
            evaluator_name = evaluator.get("name", args.id)
            print(f"Successfully created/updated evaluator: {evaluator_name}")
            return 0
        except PlatformAPIError as e:
            print(f"Error creating/updating evaluator '{args.id}': {str(e)}")
            return 1
        except Exception as e:
            print(f"Error creating/updating evaluator '{args.id}': {str(e)}")
            return 1
