from typing import Dict, List, Optional, Any, Union, Callable, Type, TypeVar, cast
import os
import importlib
import importlib.util
import inspect
import json
import requests
from pathlib import Path
from functools import wraps
import logging

from .models import (
    RewardOutput,
    MetricRewardOutput,
    Message,
    EvaluateResult,
    MetricResult,
)
from .typed_interface import reward_function

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type for reward function
T = TypeVar("T", bound=Callable[..., RewardOutput])


class RewardFunction:
    """
    A wrapper for reward functions that allows them to be run locally or remotely.

    The RewardFunction class wraps a reward function (either a local function or a remote endpoint)
    and provides a unified interface for calling it. It supports:

    - Local functions (mode="local")
    - Remote endpoints (mode="remote")
    - Fireworks-hosted models (mode="fireworks_hosted")

    Args:
        func: The local function to use (for mode="local")
        func_path: A string path to a function (e.g., "module.submodule:function_name")
        mode: The mode of operation ("local", "remote", or "fireworks_hosted")
        endpoint: The URL of the remote endpoint (for mode="remote")
        model_id: The ID of the Fireworks-hosted model (for mode="fireworks_hosted")
        **kwargs: Additional keyword arguments to pass to the function
    """

    def __init__(
        self,
        func: Optional[Callable] = None,
        func_path: Optional[str] = None,
        mode: str = "local",
        endpoint: Optional[str] = None,
        name: Optional[str] = None,
        model_id: Optional[str] = None,
        **kwargs,
    ):
        self.mode = mode
        self.func = func
        self.func_path = func_path
        self.endpoint = endpoint
        self.name = name
        self.model_id = model_id
        self.kwargs = kwargs

        if mode == "local":
            if func is None and func_path is None:
                raise ValueError(
                    "Either 'func' or 'func_path' must be provided for local mode"
                )
            if func_path and func is None:
                self.func = self._load_function_from_path(func_path)
        elif mode == "remote":
            if endpoint is None and name is None:
                raise ValueError(
                    "Either 'endpoint' or 'name' must be provided for remote mode"
                )
            if name and endpoint is None:
                # Construct endpoint URL from name (in a real implementation,
                # this would fetch from the Fireworks API)
                self.endpoint = f"https://api.fireworks.ai/v1/reward/{name}"
        elif mode == "fireworks_hosted":
            if model_id is None:
                raise ValueError(
                    "'model_id' must be provided for fireworks_hosted mode"
                )
            # Construct endpoint for the Fireworks-hosted model
            self.endpoint = f"https://api.fireworks.ai/v1/models/{model_id}/reward"
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _load_function_from_path(self, func_path: str) -> Callable:
        """
        Load a function from a path string.

        Handles two formats:
        - 'module.path:function_name' - Module with colon separator
        - 'module.path.function_name' - Module with function as last component
        """
        # Check for the colon format first (preferred)
        if ":" in func_path:
            module_path, func_name = func_path.split(":", 1)

            try:
                module = importlib.import_module(module_path)
                func = getattr(module, func_name)
                return func
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"Failed to load function from path {func_path}: {str(e)}"
                )

        # Try dot notation format: module.path.function_name
        # This assumes the last component is the function name
        parts = func_path.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid func_path format: {func_path}, expected 'module.path:function_name' or 'module.path.function_name'"
            )

        module_path = ".".join(parts[:-1])
        func_name = parts[-1]

        try:
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            return func
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to load function from path {func_path}: {str(e)}"
            )

    def __call__(
        self,
        messages: List[Dict[str, str]],
        original_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> RewardOutput:
        """
        Call the reward function with the provided messages.

        Args:
            messages: List of conversation messages, each with 'role' and 'content' keys
            original_messages: Original conversation messages (for context)
            **kwargs: Additional keyword arguments to pass to the function

        Returns:
            RewardOutput object with score and metrics
        """
        if original_messages is None:
            original_messages = messages[:-1] if messages else []

        # Combine instance kwargs with call kwargs
        combined_kwargs = {**self.kwargs, **kwargs}

        if self.mode == "local":
            if self.func is None:
                raise ValueError("No function provided for local mode")

            # Call the local function
            try:
                result = self.func(
                    messages=messages,
                    original_messages=original_messages,
                    **combined_kwargs,
                )

                # Ensure the result is a RewardOutput
                if isinstance(result, RewardOutput):
                    return result
                elif isinstance(result, tuple) and len(result) == 2:
                    # Handle legacy (score, components) tuple format
                    score, components = result
                    metrics = {
                        k: MetricRewardOutput(score=v, reason=None)
                        for k, v in components.items()
                    }
                    return RewardOutput(score=score, metrics=metrics)
                else:
                    raise TypeError(
                        f"Invalid return type from reward function: {type(result)}"
                    )

            except Exception as e:
                logger.error(f"Error calling local reward function: {str(e)}")
                raise

        elif self.mode in ["remote", "fireworks_hosted"]:
            if self.endpoint is None:
                raise ValueError(f"No endpoint provided for {self.mode} mode")

            # Prepare the payload
            payload = {
                "messages": messages,
                "original_messages": original_messages,
                **combined_kwargs,
            }

            # Get API key
            api_key = os.environ.get("FIREWORKS_API_KEY")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}" if api_key else "",
            }

            try:
                response = requests.post(self.endpoint, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()

                # Convert the result to RewardOutput
                if isinstance(result, dict) and "score" in result:
                    return RewardOutput.from_dict(result)
                else:
                    raise ValueError(f"Invalid response from remote endpoint: {result}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Error calling remote endpoint: {str(e)}")
                raise

        raise ValueError(f"Invalid mode: {self.mode}")

    def get_trl_adapter(self) -> Callable:
        """
        Create an adapter function for use with TRL library.

        The TRL library expects a function that takes batch inputs and returns a batch of reward values.
        This adapter handles:
        1. Batch of messages (List[List[Dict]]) and original messages (List[List[Dict]])
        2. Batch of texts (List[str]) for simpler cases

        Returns:
            A callable function compatible with TRL
        """

        def adapter(batch_input, batch_orig_input=None) -> List[float]:
            results = []

            # Check if this is simple text input or structured messages
            if isinstance(batch_input, list):
                if not batch_input:
                    return []

                # Case 1: List of message arrays (TRL batch format)
                if isinstance(batch_input[0], list) and all(
                    isinstance(m, dict) for m in batch_input[0]
                ):
                    for i, messages in enumerate(batch_input):
                        try:
                            # Get original messages if provided, otherwise use messages minus last one
                            original_msgs = (
                                batch_orig_input[i]
                                if batch_orig_input
                                else messages[:-1]
                            )
                            reward_output = self(
                                messages=messages, original_messages=original_msgs
                            )
                            results.append(reward_output.score)
                        except Exception as e:
                            logger.error(f"Error in TRL adapter: {str(e)}")
                            results.append(0.0)

                # Case 2: List of strings (simple TRL format)
                elif all(isinstance(text, str) for text in batch_input):
                    for text in batch_input:
                        try:
                            # TRL typically provides just the completion, so wrap it in a message
                            messages = [{"role": "assistant", "content": text}]
                            reward_output = self(messages=messages)
                            results.append(reward_output.score)
                        except Exception as e:
                            logger.error(f"Error in TRL adapter: {str(e)}")
                            results.append(0.0)
                else:
                    raise ValueError(
                        f"Unsupported input format for TRL adapter: {type(batch_input[0])}"
                    )
            else:
                raise ValueError(
                    f"Unsupported input type for TRL adapter: {type(batch_input)}"
                )

            return results

        return adapter


def reward_function(func: T) -> T:
    """
    Decorator for reward functions that adds deployment capabilities.

    This decorator wraps a function to ensure it returns a RewardOutput and adds
    a .deploy() method that can be used to deploy the function to Fireworks.

    Args:
        func: The reward function to decorate

    Returns:
        The decorated function with added deployment capabilities
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> RewardOutput:
        result = func(*args, **kwargs)

        # Ensure the result is a RewardOutput
        if isinstance(result, RewardOutput):
            return result
        elif isinstance(result, tuple) and len(result) == 2:
            # Handle legacy (score, components) tuple format
            score, components = result
            metrics = {
                k: MetricRewardOutput(score=v, reason=None)
                for k, v in components.items()
            }
            return RewardOutput(score=score, metrics=metrics)
        else:
            raise TypeError(
                f"Invalid return type from reward function: {type(result)}. "
                f"Expected RewardOutput or (float, Dict[str, float]) tuple."
            )

    def deploy(**config) -> str:
        """
        Deploy the reward function to Fireworks as an evaluation with a Python code assertion.

        Args:
            **config: Configuration options for deployment
                name (str): Name for the evaluation
                description (str, optional): Description of the evaluation
                account_id (str, optional): Fireworks account ID. If not provided,
                                           will be read from ~/.fireworks/auth.ini
                providers (list, optional): List of provider configurations
                                           Defaults to a single provider with current model

        Returns:
            A string evaluation ID that can be used in RL training
        """
        import configparser
        import os
        import requests
        from pathlib import Path

        # Get configuration parameters
        name = config.get("name", func.__name__)
        description = config.get(
            "description", f"Reward function deployed from {func.__name__}"
        )

        # Get function source code
        source = inspect.getsource(func)

        # Load authentication info
        account_id = config.get("account_id")
        auth_token = config.get("auth_token")

        # If not provided directly, try to load from config files
        if not account_id or not auth_token:
            try:
                auth_path = Path.home() / ".fireworks" / "auth.ini"
                if auth_path.exists():
                    auth_config = configparser.ConfigParser()
                    auth_config.read(auth_path)
                    if "default" in auth_config:
                        if not account_id and "account_id" in auth_config["default"]:
                            account_id = auth_config["default"]["account_id"]
                        if not auth_token and "id_token" in auth_config["default"]:
                            auth_token = auth_config["default"]["id_token"]
            except Exception as e:
                logger.error(f"Error reading auth config: {str(e)}")

        if not account_id:
            raise ValueError(
                "account_id not provided and could not be loaded from ~/.fireworks/auth.ini"
            )

        if not auth_token:
            auth_token = os.environ.get("FIREWORKS_API_KEY")
            if not auth_token:
                raise ValueError(
                    "Authentication token not found. Please run 'firectl signin' or set FIREWORKS_API_KEY"
                )

        # Get or create default providers
        providers = config.get(
            "providers",
            [
                {
                    "providerType": "fireworks",
                    "modelId": "accounts/fireworks/models/llama-v3-8b-instruct",
                }
            ],
        )

        # Create wrapper code that converts the function to a proper reward evaluation
        # This generates a Python snippet that will:
        # 1. Parse input from the evaluation framework
        # 2. Call our reward function
        # 3. Format the output appropriately
        # Check if we need to import the reward kit models
        module = inspect.getmodule(func)
        module_imports = inspect.getsource(module) if module else ""

        # Define needed imports for the wrapper code
        imports_needed = (
            "from typing import Dict, List, Optional, Any\n"
            "from dataclasses import dataclass\n\n"
            "@dataclass\n"
            "class MetricRewardOutput:\n"
            "    score: float\n"
            "    reason: Optional[str] = None\n\n"
            "@dataclass\n"
            "class RewardOutput:\n"
            "    score: float\n"
            "    metrics: Dict[str, MetricRewardOutput] = None\n"
            "    \n"
            "    def to_dict(self):\n"
            "        return {\n"
            '            "score": self.score,\n'
            '            "metrics": {\n'
            '                k: {"score": v.score, "reason": v.reason}\n'
            "                for k, v in (self.metrics or {}).items()\n"
            "            }\n"
            "        }\n"
        )

        # Only add imports if they're not already in the module
        if "class RewardOutput" not in module_imports:
            extra_imports = imports_needed
        else:
            extra_imports = ""

        # Format the wrapper code to handle execution of the reward function
        wrapper_code = (
            f"# Original function: {func.__name__}\n"
            "import json\n"
            "import sys\n"
            "from typing import Dict, List, Optional, Any\n\n"
            f"{extra_imports}\n"
            f"{source}\n\n"
            "def evaluate(input_data):\n"
            "    try:\n"
            "        # Parse input data\n"
            "        data = json.loads(input_data)\n"
            "        messages = data.get('messages', [])\n"
            "        original_messages = data.get('original_messages', messages[:-1] if messages else [])\n"
            "        kwargs = data.get('kwargs', {})\n"
            "        \n"
            f"        # Call reward function\n"
            f"        result = {func.__name__}(messages=messages, original_messages=original_messages, **kwargs)\n"
            "        \n"
            "        # Format result as expected by the evaluation system\n"
            "        if hasattr(result, 'to_dict'):\n"
            "            result_dict = result.to_dict()\n"
            "        elif hasattr(result, '__dict__'):\n"
            "            result_dict = result.__dict__\n"
            "        else:\n"
            "            result_dict = {'score': result}\n"
            "            \n"
            "        return json.dumps(result_dict)\n"
            "    except Exception as e:\n"
            "        return json.dumps({'error': str(e), 'score': 0.0})\n\n"
            "# Process input from the evaluation system\n"
            "if __name__ == '__main__':\n"
            "    input_data = sys.stdin.read()\n"
            "    output = evaluate(input_data)\n"
            "    print(output)\n"
        )

        # Create evaluation payload
        api_base = os.environ.get("FIREWORKS_API_BASE", "https://api.fireworks.ai")

        # Create an evaluator object payload as expected by the API
        evaluator = {
            "displayName": name,
            "description": description,
            "multiMetrics": False,
            "criteria": [
                {
                    "type": "CODE_SNIPPETS",
                    "name": name,
                    "description": description,
                    "codeSnippets": {
                        "language": "python",
                        "fileContents": {"main.py": wrapper_code},
                    },
                }
            ],
            "requirements": "",
            "rollupSettings": None,
        }

        # The POST body requires the evaluator nested under "evaluator" and an optional "evaluatorId"
        evaluation_payload = {"evaluator": evaluator, "evaluatorId": name}

        # Get API base URL from environment or use default
        api_base = os.environ.get("FIREWORKS_API_BASE", "https://api.fireworks.ai")

        # Send request to create evaluation
        url = f"{api_base}/v1/accounts/{account_id}/evaluators"
        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }

        # Log request details for debugging
        logger.info(f"Making request to: {url} (using API base: {api_base})")
        logger.info(f"Using account_id: {account_id}")
        logger.info(f"Auth token present: {bool(auth_token)}")

        # Check if we should force update an existing evaluation
        force = config.get("force", False)

        try:
            logger.info(
                f"Deploying reward function '{func.__name__}' as evaluation '{name}'..."
            )

            if force:
                # First try to check if evaluator already exists
                evaluator_id = name
                check_url = (
                    f"{api_base}/v1/accounts/{account_id}/evaluators/{evaluator_id}"
                )

                try:
                    # Check if the evaluator exists
                    check_response = requests.get(check_url, headers=headers)
                    if check_response.status_code == 200:
                        # Evaluator exists, delete it first then recreate
                        logger.info(
                            f"Evaluator '{evaluator_id}' already exists, deleting and recreating..."
                        )
                        delete_url = f"{api_base}/v1/accounts/{account_id}/evaluators/{evaluator_id}"
                        try:
                            # Try to delete the evaluator
                            delete_response = requests.delete(
                                delete_url, headers=headers
                            )
                            # Don't raise for status here, we'll try to create it anyway
                            if delete_response.status_code < 400:
                                logger.info(
                                    f"Successfully deleted evaluator '{evaluator_id}'"
                                )
                            else:
                                logger.warning(
                                    f"Unable to delete evaluator '{evaluator_id}', status: {delete_response.status_code}"
                                )
                        except Exception as e:
                            logger.warning(f"Error deleting evaluator: {str(e)}")

                        # Now create it
                        response = requests.post(
                            url, json=evaluation_payload, headers=headers
                        )
                    else:
                        # Evaluator doesn't exist, create it
                        response = requests.post(
                            url, json=evaluation_payload, headers=headers
                        )
                except requests.exceptions.RequestException:
                    # If checking fails, try to create it
                    response = requests.post(
                        url, json=evaluation_payload, headers=headers
                    )
            else:
                # Just try to create it
                response = requests.post(url, json=evaluation_payload, headers=headers)

            response.raise_for_status()
            result = response.json()

            evaluation_id = result.get("name", "").split("/")[-1]
            evaluation_url = (
                f"{api_base}/v1/accounts/{account_id}/evaluators/{evaluation_id}"
            )

            logger.info(f"Deployment successful. Evaluation ID: {evaluation_id}")
            logger.info(f"Evaluation URL: {evaluation_url}")

            return evaluation_id
        except Exception as e:
            logger.error(f"Error deploying evaluation: {str(e)}")
            if isinstance(e, requests.exceptions.HTTPError) and hasattr(e, "response"):
                logger.error(f"Response: {e.response.text}")
            raise

    # Add the deploy method to the function
    wrapper.deploy = deploy  # type: ignore

    return cast(T, wrapper)
