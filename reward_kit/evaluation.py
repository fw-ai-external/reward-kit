import json
import logging
import os
import time
from pathlib import Path
import types
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    # For type checking only
    import datasets  # noqa: F401

import requests

from reward_kit.auth import get_fireworks_account_id, get_fireworks_api_key

logger = logging.getLogger(__name__)

# Flag to track if the preview API was successfully used
used_preview_api = False


def huggingface_dataset_to_jsonl(
    dataset_name: str,
    split: str = "train",
    output_file: Optional[str] = None,
    max_samples: int = 100,
    message_key_map: Optional[Dict[str, str]] = None,
    response_key: str = "response",
    prompt_key: str = "prompt",
) -> str:
    """
    Converts a HuggingFace dataset to JSONL format suitable for
    reward-kit evaluation.

    Args:
        dataset_name: The name of the HuggingFace dataset
            (e.g., "deepseek-ai/DeepSeek-ProverBench")
        split: The dataset split to use (default: "train")
        output_file: Optional file path to save the JSONL output
            (if None, generates a temp file)
        max_samples: Maximum number of samples to include
        message_key_map: Optional mapping of dataset keys to
            reward-kit message keys
        response_key: Key in the dataset containing the response text
            (default: "response")
        prompt_key: Key in the dataset containing the prompt text
            (default: "prompt")

    Returns:
        Path to the generated JSONL file
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required to use this function. "
            "Please install it with 'pip install \"reward-kit[deepseek]\"'"
        )

    import tempfile

    # Load dataset from Hugging Face
    logger.info(f"Loading dataset {dataset_name} (split: {split})")
    dataset = load_dataset(dataset_name, split=split)

    # Generate output file if not provided
    if not output_file:
        temp_dir = tempfile.gettempdir()
        dataset_basename = dataset_name.split("/")[-1]
        output_file = os.path.join(
            temp_dir, f"{dataset_basename}_{split}_{int(time.time())}.jsonl"
        )

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Default key mapping if not provided
    if message_key_map is None:
        message_key_map = {}

    # Process dataset items
    # count = 0 # F841: Unused local variable
    with open(output_file, "w") as f:
        # Limit to max_samples
        processed_samples = 0
        for i, item in enumerate(dataset):
            if processed_samples >= max_samples:
                break

            # Skip items without required keys
            if prompt_key not in item and "statement" not in item:
                logger.debug(
                    f"Skipping sample {i} due to missing prompt/statement key."
                )
                continue

            # Convert dataset item to reward-kit format
            prompt_text = item.get(prompt_key, item.get("statement", ""))
            response_text = item.get(
                response_key,
                item.get("reference_solution", item.get("expected_proof", "")),
            )

            if not prompt_text or not response_text:
                logger.debug(
                    (
                        f"Skipping sample {i} due to missing prompt or "
                        "response text."
                    )
                )
                continue

            # Create messages array
            messages = [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": response_text},
            ]

            # Create the entry with messages
            entry = {"messages": messages}

            # Add additional fields based on key mapping
            for ds_key, rk_key in message_key_map.items():
                if ds_key in item:
                    entry[rk_key] = item[ds_key]

            # Add all remaining keys as kwargs
            for key, value in item.items():
                if (
                    key not in [prompt_key, response_key]
                    and key not in message_key_map
                ):
                    entry[key] = value

            # Write the entry
            f.write(json.dumps(entry) + "\n")
            processed_samples += 1

        # Use 'processed_samples' to report the count
        # If loop didn't run, i might not be defined.
        if processed_samples == 0 and i == -1:  # if dataset was empty
            logger.info(f"No samples converted to JSONL format: {output_file}")
        else:
            logger.info(
                (
                    f"Converted {processed_samples} samples to JSONL format: "
                    f"{output_file}"
                )
            )
    return output_file


class EvaluatorPreviewResult:
    """Class to store preview results for an evaluator"""

    def __init__(self):
        self.results = []
        self.total_samples = 0
        self.total_runtime_ms = 0

    def add_result(self, sample_index, success, score, per_metric_evals):
        """Add a result for a specific sample"""
        # Store as a SimpleNamespace to allow attribute access
        result_obj = types.SimpleNamespace(
            index=sample_index,
            success=success,
            score=score,
            per_metric_evals=per_metric_evals,
        )
        self.results.append(result_obj)

    def display(self):
        """Display formatted results"""
        print("Evaluation Preview Results")
        print("------------------------")
        print(f"Total Samples: {self.total_samples}")
        print(f"Total Runtime: {self.total_runtime_ms} ms\n")
        print("Individual Results:")
        print("------------------")

        for i, result_obj in enumerate(
            self.results
        ):  # Renamed result to result_obj for clarity
            print(f"Sample {result_obj.index + 1}:")  # Use attribute access
            print(f"  Success: {result_obj.success}")  # Use attribute access
            print(f"  Score: {result_obj.score}")  # Use attribute access
            # per_metric_evals is likely a dict stored in the SimpleNamespace
            if hasattr(result_obj, "per_metric_evals") and isinstance(
                result_obj.per_metric_evals, dict
            ):
                for metric, value in result_obj.per_metric_evals.items():
                    print(f"  {metric}: {value}")
            elif hasattr(
                result_obj, "per_metric_evals"
            ):  # If it's not a dict but exists
                print(f"  Per-Metric Evals: {result_obj.per_metric_evals}")
            if i < len(self.results) - 1:
                print()


class Evaluator:
    """Handles loading, previewing, and creating evaluations"""

    def __init__(
        self,
        multi_metrics=False,
        ts_mode_config: Optional[Dict[str, Any]] = None,
    ):
        self.multi_metrics = (
            multi_metrics  # Original multi_metrics flag for folder-based setup
        )
        self.ts_mode_config = ts_mode_config  # Config for TypeScript-like single code snippet mode
        self.code_files = {}  # Map of filename -> content
        self.metric_folders: dict[str, str] = (
            {}
        )  # Map of metric_name -> folder_path (used when not in ts_mode_config and not multi_metrics)
        self.description = ""
        self.display_name = ""
        self.api_base = os.environ.get(
            "FIREWORKS_API_BASE", "https://api.fireworks.ai"
        )

        if self.ts_mode_config:
            # If in ts_mode, populate code_files directly and set multi_metrics to True
            # This ensures that later checks for self.code_files and main.py pass.
            python_code = self.ts_mode_config.get("python_code")
            file_name = self.ts_mode_config.get(
                "file_name", "main.py"
            )  # Default to main.py
            if not python_code:
                raise ValueError("python_code is required in ts_mode_config")
            self.code_files[file_name] = python_code
            self.multi_metrics = True  # ts_mode implies multiMetrics: true and specific rollup settings

    def load_metric_folder(self, metric_name, folder_path):
        """
        Load code files from a metric folder

        Args:
            metric_name: Name of the metric
            folder_path: Path to the folder containing code files

        Returns:
            Dict mapping filenames to their contents
        """
        folder_path = os.path.abspath(folder_path)
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder does not exist: {folder_path}")

        if not os.path.isdir(folder_path):
            raise ValueError(f"Not a directory: {folder_path}")

        files = {}
        for file_path in Path(folder_path).glob("*.py"):
            if file_path.is_file():
                with open(file_path, "r") as f:
                    filename = file_path.name
                    content = f.read()
                    files[filename] = content

                    # Check for main.py with evaluate function
                    if filename == "main.py" and "evaluate" not in content:
                        raise ValueError(
                            f"main.py in {folder_path} must contain an evaluate function"
                        )

        if "main.py" not in files:
            raise ValueError(f"main.py is required in {folder_path}")

        self.metric_folders[metric_name] = folder_path
        for filename, content in files.items():
            self.code_files[f"{metric_name}/{filename}"] = content

        logger.info(
            f"Loaded {len(files)} Python files for metric '{metric_name}' from {folder_path}"
        )
        return files

    def load_multi_metrics_folder(self, folder_path):
        """
        Load code files from a folder with multiple metrics

        Args:
            folder_path: Path to the folder containing code files

        Returns:
            Dict mapping filenames to their contents
        """
        folder_path = os.path.abspath(folder_path)
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder does not exist: {folder_path}")

        if not os.path.isdir(folder_path):
            raise ValueError(f"Not a directory: {folder_path}")

        files = {}
        for file_path in Path(folder_path).glob("*.py"):
            if file_path.is_file():
                with open(file_path, "r") as f:
                    filename = file_path.name
                    content = f.read()
                    files[filename] = content

                    # Check for main.py with evaluate function
                    if filename == "main.py" and "evaluate" not in content:
                        raise ValueError(
                            f"main.py in {folder_path} must contain an evaluate function"
                        )

        if "main.py" not in files:
            raise ValueError(f"main.py is required in {folder_path}")

        self.code_files = files
        logger.info(
            f"Loaded {len(files)} Python files from {folder_path} "
            f"for multi-metrics evaluation"
        )
        return files

    def load_samples_from_jsonl(self, sample_file, max_samples=5):
        """
        Load samples from a JSONL file

        Args:
            sample_file: Path to the JSONL file
            max_samples: Maximum number of samples to load

        Returns:
            List of parsed JSON objects
        """
        if not os.path.exists(sample_file):
            raise ValueError(f"Sample file does not exist: {sample_file}")

        samples = []
        with open(sample_file, "r") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON on line {i + 1}, skipping")

        logger.info(f"Loaded {len(samples)} samples from {sample_file}")
        return samples

    def preview(self, sample_file, max_samples=5):
        """
        Run the evaluator against sample data using the Fireworks preview API

        Args:
            sample_file: Path to the JSONL file with samples
            max_samples: Maximum number of samples to process

        Returns:
            EvaluatorPreviewResult containing the preview results
        """
        if not self.code_files:
            raise ValueError(
                "No code files loaded. Load metric folder(s) first."
            )

        if "main.py" not in self.code_files and not any(
            k.endswith("/main.py") for k in self.code_files
        ):
            raise ValueError("No main.py found in code files")

        samples = self.load_samples_from_jsonl(sample_file, max_samples)
        if not samples:
            raise ValueError(f"No valid samples found in {sample_file}")

        # Get authentication information
        # Get authentication information
        account_id = get_fireworks_account_id()
        auth_token = get_fireworks_api_key()
        logger.debug("Show account id", account_id)

        if not account_id or not auth_token:
            logger.error(
                "Authentication error: Missing Fireworks Account ID or API Key. "
                "Please set FIREWORKS_ACCOUNT_ID and FIREWORKS_API_KEY environment variables, "
                "or configure them in ~/.fireworks/auth.ini."
            )
            raise ValueError(
                "Missing Fireworks Account ID or API Key. "
                "Ensure FIREWORKS_ACCOUNT_ID and FIREWORKS_API_KEY are set "
                "or configured in ~/.fireworks/auth.ini."
            )
        try:
            # account_id, auth_token = self._get_authentication() # Original call
            pass  # Now handled above
        except (
            ValueError
        ) as e:  # This specific except block might be less relevant if direct checks are made
            logger.error(f"Authentication error: {str(e)}")
            raise

        # Construct the evaluator payload
        # Construct the preview evaluation payload

        # Per user feedback, always set multiMetrics: true and skipRollup: true in the payload
        # to match the behavior observed as "correct" from the TypeScript client.

        evaluator = {
            "displayName": self.display_name or "Preview Evaluator",
            "description": self.description or "Preview Evaluator",
            "multiMetrics": True,  # Directly hardcoded to True
            "criteria": self._construct_criteria(),
            "requirements": "",
            "rollupSettings": {"skipRollup": True},  # Directly hardcoded
        }

        # The samples need to be passed as JSON strings in an array
        sample_strings = [json.dumps(sample) for sample in samples]

        payload = {
            "evaluator": evaluator,
            "sampleData": sample_strings,
            "maxSamples": max_samples,
        }

        # Make API request to preview evaluator
        api_base = os.environ.get(
            "FIREWORKS_API_BASE", "https://api.fireworks.ai"
        )

        # For dev environment, special handling for account_id
        if "dev.api.fireworks.ai" in api_base and account_id == "fireworks":
            account_id = "pyroworks-dev"  # Default dev account

        url = f"{api_base}/v1/accounts/{account_id}/evaluators:previewEvaluator"
        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }

        logger.info(
            f"Previewing evaluator using API endpoint: {url} with account: {account_id}"
        )

        # Log the request details for debugging
        logger.debug(f"Preview API Request URL: {url}")
        logger.debug(
            f"Preview API Request Headers: {json.dumps(headers, indent=2)}"
        )
        logger.debug(
            f"Preview API Request Payload: {json.dumps(payload, indent=2)}"
        )

        global used_preview_api
        try:
            # Make the API call to preview the evaluator
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

            # API call successful
            used_preview_api = True

            # Convert API response to EvaluatorPreviewResult
            preview_result = EvaluatorPreviewResult()
            preview_result.total_samples = result.get(
                "totalSamples", len(samples)
            )
            preview_result.total_runtime_ms = int(
                result.get("totalRuntimeMs", 0)
            )

            # Process individual sample results
            sample_results = result.get("results", [])
            for i, sample_result in enumerate(sample_results):
                preview_result.add_result(
                    sample_index=i,
                    success=sample_result.get("success", False),
                    score=sample_result.get("score", 0.0),
                    per_metric_evals=sample_result.get("perMetricEvals", {}),
                )

            return preview_result

        except Exception as e:
            logger.error(f"Error previewing evaluator: {str(e)}")
            if isinstance(e, requests.exceptions.HTTPError) and hasattr(
                e, "response"
            ):
                logger.error(f"Response: {e.response.text}")

            # Set flag to indicate fallback mode was used
            used_preview_api = False

            # Fallback to the old simulation-based preview
            logger.warning("Falling back to simulated preview mode")
            return self._simulated_preview(samples)

    def _simulated_preview(self, samples):
        """
        Simulate the preview locally without calling the API
        For fallback when the API call fails

        Args:
            samples: List of sample data

        Returns:
            EvaluatorPreviewResult with simulated results
        """
        preview_result = EvaluatorPreviewResult()
        preview_result.total_samples = len(samples)

        start_time = time.time()
        for i, sample in enumerate(samples):
            try:
                # Sample validation
                if "messages" not in sample:
                    raise ValueError(
                        f"Sample {i + 1} is missing 'messages' field"
                    )

                # We validate sample format but in this simulation we don't use these directly
                # In a real implementation, these would be passed to the evaluate function
                _ = sample.get("messages", [])
                _ = sample.get("original_messages", [])
                _ = sample.get("tools", [])

                # Additional kwargs would be passed to evaluate in a real implementation
                _ = {
                    k: v
                    for k, v in sample.items()
                    if k not in ["messages", "original_messages", "tools"]
                }

                # Simple simulation of metric evaluation
                if self.multi_metrics:
                    per_metric_evals = {
                        "quality": 0.8,
                        "relevance": 0.7,
                        "safety": 0.9,
                    }
                else:
                    per_metric_evals = {
                        metric_name: 0.75 for metric_name in self.metric_folders
                    }

                # Calculate an aggregate score
                score = sum(per_metric_evals.values()) / len(per_metric_evals)

                preview_result.add_result(
                    sample_index=i,
                    success=True,
                    score=score,
                    per_metric_evals=per_metric_evals,
                )

            except Exception as e:
                logger.error(f"Error processing sample {i + 1}: {str(e)}")
                preview_result.add_result(
                    sample_index=i,
                    success=False,
                    score=0.0,
                    per_metric_evals={"error": str(e)},
                )

        end_time = time.time()
        # Calculate runtime in milliseconds
        preview_result.total_runtime_ms = max(
            1, int((end_time - start_time) * 1000)
        )

        return preview_result

    def create(
        self, evaluator_id, display_name=None, description=None, force=False
    ):
        """
        Create the evaluation on the Fireworks platform

        Args:
            evaluator_id: ID for the evaluator
            display_name: Display name for the evaluator
            description: Description of the evaluator
            force: If True, update the evaluator if it already exists

        Returns:
            The created evaluator object
        """
        if not self.code_files:
            raise ValueError(
                "No code files loaded. Load metric folder(s) first."
            )

        # Authentication
        account_id = get_fireworks_account_id()
        auth_token = get_fireworks_api_key()

        if (
            not auth_token or not account_id
        ):  # Simplified check, length check can be added if necessary
            logger.error(
                "Authentication error: API credentials appear to be invalid or incomplete. "
                "Please set FIREWORKS_ACCOUNT_ID and FIREWORKS_API_KEY environment variables, "
                "or configure them in ~/.fireworks/auth.ini."
            )
            raise ValueError(
                "Invalid or missing API credentials. Please set valid FIREWORKS_API_KEY and "
                "FIREWORKS_ACCOUNT_ID environment variables or configure ~/.fireworks/auth.ini"
            )
        # try: # Original try-except might not be needed if checks are direct
        # account_id, auth_token = self._get_authentication() # Original call

        # Verify API key format is valid
        # if not auth_token or len(auth_token) < 10 or not account_id: # Length check removed for brevity, can be re-added
        # logger.error("API credentials appear to be invalid or incomplete")
        # raise ValueError(
        # "Invalid or missing API credentials. Please set valid FIREWORKS_API_KEY and "
        # "FIREWORKS_ACCOUNT_ID environment variables or configure ~/.fireworks/auth.ini"
        # )
        # except ValueError as e:
        # logger.error(f"Authentication error: {str(e)}")
        # raise

        # Set display name and description
        self.display_name = display_name or evaluator_id
        self.description = (
            description or f"Evaluator created from {evaluator_id}"
        )

        # Always use the new API payload format

        # Per user feedback, always set multiMetrics: true and skipRollup: true in the payload.

        payload = {
            "evaluator": {
                "displayName": self.display_name,
                "description": self.description,
                "multiMetrics": True,  # Directly hardcoded to True
                "criteria": self._construct_criteria(),  # This will now always return the new format
                "requirements": "",  # Ensure this is an empty string if not used, or appropriate value
                "rollupSettings": {"skipRollup": True},  # Directly hardcoded
            },
            "evaluatorId": evaluator_id,
        }

        # Make API request to create evaluator
        # For dev environment, special handling for account_id
        if (
            "dev.api.fireworks.ai" in self.api_base
            and account_id == "fireworks"
        ):
            account_id = "pyroworks-dev"  # Default dev account

        base_url = f"{self.api_base}/v1/accounts/{account_id}/evaluators"
        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }

        logger.info(
            f"Creating evaluator '{evaluator_id}' for account '{account_id}'..."
        )

        # Make real API call
        try:
            if force:
                # First check if the evaluator exists
                check_url = f"{base_url}/{evaluator_id}"

                try:
                    # Check if the evaluator exists
                    check_response = requests.get(check_url, headers=headers)

                    if check_response.status_code == 200:
                        # Evaluator exists, delete it first then recreate
                        logger.info(
                            f"Evaluator '{evaluator_id}' already exists, deleting and recreating..."
                        )
                        delete_url = f"{base_url}/{evaluator_id}"

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
                            logger.warning(
                                f"Error deleting evaluator: {str(e)}"
                            )

                        # Now create it
                        response = requests.post(
                            base_url, json=payload, headers=headers
                        )
                    else:
                        # Evaluator doesn't exist, create it
                        response = requests.post(
                            base_url, json=payload, headers=headers
                        )
                except requests.exceptions.RequestException:
                    # If checking fails, try to create it
                    response = requests.post(
                        base_url, json=payload, headers=headers
                    )
            else:
                # Just try to create it
                response = requests.post(
                    base_url, json=payload, headers=headers
                )

            response.raise_for_status()
            result = response.json()

            logger.info(f"Successfully created evaluator '{evaluator_id}'")
            return result
        except Exception as e:
            logger.error(f"Error creating evaluator: {str(e)}")
            if isinstance(e, requests.exceptions.HTTPError) and hasattr(
                e, "response"
            ):
                logger.error(f"Response: {e.response.text}")
            raise

    def _construct_criteria(self):
        """
        Construct the assertions for the evaluation

        Returns:
            List of assertion objects
        """
        assertions = []
        if self.ts_mode_config:
            # TypeScript-like mode: single code snippet
            python_code = self.ts_mode_config.get("python_code")
            file_name = self.ts_mode_config.get("file_name", "main.py")
            criterion_name = self.ts_mode_config.get(
                "criterion_name", "default_code_criterion"
            )
            description = self.ts_mode_config.get(
                "description", "Python code execution"
            )

            if not python_code:
                raise ValueError("python_code is required in ts_mode_config")

            assertions.append(
                {
                    "type": "CODE_SNIPPETS",  # Matches Criterion_Type.CODE_SNIPPETS
                    "name": criterion_name,
                    "description": description,
                    "codeSnippets": {
                        "language": "python",
                        "fileContents": {file_name: python_code},
                    },
                }
            )
        elif self.multi_metrics:
            # Original multi_metrics mode: load all .py files from a single folder
            file_contents = {}
            for filename, content in self.code_files.items():
                # Skip files that aren't Python
                if not filename.endswith(".py"):
                    continue
                file_contents[filename] = self._update_evaluate_signature(
                    content
                )

            assertions.append(
                {
                    "codeSnippets": {
                        "language": "python",
                        "fileContents": file_contents,
                    },
                    "name": "eval",  # Default name for multi-metric criterion
                    "type": "CODE_SNIPPETS",
                    "description": self.description
                    or "Multi-metric evaluation",
                }
            )
        else:
            # Construct individual criteria for each metric folder
            for metric_name in self.metric_folders:
                file_contents = {}
                for k, v in self.code_files.items():
                    if k.startswith(f"{metric_name}/"):
                        # Ensure k is a path-like string before splitting
                        if isinstance(k, str) and "/" in k:
                            simple_name = k.split("/", 1)[1]
                        else:
                            simple_name = k  # Fallback if no slash
                        file_contents[simple_name] = (
                            self._update_evaluate_signature(v)
                        )

                if not file_contents:  # Skip if no files found for this metric
                    logger.warning(
                        f"No Python files found for metric '{metric_name}', skipping its criterion."
                    )
                    continue

                assertions.append(
                    {
                        "codeSnippets": {
                            "language": "python",
                            "fileContents": file_contents,
                        },
                        "name": metric_name,
                        "type": "CODE_SNIPPETS",
                        "description": f"Metric: {metric_name}",
                    }
                )

        if not assertions:
            # This case should ideally be prevented by checks in load_metric_folder etc.
            # but as a safeguard:
            # Ensure this check is only for non-ts_mode_config cases
            if not self.ts_mode_config and not assertions:
                raise ValueError(
                    "No valid criteria could be constructed. Ensure metric folders contain Python files or ts_mode_config is correctly set."
                )

        return assertions

    def _update_evaluate_signature(self, content):
        """
        Update the evaluate function signature to the new format

        Args:
            content: The code content to update

        Returns:
            Updated code content
        """
        import re

        # Simple regex to match the old evaluate function signature
        old_pattern = r"def\s+evaluate\s*\(\s*entry\s*(?::\s*dict)?\s*\)"
        new_signature = "def evaluate(messages, original_messages=None, tools=None, **kwargs)"

        # Check if the old pattern exists
        if re.search(old_pattern, content):
            # Replace the old signature with the new one
            updated_content = re.sub(
                old_pattern, new_signature, content, count=1
            )

            # Also add a compatibility layer at the beginning of the function
            compat_layer = """
    # Compatibility layer for old format
    if original_messages is None:
        original_messages = messages
    entry = {"messages": messages, "original_messages": original_messages, "tools": tools}
    entry.update(kwargs)
"""

            # Find the function body indent level
            func_match = re.search(
                r"def\s+evaluate.*?:\s*\n(\s+)", updated_content, re.DOTALL
            )
            if func_match:
                indent = func_match.group(1)
                # Adjust indentation of compatibility layer
                compat_layer = "\n".join(
                    indent + line for line in compat_layer.strip().split("\n")
                )

                # Insert compatibility layer after function definition
                updated_content = re.sub(
                    re.escape(new_signature) + r"\s*:",
                    new_signature + ":" + compat_layer,
                    updated_content,
                    count=1,
                )

            return updated_content

        return content

    def _get_combined_code(self):
        """
        Combine all code files into a single code block

        Returns:
            A string containing all code
        """
        code_parts = []

        # Start with imports
        code_parts.append("import json\nimport sys\n\n")

        # Add all code files
        for filename, content in self.code_files.items():
            # Skip files that aren't Python
            if not filename.endswith(".py"):
                continue

            # Add file as a section
            simple_name = (
                filename.split("/")[-1] if "/" in filename else filename
            )
            updated_content = self._update_evaluate_signature(content)
            code_parts.append(f"# From {simple_name}\n{updated_content}\n\n")

        # Add the wrapper code for handling input/output
        code_parts.append(
            """
# Process input from the evaluation system
if __name__ == '__main__':
    import json
    try:
        input_data = json.loads(sys.stdin.read())
        messages = input_data.get('messages', [])
        original_messages = input_data.get('original_messages', messages)
        tools = input_data.get('tools', [])
        result = evaluate(messages, original_messages, tools, **{k: v for k, v in input_data.items()
                                                            if k not in ['messages', 'original_messages', 'tools']})
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
"""
        )

        return "".join(code_parts)

    def _get_code_from_files(self, files):
        """
        Convert a dict of files into a single code block

        Args:
            files: Dict mapping filenames to content

        Returns:
            A string containing all code
        """
        code_parts = []

        # Start with imports
        code_parts.append("import json\nimport sys\n\n")

        # Add all files
        for filename, content in files.items():
            # Skip files that aren't Python
            if not filename.endswith(".py"):
                continue

            # Add file as a section
            updated_content = self._update_evaluate_signature(content)
            code_parts.append(f"# From {filename}\n{updated_content}\n\n")

        # Add the wrapper code for handling input/output
        code_parts.append(
            """
# Process input from the evaluation system
if __name__ == '__main__':
    import json
    try:
        input_data = json.loads(sys.stdin.read())
        messages = input_data.get('messages', [])
        original_messages = input_data.get('original_messages', messages)
        tools = input_data.get('tools', [])
        result = evaluate(messages, original_messages, tools, **{k: v for k, v in input_data.items()
                                                            if k not in ['messages', 'original_messages', 'tools']})
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
"""
        )

        return "".join(code_parts)

    def _get_authentication(self):
        """
        Get authentication information for the Fireworks API.
        This method now uses the centralized functions from reward_kit.auth.

        Returns:
            Tuple of (account_id, auth_token)

        Raises:
            ValueError: If either account_id or auth_token couldn't be found.
        """
        account_id = get_fireworks_account_id()
        auth_token = get_fireworks_api_key()

        if not account_id:
            logger.error(
                "Authentication error: Fireworks Account ID not found. "
                "Please set the FIREWORKS_ACCOUNT_ID environment variable "
                "or configure it in ~/.fireworks/auth.ini."
            )
            raise ValueError(
                "Fireworks Account ID not found. Set FIREWORKS_ACCOUNT_ID environment variable "
                "or configure it in ~/.fireworks/auth.ini."
            )

        if not auth_token:
            logger.error(
                "Authentication error: Fireworks API Key not found. "
                "Please set the FIREWORKS_API_KEY environment variable "
                "or configure it in ~/.fireworks/auth.ini."
            )
            raise ValueError(
                "Fireworks API Key not found. Set FIREWORKS_API_KEY environment variable "
                "or configure it in ~/.fireworks/auth.ini."
            )

        return account_id, auth_token


# Helper functions for CLI commands
def preview_evaluation(
    # Parameters for folder-based evaluation
    metric_folders: Optional[List[str]] = None,
    multi_metrics: bool = False,  # For folder-based multi-metrics
    folder: Optional[str] = None,  # For folder-based multi-metrics
    # Parameters for TypeScript-like single code snippet evaluation
    python_code_to_evaluate: Optional[str] = None,
    python_file_name_for_code: str = "main.py",  # Default filename for the snippet
    criterion_name_for_code: str = "default_code_criterion",  # Default criterion name
    criterion_description_for_code: str = "Python code execution",  # Default criterion description
    # Common parameters
    sample_file: Optional[str] = None,
    max_samples: int = 5,
    huggingface_dataset: Optional[str] = None,
    huggingface_split: str = "train",
    huggingface_message_key_map: Optional[Dict[str, str]] = None,
    huggingface_response_key: str = "response",
    huggingface_prompt_key: str = "prompt",
):
    """
    Preview an evaluation with sample data. Supports folder-based or direct Python code evaluation.

    Args:
        metric_folders: List of METRIC_NAME=folder_path pairs (for folder-based).
        multi_metrics: Whether to use multi-metrics mode (for folder-based).
        folder: Path to folder with multiple metrics (for folder-based multi_metrics mode).
        python_code_to_evaluate: A string containing Python code to evaluate (for TS-like mode).
        python_file_name_for_code: Filename to associate with `python_code_to_evaluate`.
        criterion_name_for_code: Name for the criterion created for `python_code_to_evaluate`.
        criterion_description_for_code: Description for the criterion.
        sample_file: Path to sample JSONL file.
        max_samples: Maximum number of samples to process.
        huggingface_dataset: Optional HuggingFace dataset name.
        huggingface_split: Dataset split to use.
        huggingface_message_key_map: Optional mapping of dataset keys.
        huggingface_response_key: Key in the dataset for the response.
        huggingface_prompt_key: Key in the dataset for the prompt.

    Returns:
        EvaluatorPreviewResult with preview results.
    """
    ts_mode_config = None
    if python_code_to_evaluate:
        if metric_folders or folder or multi_metrics:
            raise ValueError(
                "Cannot use python_code_to_evaluate with folder-based parameters "
                "(metric_folders, folder, multi_metrics)."
            )
        ts_mode_config = {
            "python_code": python_code_to_evaluate,
            "file_name": python_file_name_for_code,
            "criterion_name": criterion_name_for_code,
            "description": criterion_description_for_code,
        }
        # When python_code_to_evaluate is used, multi_metrics in Evaluator constructor is effectively True
        # due to how ts_mode_config is handled (sets self.multi_metrics = True).
        evaluator = Evaluator(ts_mode_config=ts_mode_config)
    else:
        # Folder-based evaluation
        evaluator = Evaluator(multi_metrics=multi_metrics)
        if multi_metrics:
            if not folder:
                raise ValueError(
                    "`folder` must be specified when using multi_metrics mode without python_code_to_evaluate."
                )
            evaluator.load_multi_metrics_folder(folder)
        else:
            if not metric_folders:
                raise ValueError(
                    "At least one metric_folder must be specified for non-multi_metrics mode without python_code_to_evaluate."
                )
            for pair in metric_folders:
                if "=" not in pair:
                    raise ValueError(
                        f"Invalid metric-folder format: {pair}. Expected METRIC_NAME=folder_path"
                    )
                metric_name, folder_path = pair.split("=", 1)
                evaluator.load_metric_folder(metric_name, folder_path)

    # If HuggingFace dataset is specified, convert it to JSONL first
    if huggingface_dataset:
        if sample_file:
            logger.warning(
                f"Both sample_file and huggingface_dataset specified. "
                f"Using HuggingFace dataset: {huggingface_dataset}"
            )

        logger.info(
            f"Converting HuggingFace dataset to JSONL: {huggingface_dataset}"
        )
        sample_file = huggingface_dataset_to_jsonl(
            dataset_name=huggingface_dataset,
            split=huggingface_split,
            max_samples=max_samples,
            message_key_map=huggingface_message_key_map,
            response_key=huggingface_response_key,
            prompt_key=huggingface_prompt_key,
        )
        logger.info(f"Converted dataset saved to: {sample_file}")

    if not sample_file:
        raise ValueError(
            "Either sample_file or huggingface_dataset must be specified"
        )

    return evaluator.preview(sample_file, max_samples)


def preview_folder_evaluation(
    evaluator_folder,
    sample_file=None,
    max_samples=5,
    multi_metrics=False,
    huggingface_dataset=None,
    huggingface_split="train",
    huggingface_message_key_map=None,
    huggingface_response_key="response",
    huggingface_prompt_key="prompt",
):
    """
    Preview an evaluation from a folder with sample data.
    This is a more convenient interface that automatically detects the
    folder structure and handles both single and multi-metrics evaluations.

    Args:
        evaluator_folder: Path to the folder containing the evaluator code
        sample_file: Path to the sample JSONL file
        max_samples: Maximum number of samples to process
        multi_metrics: Whether this is a multi-metrics evaluation
        huggingface_dataset: Optional HuggingFace dataset name (e.g., "deepseek-ai/DeepSeek-ProverBench")
        huggingface_split: Dataset split to use (default: "train")
        huggingface_message_key_map: Optional mapping of dataset keys to reward-kit message keys
        huggingface_response_key: Key in the dataset containing the response (default: "response")
        huggingface_prompt_key: Key in the dataset containing the prompt (default: "prompt")

    Returns:
        EvaluatorPreviewResult with preview results
    """

    evaluator_folder = os.path.abspath(evaluator_folder)

    # Check if folder exists
    if not os.path.exists(evaluator_folder):
        raise ValueError(f"Evaluator folder does not exist: {evaluator_folder}")

    if not os.path.isdir(evaluator_folder):
        raise ValueError(f"Not a directory: {evaluator_folder}")

    # Determine if this is a multi-metric evaluator or single-metric
    # Multi-metric evaluator has main.py directly in the folder
    # Single-metric has subdirectories for each metric
    has_main_py = os.path.exists(os.path.join(evaluator_folder, "main.py"))

    # Auto-detect multi_metrics if not specified
    if has_main_py and not multi_metrics:
        # Look for a structure that suggests multi-metrics
        py_files = list(Path(evaluator_folder).glob("*.py"))
        if len(py_files) > 1:
            logger.info(
                "Auto-detecting multi-metrics mode based on folder structure"
            )
            multi_metrics = True

    # Create and load evaluator
    evaluator = Evaluator(multi_metrics=multi_metrics)

    if multi_metrics:
        # Load the folder directly as a multi-metric evaluation
        evaluator.load_multi_metrics_folder(evaluator_folder)
    else:
        # Treat each subdirectory with a main.py as a separate metric
        metric_folders = []

        # Check if the folder itself has a main.py (single metric case)
        if has_main_py:
            metric_name = os.path.basename(evaluator_folder)
            evaluator.load_metric_folder(metric_name, evaluator_folder)
        else:
            # Look for subdirectories with main.py
            for item in os.listdir(evaluator_folder):
                item_path = os.path.join(evaluator_folder, item)
                if os.path.isdir(item_path) and os.path.exists(
                    os.path.join(item_path, "main.py")
                ):
                    metric_name = item
                    evaluator.load_metric_folder(metric_name, item_path)
                    metric_folders.append(f"{metric_name}={item_path}")

        if not evaluator.metric_folders:
            raise ValueError(
                f"No valid metrics found in {evaluator_folder}. Each metric folder must contain a main.py file."
            )

    # If HuggingFace dataset is specified, convert it to JSONL first
    if huggingface_dataset:
        if sample_file:
            logger.warning(
                f"Both sample_file and huggingface_dataset specified. "
                f"Using HuggingFace dataset: {huggingface_dataset}"
            )

        logger.info(
            f"Converting HuggingFace dataset to JSONL: {huggingface_dataset}"
        )
        sample_file = huggingface_dataset_to_jsonl(
            dataset_name=huggingface_dataset,
            split=huggingface_split,
            max_samples=max_samples,
            message_key_map=huggingface_message_key_map,
            response_key=huggingface_response_key,
            prompt_key=huggingface_prompt_key,
        )
        logger.info(f"Converted dataset saved to: {sample_file}")

    if not sample_file:
        raise ValueError(
            "Either sample_file or huggingface_dataset must be specified"
        )

    # Run the preview
    return evaluator.preview(sample_file, max_samples)


def create_evaluation(
    evaluator_id: str,
    # Parameters for folder-based evaluation
    metric_folders: Optional[List[str]] = None,
    multi_metrics: bool = False,  # For folder-based multi-metrics
    folder: Optional[str] = None,  # For folder-based multi-metrics
    # Parameters for TypeScript-like single code snippet evaluation
    python_code_to_evaluate: Optional[str] = None,
    python_file_name_for_code: str = "main.py",
    criterion_name_for_code: str = "default_code_criterion",
    criterion_description_for_code: str = "Python code execution",
    # Common parameters
    display_name: Optional[str] = None,
    description: Optional[str] = None,
    force: bool = False,
    huggingface_dataset: Optional[
        str
    ] = None,  # Note: HF dataset is for preview only currently
    huggingface_split: str = "train",
    huggingface_message_key_map: Optional[Dict[str, str]] = None,
    huggingface_response_key: str = "response",
    huggingface_prompt_key: str = "prompt",
):
    """
    Create an evaluation on the Fireworks platform. Supports folder-based or direct Python code evaluation.

    Args:
        evaluator_id: ID for the evaluator.
        metric_folders: List of METRIC_NAME=folder_path pairs (for folder-based).
        multi_metrics: Whether to use multi-metrics mode (for folder-based).
        folder: Path to folder with multiple metrics (for folder-based multi_metrics mode).
        python_code_to_evaluate: A string containing Python code to evaluate (for TS-like mode).
        python_file_name_for_code: Filename to associate with `python_code_to_evaluate`.
        criterion_name_for_code: Name for the criterion created for `python_code_to_evaluate`.
        criterion_description_for_code: Description for the criterion.
        display_name: Display name for the evaluator.
        description: Description of the evaluator.
        force: If True, update the evaluator if it already exists.
        huggingface_dataset: Optional HuggingFace dataset name (currently for preview only).
        huggingface_split: Dataset split to use.
        huggingface_message_key_map: Optional mapping of dataset keys.
        huggingface_response_key: Key in the dataset for the response.
        huggingface_prompt_key: Key in the dataset for the prompt.

    Returns:
        Created evaluator object.
    """
    ts_mode_config = None
    if python_code_to_evaluate:
        if (
            metric_folders or folder or multi_metrics
        ):  # Check original multi_metrics flag
            raise ValueError(
                "Cannot use python_code_to_evaluate with folder-based parameters "
                "(metric_folders, folder, multi_metrics)."
            )
        ts_mode_config = {
            "python_code": python_code_to_evaluate,
            "file_name": python_file_name_for_code,
            "criterion_name": criterion_name_for_code,
            "description": criterion_description_for_code,
        }
        evaluator = Evaluator(ts_mode_config=ts_mode_config)
    else:
        # Folder-based evaluation
        evaluator = Evaluator(
            multi_metrics=multi_metrics
        )  # Pass the original multi_metrics here
        if multi_metrics:  # Check original multi_metrics flag
            if not folder:
                raise ValueError(
                    "`folder` must be specified when using multi_metrics mode without python_code_to_evaluate."
                )
            evaluator.load_multi_metrics_folder(folder)
        else:
            if not metric_folders:
                raise ValueError(
                    "At least one metric_folder must be specified for non-multi_metrics mode without python_code_to_evaluate."
                )
            for pair in metric_folders:
                if "=" not in pair:
                    raise ValueError(
                        f"Invalid metric-folder format: {pair}. Expected METRIC_NAME=folder_path"
                    )
                metric_name, folder_path = pair.split("=", 1)
                evaluator.load_metric_folder(metric_name, folder_path)

    # If using HuggingFace dataset, we need to convert it to JSONL and upload it
    # Currently we only support preview with HF datasets
    # Future work: Handle actual uploads of HF datasets to Fireworks
    if huggingface_dataset:
        logger.info(f"HuggingFace dataset specified: {huggingface_dataset}")
        logger.info(
            "Currently, HuggingFace datasets are supported for evaluation preview only."
        )
        logger.info(
            "To use in full evaluation, first convert to JSONL with huggingface_dataset_to_jsonl()"
        )

        # We could add dataset upload code here in the future

    return evaluator.create(evaluator_id, display_name, description, force)


def deploy_folder_evaluation(
    evaluator_id,
    evaluator_folder,
    display_name=None,
    description=None,
    force=False,
    multi_metrics=False,
    huggingface_dataset=None,
    huggingface_split="train",
    huggingface_message_key_map=None,
    huggingface_response_key="response",
    huggingface_prompt_key="prompt",
):
    """
    Deploy an evaluation from a folder to the Fireworks platform.
    This is a more convenient interface that automatically detects the
    folder structure and handles both single and multi-metrics evaluations.

    Args:
        evaluator_id: ID for the evaluator
        evaluator_folder: Path to the folder containing the evaluator code
        display_name: Display name for the evaluator
        description: Description of the evaluator
        force: If True, update the evaluator if it already exists
        multi_metrics: Whether this is a multi-metrics evaluation (auto-detected if not specified)
        huggingface_dataset: Optional HuggingFace dataset name to use as evaluation data
        huggingface_split: Dataset split to use (default: "train")
        huggingface_message_key_map: Optional mapping of dataset keys to reward-kit message keys
        huggingface_response_key: Key in the dataset containing the response (default: "response")
        huggingface_prompt_key: Key in the dataset containing the prompt (default: "prompt")

    Returns:
        Created evaluator object
    """

    evaluator_folder = os.path.abspath(evaluator_folder)

    # Check if folder exists
    if not os.path.exists(evaluator_folder):
        raise ValueError(f"Evaluator folder does not exist: {evaluator_folder}")

    if not os.path.isdir(evaluator_folder):
        raise ValueError(f"Not a directory: {evaluator_folder}")

    # Determine if this is a multi-metric evaluator or single-metric
    # Multi-metric evaluator has main.py directly in the folder
    # Single-metric has subdirectories for each metric
    has_main_py = os.path.exists(os.path.join(evaluator_folder, "main.py"))

    # Auto-detect multi_metrics if not specified
    if has_main_py and not multi_metrics:
        # Look for a structure that suggests multi-metrics
        py_files = list(Path(evaluator_folder).glob("*.py"))
        if len(py_files) > 1:
            logger.info(
                "Auto-detecting multi-metrics mode based on folder structure"
            )
            multi_metrics = True

    # Default display name if not provided
    if not display_name:
        display_name = evaluator_id

    # Default description if not provided
    if not description:
        description = f"Evaluator '{evaluator_id}' deployed from folder {os.path.basename(evaluator_folder)}"

    # Create and load evaluator
    evaluator = Evaluator(multi_metrics=multi_metrics)

    if multi_metrics:
        # Load the folder directly as a multi-metric evaluation
        evaluator.load_multi_metrics_folder(evaluator_folder)
    else:
        # Treat each subdirectory with a main.py as a separate metric
        metric_folders = []

        # Check if the folder itself has a main.py (single metric case)
        if has_main_py:
            metric_name = os.path.basename(evaluator_folder)
            evaluator.load_metric_folder(metric_name, evaluator_folder)
        else:
            # Look for subdirectories with main.py
            for item in os.listdir(evaluator_folder):
                item_path = os.path.join(evaluator_folder, item)
                if os.path.isdir(item_path) and os.path.exists(
                    os.path.join(item_path, "main.py")
                ):
                    metric_name = item
                    evaluator.load_metric_folder(metric_name, item_path)
                    metric_folders.append(f"{metric_name}={item_path}")

        if not evaluator.metric_folders:
            raise ValueError(
                f"No valid metrics found in {evaluator_folder}. Each metric folder must contain a main.py file."
            )

    # If using HuggingFace dataset, we need to convert it to JSONL and upload it
    # Currently we only support preview with HF datasets
    # Future work: Handle actual uploads of HF datasets to Fireworks
    if huggingface_dataset:
        logger.info(f"HuggingFace dataset specified: {huggingface_dataset}")
        logger.info(
            "Currently, HuggingFace datasets are supported for evaluation preview only."
        )
        logger.info(
            "To use in full evaluation, first convert to JSONL with huggingface_dataset_to_jsonl()"
        )

        # We could add dataset upload code here in the future

    # Deploy the evaluation
    return evaluator.create(evaluator_id, display_name, description, force)
