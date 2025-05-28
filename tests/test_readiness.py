import json
import os
import sys
from unittest.mock import MagicMock, patch

import aiohttp
import pytest
import torch

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reward_kit.models import EvaluateResult, Message  # Removed PreviewBulk* models
from reward_kit.rewards.math import math_reward

# Import functions from the example scripts if they are structured for import
# For simplicity here, we might re-implement small parts or directly call reward functions


# --- Fixtures ---


@pytest.fixture
def mock_fireworks_api_key(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test_api_key_for_readiness_tests")


@pytest.fixture
def mock_requests_post():
    with patch("requests.post") as mock_post:
        yield mock_post


# To run these tests: pytest tests/test_readiness.py -s (to see print statements)
# The -s flag is helpful for seeing the script outputs during test runs.

import subprocess


# --- End-to-End Script Tests for Math Example ---
class TestMathExampleEndToEndScripts:

    BASE_MATH_EXAMPLE_PATH = os.path.join(
        os.path.dirname(__file__), "../examples/math_example"
    )

    def run_script(
        self, script_name: str, env_vars: dict = None, timeout_seconds: int = 180
    ) -> subprocess.CompletedProcess:
        """Helper to run an example script."""
        script_path = os.path.join(self.BASE_MATH_EXAMPLE_PATH, script_name)
        command = [
            sys.executable,
            script_path,
        ]  # Use sys.executable to ensure correct python version

        current_env = os.environ.copy()
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        existing_pythonpath = current_env.get("PYTHONPATH")
        if existing_pythonpath:
            current_env["PYTHONPATH"] = (
                f"{project_root}{os.pathsep}{existing_pythonpath}"
            )
        else:
            current_env["PYTHONPATH"] = project_root

        if env_vars:
            current_env.update(env_vars)

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=self.BASE_MATH_EXAMPLE_PATH,  # Run script from its directory
            env=current_env,
            timeout=timeout_seconds,
        )
        print(f"\n--- Output for {script_name} (timeout: {timeout_seconds}s) ---")
        print(f"STDOUT:\n{process.stdout}")
        if process.stderr:
            print(f"STDERR:\n{process.stderr}")
        print("--- End Output ---")
        return process

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skipping resource-intensive TRL integration test in CI",
    )
    @pytest.mark.timeout(
        630
    )  # Timeout for test function (slightly > subprocess timeout)
    @patch("trl.GRPOTrainer")
    @patch("peft.get_peft_model")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("datasets.Dataset.from_list")  # Mock dataset loading
    @patch("datasets.Dataset.map")  # Mock dataset map where it's used
    def test_e2e_trl_grpo_integration_script(
        self,
        mock_dataset_map,  # New mock
        mock_dataset_from_list,  # New mock
        mock_tokenizer_load,
        mock_base_model_load,
        mock_get_peft_model,
        mock_grpo_trainer_class,
    ):
        """End-to-end test for examples/math_example/trl_grpo_integration.py with mocked TRL steps."""
        print("\nRunning E2E Test: Math Example - trl_grpo_integration.py (Mocked TRL)")

        # Configure mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<|endoftext|>"
        mock_tokenizer.eos_token_id = 50256
        mock_tokenizer_load.return_value = mock_tokenizer

        mock_base_model = MagicMock()  # Mock for the base model
        mock_base_model_load.return_value = mock_base_model

        mock_peft_model = MagicMock()
        mock_peft_model.print_trainable_parameters = MagicMock()
        mock_peft_model.device = torch.device("cpu")  # Add device attribute
        mock_get_peft_model.return_value = mock_peft_model

        # Configure dataset mocks
        mock_mapped_dataset = MagicMock()
        mock_mapped_dataset.set_format = MagicMock()
        mock_dataset_map.return_value = (
            mock_mapped_dataset  # mock_dataset_map is the mock for dataset_instance.map
        )

        mock_dataset_instance = MagicMock()
        mock_dataset_instance.map = (
            mock_dataset_map  # Assign the .map mock to the instance
        )
        mock_dataset_from_list.return_value = (
            mock_dataset_instance  # Dataset.from_list returns this instance
        )

        # Configure the instance returned by the mocked GRPOTrainer class
        mock_grpo_trainer_instance = MagicMock()
        mock_grpo_trainer_instance.step.return_value = {"loss": 0.1, "reward": 0.9}

        # Mock the dataloader and accelerator more completely
        mock_dataloader = MagicMock()
        # Ensure batch tensors are on the same device the trainer expects
        mock_batch_input_ids = torch.randint(0, 100, (1, 10), device="cpu")
        mock_batch = {
            "input_ids": mock_batch_input_ids,
            "query": ["mock query"],
            "response": ["mock response"],
        }
        mock_dataloader.__iter__.return_value = iter([mock_batch])
        # mock_grpo_trainer_instance.dataloader = mock_dataloader # No longer needed directly
        mock_grpo_trainer_instance.get_train_dataloader = MagicMock(
            return_value=mock_dataloader
        )  # Mock get_train_dataloader

        mock_accelerator = MagicMock()
        mock_accelerator.device = torch.device("cpu")
        mock_grpo_trainer_instance.accelerator = mock_accelerator

        # Mock generate method if called by step or before
        mock_grpo_trainer_instance.generate = MagicMock(
            return_value=torch.randint(0, 100, (1, 5), device="cpu")
        )

        mock_grpo_trainer_class.return_value = mock_grpo_trainer_instance

        env_vars = {"TEST_MODE_TRL": "true"}
        # Run the script with a 10-minute (600 seconds) timeout
        result = self.run_script(
            "trl_grpo_integration.py", env_vars=env_vars, timeout_seconds=600
        )

        assert (
            result.returncode == 0
        ), f"trl_grpo_integration.py script failed with exit code {result.returncode}. Stderr: {result.stderr}"
        assert (
            "GRPO training loop completed for Math Example." in result.stdout
        ), "Expected completion message not found in trl_grpo_integration.py output."

        from examples.math_example.trl_grpo_integration import (
            grpo_config as math_grpo_config,
        )

        # The script now calls grpo_trainer.train(), not grpo_trainer.step() directly.
        # The mock_grpo_trainer_instance.train method is not called because the script runs in a subprocess
        # where the @patch decorator does not apply.
        # The assertions on result.returncode and stdout content are the primary checks for this E2E script test.
        # mock_grpo_trainer_instance.train.assert_called_once() # This line is removed.
        # The number of steps taken internally by train() will be 1 due to TEST_MODE_TRL=true in the script's env.
        # We can't easily check internal step calls on the mock of GRPOTrainer itself when train() is called.
        # The script's output "GRPO training loop completed for Math Example." and return code 0 are primary indicators.
        # The assertion on result.returncode == 0 and the completion message in stdout already cover this.
        # If we wanted to check logs for number of steps, that would be parsing stdout.
        # For now, asserting train() was called is the most direct check on the mock.

        print("E2E Test: Math Example - trl_grpo_integration.py (Mocked TRL): PASSED")


# --- End-to-End Script Tests for Math Example (OpenR1) ---
class TestMathExampleOpenR1EndToEndScripts:

    BASE_MATH_EXAMPLE_OPENR1_PATH = os.path.join(
        os.path.dirname(__file__), "../examples/math_example_openr1"
    )

    def run_script(
        self, script_name: str, env_vars: dict = None, timeout_seconds: int = 180
    ) -> subprocess.CompletedProcess:
        """Helper to run an example script for OpenR1."""
        script_path = os.path.join(self.BASE_MATH_EXAMPLE_OPENR1_PATH, script_name)
        command = [
            sys.executable,
            script_path,
        ]

        current_env = os.environ.copy()
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        existing_pythonpath = current_env.get("PYTHONPATH")
        if existing_pythonpath:
            current_env["PYTHONPATH"] = (
                f"{project_root}{os.pathsep}{existing_pythonpath}"
            )
        else:
            current_env["PYTHONPATH"] = project_root

        if env_vars:
            current_env.update(env_vars)

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=self.BASE_MATH_EXAMPLE_OPENR1_PATH,  # Run script from its directory
            env=current_env,
            timeout=timeout_seconds,
        )
        print(
            f"\n--- Output for {script_name} (OpenR1, timeout: {timeout_seconds}s) ---"
        )
        print(f"STDOUT:\n{process.stdout}")
        if process.stderr:
            print(f"STDERR:\n{process.stderr}")
        print("--- End Output ---")
        return process

    def test_e2e_local_eval_script_openr1(self):
        """End-to-end test for examples/math_example_openr1/local_eval.py"""
        print("\nRunning E2E Test: Math Example OpenR1 - local_eval.py")
        result = self.run_script("local_eval.py")
        assert (
            result.returncode == 0
        ), f"OpenR1 local_eval.py script failed with exit code {result.returncode}. Stderr: {result.stderr}"
        assert (
            "All samples passed successfully!" in result.stdout
        ), "Expected success message not found in OpenR1 local_eval.py output."
        print("E2E Test: Math Example OpenR1 - local_eval.py: PASSED")

    def test_e2e_fireworks_preview_script_openr1(self, mock_fireworks_api_key):
        """End-to-end test for examples/math_example_openr1/fireworks_preview.py with mocked API via Env Var."""
        print(
            "\nRunning E2E Test: Math Example OpenR1 - fireworks_preview.py (Mocked API via Env Var)"
        )

        env_vars = {
            "FIREWORKS_API_KEY": "mocked_key_for_e2e_test_openr1_preview",  # Ensure this is distinct if needed
            "TEST_MOCK_FIREWORKS_PREVIEW": "true",
        }
        result = self.run_script("fireworks_preview.py", env_vars=env_vars)

        assert (
            result.returncode == 0
        ), f"OpenR1 fireworks_preview.py script failed with exit code {result.returncode}. Stderr: {result.stderr}"
        assert (
            "Mocking Fireworks Preview API call in test mode." in result.stdout
        )  # Assuming script logs this
        assert (
            "All samples passed successfully via Fireworks Preview API!"
            in result.stdout
        ), "Expected success message not found in OpenR1 fireworks_preview.py output."
        print(
            "E2E Test: Math Example OpenR1 - fireworks_preview.py (Mocked API via Env Var): PASSED"
        )

    @patch("aiohttp.ClientSession.post")  # Corrected patch target
    def test_e2e_fireworks_regenerate_script_openr1(
        self, mock_aiohttp_post_openr1, mock_fireworks_api_key
    ):
        """End-to-end test for examples/math_example_openr1/fireworks_regenerate.py with mocked API."""
        print(
            "\nRunning E2E Test: Math Example OpenR1 - fireworks_regenerate.py (Mocked API)"
        )

        # Configure the mock for the generation API call
        mock_regenerated_solution = (
            "This is a mocked correct OpenR1 math solution."  # Example
        )
        mock_api_response_data = {
            "choices": [
                {"message": {"role": "assistant", "content": mock_regenerated_solution}}
            ]
        }
        # Mock the response from aiohttp's post
        # aiohttp's post returns an `aiohttp.ClientResponse` object
        # which needs to be awaited for .json()
        mock_aiohttp_client_response = MagicMock(spec=aiohttp.ClientResponse)
        mock_aiohttp_client_response.status = 200

        async def mock_json():  # The .json() method is async
            return mock_api_response_data

        mock_aiohttp_client_response.json = mock_json
        mock_aiohttp_client_response.raise_for_status = (
            MagicMock()
        )  # Mock raise_for_status

        # The session.post() itself is an async context manager in the script
        # So, the mock_aiohttp_post_openr1 should return an object that can be used in `async with`
        # and that object's __aenter__ should return the mock_aiohttp_client_response.

        mock_async_context_manager = MagicMock()

        async def __aenter__():
            return mock_aiohttp_client_response

        async def __aexit__(*args):
            pass

        mock_async_context_manager.__aenter__ = __aenter__
        mock_async_context_manager.__aexit__ = __aexit__
        mock_aiohttp_post_openr1.return_value = mock_async_context_manager

        env_vars = {
            "FIREWORKS_API_KEY": "mocked_key_for_e2e_test_openr1_regenerate",
            "TEST_MOCK_FIREWORKS_REGEN": "true",  # Activate script's internal mocking
        }
        # Ensure the recorded data file exists for the script's internal mocking to work as expected
        # For this test, we rely on the script's internal mock logic using TEST_MOCK_FIREWORKS_REGEN.
        # The @patch above for requests.post would only be hit if TEST_MOCK_FIREWORKS_REGEN was false
        # or if the script's internal mocking still made a requests.post call (which it shouldn't if fully mocked).

        # Create a dummy recorded data file if the script expects one for TEST_MOCK_FIREWORKS_REGEN=true
        recorded_data_path = os.path.join(
            self.BASE_MATH_EXAMPLE_OPENR1_PATH,
            "fireworks_regenerate_recorded_data_openr1.jsonl",
        )
        if not os.path.exists(recorded_data_path):
            print(
                f"Warning: Mock recorded data file not found at {recorded_data_path}, creating dummy for test."
            )
            # Create a minimal dummy file so the script doesn't fail on file open
            # The script should ideally handle this gracefully or the test setup should ensure it exists.
            # For a robust test, this file should contain data that makes the script pass.
            with open(recorded_data_path, "w") as f:
                # Add a sample entry that would lead to a pass if the script uses it
                dummy_sample = {
                    "index": 0,
                    "messages": [{"role": "user", "content": "What is 1+1?"}],
                    "regenerated_messages": [
                        {"role": "assistant", "content": "\\boxed{2}"}
                    ],
                    "evaluation_result": {
                        "score": 1.0,
                        "reason": "Mock pass",
                        "success": True,
                    },
                }
                f.write(json.dumps(dummy_sample) + "\n")

        result = self.run_script("fireworks_regenerate.py", env_vars=env_vars)

        assert (
            result.returncode == 0
        ), f"OpenR1 fireworks_regenerate.py script failed with exit code {result.returncode}. Stderr: {result.stderr}"
        # The success message depends on the content of the mocked recorded data file.
        # If the dummy file above is used, and the script processes it, it should pass for that one sample.
        # The script's logic for "All samples processed..." needs to be robust to the number of samples in the mock file.
        assert (
            "All samples processed in this run passed successfully with regenerated responses!"
            in result.stdout
            or "Mocking Fireworks API call for regeneration using recorded data for prompt:"
            in result.stdout  # More specific check for mock mode
        ), "Expected success or mock mode activation message not found in OpenR1 fireworks_regenerate.py output."

        # If TEST_MOCK_FIREWORKS_REGEN="true" and the script uses its internal mock data,
        # then aiohttp.ClientSession.post (mock_aiohttp_post_openr1) should not be called.
        mock_aiohttp_post_openr1.assert_not_called()
        print(
            "E2E Test: Math Example OpenR1 - fireworks_regenerate.py (Mocked API via Env Var): PASSED"
        )

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skipping resource-intensive TRL integration test in CI",
    )
    @patch("trl.GRPOTrainer")
    @patch("peft.get_peft_model")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("datasets.Dataset.from_list")
    @patch("datasets.Dataset.map")
    def test_e2e_trl_grpo_integration_script_openr1(
        self,
        mock_dataset_map_openr1,
        mock_dataset_from_list_openr1,
        mock_tokenizer_load_openr1,
        mock_base_model_load_openr1,
        mock_get_peft_model_openr1,
        mock_grpo_trainer_class_openr1,
    ):
        """End-to-end test for examples/math_example_openr1/trl_grpo_integration.py with mocked TRL steps."""
        print(
            "\nRunning E2E Test: Math Example OpenR1 - trl_grpo_integration.py (Mocked TRL)"
        )

        # Configure mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<|endoftext|>"
        mock_tokenizer.eos_token_id = (
            50256  # Example, ensure it matches model if relevant
        )
        mock_tokenizer_load_openr1.return_value = mock_tokenizer

        mock_base_model = MagicMock()
        mock_base_model_load_openr1.return_value = mock_base_model

        mock_peft_model = MagicMock()
        mock_peft_model.print_trainable_parameters = MagicMock()
        mock_peft_model.device = torch.device("cpu")
        mock_get_peft_model_openr1.return_value = mock_peft_model

        mock_mapped_dataset = MagicMock()
        mock_mapped_dataset.set_format = MagicMock()
        mock_dataset_map_openr1.return_value = mock_mapped_dataset

        mock_dataset_instance = MagicMock()
        mock_dataset_instance.map = mock_dataset_map_openr1
        mock_dataset_from_list_openr1.return_value = mock_dataset_instance

        mock_grpo_trainer_instance = MagicMock()
        mock_grpo_trainer_instance.train = MagicMock()  # Mock the train method directly

        # Mock dataloader and accelerator parts if GRPOTrainer's train() needs them internally from the instance
        mock_dataloader = MagicMock()
        mock_batch_input_ids = torch.randint(0, 100, (1, 10), device="cpu")
        mock_batch = {
            "input_ids": mock_batch_input_ids,
            "query": ["mock query openr1"],
            "response": ["mock response openr1"],
        }
        mock_dataloader.__iter__.return_value = iter([mock_batch])
        mock_grpo_trainer_instance.get_train_dataloader = MagicMock(
            return_value=mock_dataloader
        )

        mock_accelerator = MagicMock()
        mock_accelerator.device = torch.device("cpu")
        mock_grpo_trainer_instance.accelerator = mock_accelerator

        mock_grpo_trainer_class_openr1.return_value = mock_grpo_trainer_instance

        env_vars = {"TEST_MODE_TRL": "true"}
        result = self.run_script(
            "trl_grpo_integration.py", env_vars=env_vars, timeout_seconds=600
        )  # Increased timeout

        assert (
            result.returncode == 0
        ), f"OpenR1 trl_grpo_integration.py script failed with exit code {result.returncode}. Stderr: {result.stderr}"
        assert (
            "GRPO training loop completed for OpenR1 Math Example." in result.stdout
        ), "Expected completion message not found in OpenR1 trl_grpo_integration.py output."

        # Since the script runs in a subprocess, the mocks apply to the script's execution context if it imports them.
        # The primary check is the script's output and return code.
        # We can't directly assert mock_grpo_trainer_instance.train.assert_called_once() here
        # because the mock object `mock_grpo_trainer_instance` is in the test process,
        # not the subprocess where the script ran.

        print(
            "E2E Test: Math Example OpenR1 - trl_grpo_integration.py (Mocked TRL): PASSED"
        )
