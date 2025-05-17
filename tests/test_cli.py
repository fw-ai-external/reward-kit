import argparse
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from reward_kit.cli import deploy_command, main, parse_args, preview_command


class TestCLI:
    """Tests for the CLI functionality."""

    def test_parse_args(self):
        """Test the argument parser."""
        # Test preview command
        args = parse_args(["preview", "--samples", "test.jsonl"])
        assert args.command == "preview"
        assert args.samples == "test.jsonl"
        assert args.max_samples == 5  # default value

        # Test deploy command
        args = parse_args(
            ["deploy", "--id", "test-eval", "--metrics-folders", "test=./test"]
        )
        assert args.command == "deploy"
        assert args.id == "test-eval"
        assert args.metrics_folders == ["test=./test"]
        assert not args.force  # default value

    @patch(
        "reward_kit.cli_commands.preview.check_environment", return_value=True
    )  # Patched where preview_command looks it up
    @patch("reward_kit.cli_commands.preview.preview_evaluation")
    def test_preview_command(
        self, mock_preview_eval, mock_preview_check_env
    ):  # Renamed mock args for clarity
        """Test the preview command."""
        # Setup mock
        mock_preview_result = MagicMock()
        mock_preview_result.display = MagicMock()
        mock_preview_eval.return_value = mock_preview_result

        # Create args
        args = argparse.Namespace()
        args.metrics_folders = ["test=./test"]
        args.samples = "test.jsonl"
        args.max_samples = 5
        # Add HuggingFace attributes
        args.huggingface_dataset = None
        args.huggingface_split = "train"
        args.huggingface_prompt_key = "prompt"
        args.huggingface_response_key = "response"
        args.huggingface_key_map = None

        # Mock Path.exists to return True
        with patch(
            "reward_kit.cli_commands.preview.Path.exists", return_value=True
        ):  # Corrected patch target
            # Run the command
            result = preview_command(args)

            # Check result
            assert result == 0
            mock_preview_check_env.assert_called_once()  # Check this specific mock
            mock_preview_eval.assert_called_once_with(
                metric_folders=["test=./test"],
                sample_file="test.jsonl",
                max_samples=5,
                huggingface_dataset=None,
                huggingface_split="train",
                huggingface_prompt_key="prompt",
                huggingface_response_key="response",
                huggingface_message_key_map=None,
            )
            mock_preview_result.display.assert_called_once()

    @patch(
        "reward_kit.cli_commands.deploy.check_environment", return_value=True
    )  # Patched where deploy_command looks it up
    @patch("reward_kit.cli_commands.deploy.create_evaluation")
    def test_deploy_command(
        self, mock_create_eval, mock_deploy_check_env
    ):  # Renamed mock args for clarity
        """Test the deploy command."""
        # Setup mock
        mock_create_eval.return_value = {"name": "test-evaluator"}

        # Create args
        args = argparse.Namespace()
        args.metrics_folders = ["test=./test"]
        args.id = "test-eval"
        args.display_name = "Test Evaluator"
        args.description = "Test description"
        args.force = True
        # Add HuggingFace attributes
        args.huggingface_dataset = None
        args.huggingface_split = "train"
        args.huggingface_prompt_key = "prompt"
        args.huggingface_response_key = "response"
        args.huggingface_key_map = None

        # Run the command
        result = deploy_command(args)

        # Check result
        assert result == 0
        mock_deploy_check_env.assert_called_once()  # Check this specific mock
        mock_create_eval.assert_called_once_with(
            evaluator_id="test-eval",
            metric_folders=["test=./test"],
            display_name="Test Evaluator",
            description="Test description",
            force=True,
            huggingface_dataset=None,
            huggingface_split="train",
            huggingface_message_key_map=None,
            huggingface_prompt_key="prompt",
            huggingface_response_key="response",
        )

    @patch(
        "reward_kit.cli_commands.deploy.check_environment", return_value=False
    )  # Patch for deploy_command's check
    @patch(
        "reward_kit.cli_commands.preview.check_environment", return_value=False
    )  # Patch for preview_command's check
    def test_command_environment_check(
        self, mock_preview_check_env, mock_deploy_check_env
    ):
        """Test that commands check the environment."""
        # Create args with essential attributes needed by the commands
        # before or around the check_environment call.
        preview_args = argparse.Namespace()
        preview_args.metrics_folders = ["test=./test"]  # Added
        preview_args.samples = "test.jsonl"  # Added
        preview_args.max_samples = 1  # Added, as it's often used with samples
        preview_args.huggingface_dataset = (
            None  # Add other relevant args from test_preview_command
        )
        preview_args.huggingface_split = "train"
        preview_args.huggingface_prompt_key = "prompt"
        preview_args.huggingface_response_key = "response"
        preview_args.huggingface_key_map = None

        deploy_args = argparse.Namespace()
        deploy_args.metrics_folders = ["test=./test"]  # Added
        deploy_args.id = "test-eval"  # Added
        deploy_args.display_name = (
            None  # Add other relevant args from test_deploy_command
        )
        deploy_args.description = None
        deploy_args.force = False
        deploy_args.huggingface_dataset = None
        deploy_args.huggingface_split = "train"
        deploy_args.huggingface_prompt_key = "prompt"
        deploy_args.huggingface_response_key = "response"
        deploy_args.huggingface_key_map = None

        # Run the commands
        # Assuming preview_command and deploy_command are still the correct entry points from reward_kit.cli
        # and that they correctly call the underlying functions from cli_commands.
        preview_result = preview_command(preview_args)
        deploy_result = deploy_command(deploy_args)

        # Both should fail if environment check fails
        assert preview_result == 1
        assert deploy_result == 1
        mock_preview_check_env.assert_called_once()
        mock_deploy_check_env.assert_called_once()
