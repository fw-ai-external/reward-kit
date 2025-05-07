import pytest
from unittest.mock import MagicMock, patch
import sys
import os
import argparse

from reward_kit.cli import parse_args, preview_command, deploy_command, main


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

    @patch("reward_kit.cli.preview_evaluation")
    def test_preview_command(self, mock_preview):
        """Test the preview command."""
        # Setup mock
        mock_preview_result = MagicMock()
        mock_preview_result.display = MagicMock()
        mock_preview.return_value = mock_preview_result

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
        with patch("reward_kit.cli.Path.exists", return_value=True):
            # Run the command
            result = preview_command(args)

            # Check result
            assert result == 0
            mock_preview.assert_called_once_with(
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

    @patch("reward_kit.cli.create_evaluation")
    def test_deploy_command(self, mock_create):
        """Test the deploy command."""
        # Setup mock
        mock_create.return_value = {"name": "test-evaluator"}

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
        mock_create.assert_called_once_with(
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

    @patch("reward_kit.cli.check_environment", return_value=False)
    def test_command_environment_check(self, mock_check):
        """Test that commands check the environment."""
        # Create args with huggingface attributes
        preview_args = argparse.Namespace()
        preview_args.huggingface_key_map = None

        deploy_args = argparse.Namespace()
        deploy_args.huggingface_key_map = None

        # Run the commands
        preview_result = preview_command(preview_args)
        deploy_result = deploy_command(deploy_args)

        # Both should fail if environment check fails
        assert preview_result == 1
        assert deploy_result == 1
