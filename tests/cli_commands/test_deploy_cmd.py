import json
from unittest.mock import MagicMock, patch

import pytest

# Module to be tested
from reward_kit.cli_commands import deploy as deploy_cmd_module  # aliasing
from reward_kit.cli_commands.deploy import deploy_command


# --- Mocking argparse.Namespace to simulate parsed CLI arguments ---
class MockArgs:
    def __init__(self, **kwargs):
        self.verbose = False
        self.id = None
        self.metrics_folders = None
        self.display_name = None
        self.description = None
        self.force = False
        self.huggingface_dataset = None
        self.huggingface_split = "train"
        self.huggingface_prompt_key = "prompt"
        self.huggingface_response_key = "response"
        self.huggingface_key_map = None  # This is what args will have
        self.remote_url = None
        self.__dict__.update(kwargs)


@pytest.fixture
def mock_check_environment():
    # Patching where 'check_environment' is looked up by the 'deploy' module.
    with patch(
        "reward_kit.cli_commands.deploy.check_environment", return_value=True
    ) as mock_check:
        yield mock_check


class TestDeployCommandRemoteUrl:

    def test_deploy_remote_url_success(self, mock_check_environment, capsys):
        """Test successful registration of a remote URL (placeholder logic)."""
        args = MockArgs(
            id="my-remote-eval",
            remote_url="http://my-evaluator.com/evaluate",
            display_name="My Remote Eval",
            description="A cool remote evaluator.",
        )
        return_code = deploy_command(args)
        assert return_code == 0
        captured = capsys.readouterr()
        assert (
            f"Registering remote evaluator '{args.id}' with URL: {args.remote_url}"
            in captured.out
        )
        # Corrected assertion based on actual print in deploy_command
        assert (
            f"SUCCESS (Placeholder): Remote evaluator '{args.id}' would be registered with URL '{args.remote_url}'."
            in captured.out
        )
        assert (
            f"Successfully registered/updated remote evaluator: {args.id}"
            in captured.out
        )

    def test_deploy_remote_url_with_metrics_folders_warning(
        self, mock_check_environment, capsys
    ):
        args = MockArgs(
            id="my-remote-eval",
            remote_url="http://my-evaluator.com/evaluate",
            metrics_folders=["mf=path"],
        )
        deploy_command(args)
        captured = capsys.readouterr()
        # Corrected assertion based on actual print in deploy_command
        assert (
            "Info: --metrics-folders are ignored when deploying with --remote-url."
            in captured.out
        )

    def test_deploy_remote_url_with_hf_dataset_warning(
        self, mock_check_environment, capsys
    ):
        args = MockArgs(
            id="my-remote-eval",
            remote_url="http://my-evaluator.com/evaluate",
            huggingface_dataset="test/dataset",
        )
        deploy_command(args)
        captured = capsys.readouterr()
        # Corrected assertion based on actual print in deploy_command
        assert (
            "Info: HuggingFace dataset arguments are ignored when deploying with --remote-url."
            in captured.out
        )

    def test_deploy_remote_url_invalid_url_format(self, mock_check_environment, capsys):
        args = MockArgs(id="my-eval", remote_url="ftp://invalid.com")
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert "Error: Invalid --remote-url 'ftp://invalid.com'" in captured.out


class TestDeployCommandLocalMode:

    @patch("reward_kit.cli_commands.deploy.create_evaluation")  # Corrected patch target
    def test_deploy_local_mode_success(
        self, mock_create_eval, mock_check_environment, capsys
    ):
        """Test successful local deployment."""
        mock_create_eval.return_value = {"name": "my-local-eval", "id": "my-local-eval"}

        args = MockArgs(
            id="my-local-eval",
            metrics_folders=["mf=./path"],
            display_name="My Local Eval",
            description="A local one.",
            # huggingface_key_map is None by default in MockArgs
        )
        return_code = deploy_command(args)
        assert return_code == 0

        # In deploy_command, huggingface_message_key_map will be None if args.huggingface_key_map is None
        expected_hf_message_key_map = None
        if (
            args.huggingface_key_map
        ):  # pragma: no cover (not covered by this specific test case)
            expected_hf_message_key_map = json.loads(args.huggingface_key_map)

        mock_create_eval.assert_called_once_with(
            evaluator_id=args.id,
            metric_folders=args.metrics_folders,
            display_name=args.display_name or args.id,
            description=args.description or f"Evaluator: {args.id}",
            force=args.force,
            huggingface_dataset=args.huggingface_dataset,
            huggingface_split=args.huggingface_split,
            huggingface_message_key_map=expected_hf_message_key_map,  # Use the transformed value
            huggingface_prompt_key=args.huggingface_prompt_key,
            huggingface_response_key=args.huggingface_response_key,
        )
        captured = capsys.readouterr()
        assert "Successfully created/updated evaluator: my-local-eval" in captured.out

    def test_deploy_local_mode_missing_metrics_folders(
        self, mock_check_environment, capsys
    ):
        args = MockArgs(id="my-local-eval")
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert (
            "Error: --metrics-folders are required if not using --remote-url."
            in captured.out
        )

    @patch("reward_kit.cli_commands.deploy.create_evaluation")  # Corrected patch target
    def test_deploy_local_mode_create_evaluation_fails(
        self, mock_create_eval, mock_check_environment, capsys
    ):
        mock_create_eval.side_effect = Exception("Platform API error")
        args = MockArgs(id="my-local-eval", metrics_folders=["mf=./path"])
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert "Error creating/updating evaluator: Platform API error" in captured.out
