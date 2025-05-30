import json
from unittest.mock import MagicMock, patch

import pytest

# Module to be tested
from reward_kit.cli_commands.deploy import deploy_command
from reward_kit.platform_api import PlatformAPIError  # Import for exception testing


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
        self.huggingface_key_map = None
        self.remote_url = None
        # For GCP
        self.target = "fireworks"
        self.function_ref = None
        self.gcp_project = None
        self.gcp_region = None
        self.gcp_ar_repo = None
        self.service_account = None
        self.entry_point = "reward_function"
        self.runtime = "python311"
        self.gcp_auth_mode = None  # Added gcp_auth_mode
        self.__dict__.update(kwargs)


@pytest.fixture
def mock_check_environment():
    with patch(
        "reward_kit.cli_commands.deploy.check_environment", return_value=True
    ) as mock_check:
        yield mock_check


@pytest.fixture
def mock_gcp_tools():
    # Note: Removed trailing backslashes from the with statement lines
    # Patching where the functions are looked up (in cli_commands.deploy module)
    with patch(
        "reward_kit.cli_commands.deploy.ensure_artifact_registry_repo_exists"
    ) as mock_ensure_repo, patch(
        "reward_kit.cli_commands.deploy.generate_dockerfile_content"
    ) as mock_gen_dockerfile, patch(
        "reward_kit.cli_commands.deploy.build_and_push_docker_image"
    ) as mock_build_push, patch(
        "reward_kit.cli_commands.deploy.deploy_to_cloud_run"
    ) as mock_deploy_run, patch(
        "reward_kit.cli_commands.deploy.ensure_gcp_secret"
    ) as mock_ensure_gcp_secret:

        mock_ensure_repo.return_value = True
        mock_gen_dockerfile.return_value = "DOCKERFILE CONTENT"
        mock_build_push.return_value = True
        mock_deploy_run.return_value = "http://mock-cloud-run-url.com/service"
        mock_ensure_gcp_secret.return_value = (
            "projects/test-proj/secrets/mocksecret/versions/1"  # Mock success
        )
        yield {
            "ensure_repo": mock_ensure_repo,
            "gen_dockerfile": mock_gen_dockerfile,
            "build_push": mock_build_push,
            "deploy_run": mock_deploy_run,
            "ensure_gcp_secret": mock_ensure_gcp_secret,  # Add to yielded dict
        }


class TestDeployCommandRemoteUrl:

    @patch("reward_kit.cli_commands.deploy.create_evaluation")
    def test_deploy_remote_url_success(
        self, mock_create_evaluation_call, mock_check_environment, capsys
    ):
        """Test successful registration of a remote URL via create_evaluation."""
        args = MockArgs(
            id="my-remote-eval",
            remote_url="http://my-evaluator.com/evaluate",
            display_name="My Remote Eval",
            description="A cool remote evaluator.",
        )
        mock_create_evaluation_call.return_value = {
            "name": args.id,
            "id": args.id,
            "config": {"remote_url": args.remote_url},
        }

        return_code = deploy_command(args)
        assert return_code == 0

        mock_create_evaluation_call.assert_called_once_with(
            evaluator_id=args.id,
            remote_url=args.remote_url,
            display_name=args.display_name or args.id,
            description=args.description
            or f"Remote proxy evaluator for {args.id} at {args.remote_url}",
            force=args.force,
            huggingface_dataset=args.huggingface_dataset,
            huggingface_split=args.huggingface_split,
            huggingface_message_key_map=None,
            huggingface_prompt_key=args.huggingface_prompt_key,
            huggingface_response_key=args.huggingface_response_key,
        )

        captured = capsys.readouterr()
        assert (
            f"Deploying evaluator '{args.id}' configured to proxy to remote URL: {args.remote_url}"
            in captured.out
        )
        assert (
            f"Successfully created/updated evaluator '{args.id}' to proxy to {args.remote_url}"
            in captured.out
        )

    @patch("reward_kit.cli_commands.deploy.create_evaluation")
    def test_deploy_remote_url_with_metrics_folders_warning(
        self, mock_create_eval, mock_check_environment, capsys
    ):
        args = MockArgs(
            id="my-remote-eval",
            remote_url="http://my-evaluator.com/evaluate",
            metrics_folders=["mf=path"],
        )
        mock_create_eval.return_value = {"name": args.id}
        deploy_command(args)
        captured = capsys.readouterr()
        assert (
            "Info: --metrics-folders are not packaged when deploying with --remote-url."
            in captured.out
        )

    def test_deploy_remote_url_invalid_url_format(self, mock_check_environment, capsys):
        args = MockArgs(id="my-eval", remote_url="ftp://invalid.com")
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert "Error: Invalid --remote-url 'ftp://invalid.com'" in captured.out

    @patch("reward_kit.cli_commands.deploy.create_evaluation")
    def test_deploy_remote_url_platform_api_error(
        self, mock_create_eval, mock_check_environment, capsys
    ):
        args = MockArgs(
            id="my-remote-eval-fail", remote_url="http://my-evaluator.com/evaluate"
        )
        mock_create_eval.side_effect = PlatformAPIError(
            "Platform connection failed", status_code=500
        )

        return_code = deploy_command(args)
        assert return_code == 1

        captured = capsys.readouterr()
        assert (
            f"Error deploying remote proxy evaluator '{args.id}': Platform connection failed (Status: 500"
            in captured.out
        )

    @patch("reward_kit.cli_commands.deploy.create_evaluation")
    def test_deploy_remote_url_unexpected_error(
        self, mock_create_eval, mock_check_environment, capsys
    ):
        args = MockArgs(
            id="my-remote-eval-generic-fail",
            remote_url="http://my-evaluator.com/evaluate",
        )
        mock_create_eval.side_effect = Exception("Something broke")

        return_code = deploy_command(args)
        assert return_code == 1

        captured = capsys.readouterr()
        assert (
            f"An unexpected error occurred while deploying remote proxy evaluator '{args.id}': Something broke"
            in captured.out
        )


class TestDeployCommandLocalMode:

    @patch("reward_kit.cli_commands.deploy.create_evaluation")
    def test_deploy_local_mode_success(
        self, mock_create_eval, mock_check_environment, capsys
    ):
        mock_create_eval.return_value = {"name": "my-local-eval", "id": "my-local-eval"}
        args = MockArgs(
            id="my-local-eval",
            metrics_folders=["mf=./path"],
            display_name="My Local Eval",
            description="A local one.",
        )
        return_code = deploy_command(args)
        assert return_code == 0
        expected_hf_message_key_map = None
        mock_create_eval.assert_called_once_with(
            evaluator_id=args.id,
            metric_folders=args.metrics_folders,
            display_name=args.display_name or args.id,
            description=args.description or f"Evaluator: {args.id}",
            force=args.force,
            huggingface_dataset=args.huggingface_dataset,
            huggingface_split=args.huggingface_split,
            huggingface_message_key_map=expected_hf_message_key_map,
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
            "Error: --metrics-folders are required if not using --remote-url and target is not gcp-cloud-run."
            in captured.out
        )

    @patch("reward_kit.cli_commands.deploy.create_evaluation")
    def test_deploy_local_mode_create_evaluation_fails(
        self, mock_create_eval, mock_check_environment, capsys
    ):
        mock_create_eval.side_effect = PlatformAPIError(
            "Platform API error", status_code=503
        )
        args = MockArgs(id="my-local-eval", metrics_folders=["mf=./path"])
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert (
            "Error creating/updating evaluator 'my-local-eval': Platform API error (Status: 503"
            in captured.out
        )

    @patch("reward_kit.cli_commands.deploy.create_evaluation")
    def test_deploy_local_mode_create_evaluation_fails_generic_exception(
        self, mock_create_eval, mock_check_environment, capsys
    ):
        mock_create_eval.side_effect = Exception("Generic error")
        args = MockArgs(id="my-local-eval", metrics_folders=["mf=./path"])
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert (
            "Error creating/updating evaluator 'my-local-eval': Generic error"
            in captured.out
        )


class TestDeployCommandGCPMode:
    @patch("reward_kit.cli_commands.deploy.create_evaluation")
    def test_deploy_gcp_mode_success(
        self,
        mock_create_evaluation_final_step,
        mock_check_environment,
        mock_gcp_tools,
        capsys,
    ):
        args = MockArgs(
            target="gcp-cloud-run",
            id="gcp-eval-test",
            function_ref="my_module.my_func",
            gcp_project="test-proj",
            gcp_region="us-central1",
            gcp_ar_repo="test-repo",
            runtime="python310",
            gcp_auth_mode="api-key",  # Explicitly test api-key mode
        )
        mock_create_evaluation_final_step.return_value = {"name": args.id}

        return_code = deploy_command(args)
        assert return_code == 0

        mock_gcp_tools["ensure_repo"].assert_called_once()
        mock_gcp_tools["gen_dockerfile"].assert_called_once()
        mock_gcp_tools["build_push"].assert_called_once()
        mock_gcp_tools["ensure_gcp_secret"].assert_called_once()  # Assert it was called
        mock_gcp_tools["deploy_run"].assert_called_once()
        mock_create_evaluation_final_step.assert_called_once()

        captured = capsys.readouterr()
        assert f"Deploying evaluator '{args.id}' to GCP Cloud Run..." in captured.out
        assert f"Successfully built and pushed Docker image" in captured.out
        assert (
            f"Successfully deployed to Cloud Run. Service URL: http://mock-cloud-run-url.com/service"
            in captured.out
        )
        assert (
            f"Successfully registered GCP Cloud Run service as evaluator '{args.id}' on Fireworks AI."
            in captured.out
        )

    def test_deploy_gcp_mode_missing_args(self, mock_check_environment, capsys):
        args = MockArgs(target="gcp-cloud-run", id="gcp-eval-incomplete")
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert (
            "Error: --function-ref is required for GCP Cloud Run deployment."
            in captured.out
        )

    @patch(
        "reward_kit.cli_commands.deploy.ensure_artifact_registry_repo_exists",
        return_value=False,
    )
    def test_deploy_gcp_mode_ensure_repo_fails(
        self, mock_ensure_repo_fails, mock_check_environment, capsys
    ):
        args = MockArgs(
            target="gcp-cloud-run",
            id="gcp-eval",
            function_ref="a.b",
            gcp_project="p",
            gcp_region="r",
            gcp_ar_repo="repo",
        )
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert "Failed to ensure Artifact Registry repository" in captured.out

    @patch(
        "reward_kit.cli_commands.deploy.ensure_artifact_registry_repo_exists",
        return_value=True,
    )
    @patch(
        "reward_kit.cli_commands.deploy.generate_dockerfile_content", return_value=None
    )
    def test_deploy_gcp_mode_gen_dockerfile_fails(
        self,
        mock_gen_dockerfile_fails,
        mock_ensure_repo,
        mock_check_environment,
        capsys,
    ):
        args = MockArgs(
            target="gcp-cloud-run",
            id="gcp-eval",
            function_ref="a.b",
            gcp_project="p",
            gcp_region="r",
            gcp_ar_repo="repo",
        )
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert "Failed to generate Dockerfile content. Aborting." in captured.out

    @patch(
        "reward_kit.cli_commands.deploy.ensure_artifact_registry_repo_exists",
        return_value=True,
    )
    @patch(
        "reward_kit.cli_commands.deploy.generate_dockerfile_content",
        return_value="Dockerfile",
    )
    @patch(
        "reward_kit.cli_commands.deploy.build_and_push_docker_image", return_value=False
    )
    def test_deploy_gcp_mode_build_fails(
        self,
        mock_build_fails,
        mock_gen_dockerfile,
        mock_ensure_repo,
        mock_check_environment,
        capsys,
    ):
        args = MockArgs(
            target="gcp-cloud-run",
            id="gcp-eval",
            function_ref="a.b",
            gcp_project="p",
            gcp_region="r",
            gcp_ar_repo="repo",
        )
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert "Failed to build and push Docker image" in captured.out

    @patch(
        "reward_kit.cli_commands.deploy.ensure_artifact_registry_repo_exists",
        return_value=True,
    )
    @patch(
        "reward_kit.cli_commands.deploy.generate_dockerfile_content",
        return_value="Dockerfile",
    )
    @patch(
        "reward_kit.cli_commands.deploy.build_and_push_docker_image", return_value=True
    )
    @patch("reward_kit.cli_commands.deploy.deploy_to_cloud_run", return_value=None)
    @patch(
        "reward_kit.cli_commands.deploy.ensure_gcp_secret",
        return_value="projects/p/secrets/mocksecret/versions/1",
    )
    def test_deploy_gcp_mode_cloud_run_deploy_fails(
        self,
        mock_ensure_gcp_secret_individual,
        mock_deploy_run_fails,
        mock_build_push,
        mock_gen_dockerfile,
        mock_ensure_repo,
        mock_check_environment,
        capsys,
    ):
        args = MockArgs(
            target="gcp-cloud-run",
            id="gcp-eval",
            function_ref="a.b",
            gcp_project="p",
            gcp_region="r",
            gcp_ar_repo="repo",
            gcp_auth_mode="api-key",  # Ensure auth mode is set to trigger ensure_gcp_secret path
        )
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert (
            "Failed to deploy to Cloud Run or retrieve service URL. Aborting."
            in captured.out
        )

    @patch("reward_kit.cli_commands.deploy.create_evaluation")
    @patch(
        "reward_kit.cli_commands.deploy.ensure_gcp_secret",
        return_value="projects/p/secrets/mocksecret/versions/1",
    )
    def test_deploy_gcp_mode_final_registration_fails_platform_error(
        self,
        mock_ensure_gcp_secret_individual,
        mock_create_evaluation_final_step,
        mock_check_environment,
        mock_gcp_tools,
        capsys,
    ):
        args = MockArgs(
            target="gcp-cloud-run",
            id="gcp-eval-reg-fail",
            function_ref="a.b",
            gcp_project="p",
            gcp_region="r",
            gcp_ar_repo="repo",
            gcp_auth_mode="api-key",  # Ensure auth mode is set
        )
        mock_create_evaluation_final_step.side_effect = PlatformAPIError(
            "Registration failed", status_code=400
        )
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert (
            "Error registering GCP Cloud Run service URL with Fireworks AI: Registration failed (Status: 400"
            in captured.out
        )

    @patch("reward_kit.cli_commands.deploy.create_evaluation")
    @patch(
        "reward_kit.cli_commands.deploy.ensure_gcp_secret",
        return_value="projects/p/secrets/mocksecret/versions/1",
    )
    def test_deploy_gcp_mode_final_registration_fails_generic_error(
        self,
        mock_ensure_gcp_secret_individual,
        mock_create_evaluation_final_step,
        mock_check_environment,
        mock_gcp_tools,
        capsys,
    ):
        args = MockArgs(
            target="gcp-cloud-run",
            id="gcp-eval-reg-fail-gen",
            function_ref="a.b",
            gcp_project="p",
            gcp_region="r",
            gcp_ar_repo="repo",
            gcp_auth_mode="api-key",  # Ensure auth mode is set
        )
        mock_create_evaluation_final_step.side_effect = Exception(
            "Unexpected registration issue"
        )
        return_code = deploy_command(args)
        assert return_code == 1
        captured = capsys.readouterr()
        assert (
            "An unexpected error occurred during Fireworks AI registration: Unexpected registration issue"
            in captured.out
        )
