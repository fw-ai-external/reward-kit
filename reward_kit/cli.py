import argparse
import os
import sys
import logging
import importlib
import importlib.util
from typing import Optional, List
import typer
import uvicorn

from .server import serve, serve_tunnel
from .models import RewardOutput
from .evaluation import (
    preview_evaluation,
    create_evaluation,
    preview_folder_evaluation,
    deploy_folder_evaluation,
)

# Set up Typer app for modern CLI interface
app = typer.Typer(help="Fireworks Reward Kit CLI")

# Import version from package
try:
    from . import __version__
except ImportError:
    __version__ = "0.1.0"  # Default version if not found

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fireworks Reward Kit CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Serve a reward function")
    serve_parser.add_argument(
        "func_path",
        help="Path to the reward function to serve (e.g., 'module.path:function_name')",
    )
    serve_parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind the server to"
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to"
    )

    # Serve tunnel command
    tunnel_parser = subparsers.add_parser(
        "serve-tunnel", help="Serve a reward function with a tunnel"
    )
    tunnel_parser.add_argument("func_path", help="Path to the reward function to serve")
    tunnel_parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to"
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Test a reward function locally")
    test_parser.add_argument("func_path", help="Path to the reward function to test")
    test_parser.add_argument(
        "--messages", required=True, help="JSON string of messages to test with"
    )

    # Deploy command (placeholder)
    deploy_parser = subparsers.add_parser(
        "deploy", help="Deploy a reward function to Fireworks"
    )
    deploy_parser.add_argument(
        "func_path", help="Path to the reward function to deploy"
    )
    deploy_parser.add_argument(
        "--name", required=True, help="Name for the deployed function"
    )

    # Deploy to Cloud Run (placeholder)
    cloudrun_parser = subparsers.add_parser(
        "deploy-cloudrun", help="Deploy a reward function to Cloud Run"
    )
    cloudrun_parser.add_argument(
        "func_path", help="Path to the reward function to deploy"
    )
    cloudrun_parser.add_argument(
        "--project", required=True, help="Google Cloud project ID"
    )
    cloudrun_parser.add_argument(
        "--name", required=True, help="Name for the Cloud Run service"
    )

    return parser.parse_args()


def validate_function_path(function_path: str):
    """
    Validate and import a function from a module path.

    Args:
        function_path: A string in the format "module.path:function_name" or "file.py:function_name"

    Returns:
        The imported function

    Raises:
        ImportError: If the function could not be imported
    """
    if ":" not in function_path:
        raise ImportError(
            f"Invalid function path format: {function_path}, expected 'module.path:function_name' or 'file.py:function_name'"
        )

    # Split into module/file path and function name
    path, func_name = function_path.split(":", 1)

    # Check if it's a file path (ends with .py)
    if path.endswith(".py"):
        if not os.path.exists(path):
            raise ImportError(f"File not found: {path}")

        # Load module from file path
        module_name = os.path.basename(path).replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None:
            raise ImportError(f"Could not create module spec from {path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
    else:
        # Standard module import
        try:
            module = importlib.import_module(path)
        except ImportError as e:
            raise ImportError(f"Failed to import module {path}: {str(e)}")

    # Get the function from the module
    try:
        func = getattr(module, func_name)
        return func
    except AttributeError:
        raise ImportError(f"Function '{func_name}' not found in module {path}")


@app.command()
def version():
    """Show the version of the reward kit."""
    typer.echo(f"Fireworks Reward Kit v{__version__}")


@app.command()
def serve_app(
    function_path: str = typer.Argument(
        ..., help="Path to the reward function (e.g., 'module.path:function_name')"
    ),
    host: str = typer.Option("0.0.0.0", help="Host to bind the server to"),
    port: int = typer.Option(8000, help="Port to bind the server to"),
):
    """Serve a reward function as an HTTP API."""
    try:
        # Validate the function path
        validate_function_path(function_path)

        # Start the server
        typer.echo(
            f"Starting reward server on {host}:{port} for function {function_path}"
        )
        # Use the serve function imported from server.py
        serve(function_path, host, port)
    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(code=1)


@app.command()
def serve_tunnel_cmd(
    function_path: str = typer.Argument(..., help="Path to the reward function"),
    port: int = typer.Option(8000, help="Port to bind the server to"),
):
    """Serve a reward function with a tunnel for external access."""
    try:
        serve_tunnel(function_path, port)
    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(code=1)


@app.command()
def test(
    function_path: str = typer.Argument(
        ..., help="Path to the reward function to test"
    ),
    messages_json: str = typer.Option(
        ..., "--messages", help="JSON string of messages to test with"
    ),
):
    """Test a reward function locally."""
    import json

    try:
        # Parse the messages
        messages = json.loads(messages_json)

        # Validate and load the function
        func = validate_function_path(function_path)

        # Call the function
        result = func(messages=messages)

        # Print the result
        if isinstance(result, RewardOutput):
            typer.echo(f"Score: {result.score}")
            typer.echo("Metrics:")
            for name, metric in result.metrics.items():
                typer.echo(
                    f"  {name}: {metric.score} ({metric.reason or 'No reason provided'})"
                )
        elif isinstance(result, tuple) and len(result) == 2:
            score, components = result
            typer.echo(f"Score: {score}")
            typer.echo("Components:")
            for name, value in components.items():
                typer.echo(f"  {name}: {value}")
        else:
            typer.echo(f"Invalid result type: {type(result)}", err=True)
            raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error testing reward function: {str(e)}", err=True)
        raise typer.Exit(code=1)


@app.command()
def deploy(
    function_path: str = typer.Argument(
        ..., help="Path to the reward function to deploy"
    ),
    name: str = typer.Option(..., help="Name for the deployed function"),
):
    """Deploy a reward function to Fireworks."""
    typer.echo(f"Deploying {function_path} as {name}...")
    typer.echo(
        "This is a placeholder. In a complete implementation, this would deploy the function to Fireworks."
    )


@app.command()
def deploy_cloudrun(
    function_path: str = typer.Argument(
        ..., help="Path to the reward function to deploy"
    ),
    project: str = typer.Option(..., help="Google Cloud project ID"),
    name: str = typer.Option(..., help="Name for the Cloud Run service"),
):
    """Deploy a reward function to Google Cloud Run."""
    typer.echo(
        f"Deploying {function_path} to Cloud Run as {name} in project {project}..."
    )
    typer.echo(
        "This is a placeholder. In a complete implementation, this would deploy to Google Cloud Run."
    )


@app.command("preview")
def preview_cmd(
    metric_folder: List[str] = typer.Option(
        None, "--metric-folder", help="Format as METRIC_NAME=folder_path"
    ),
    sample_file: str = typer.Option(
        ..., "--sample-file", help="Path to sample JSONL file"
    ),
    multi_metrics: bool = typer.Option(
        False,
        "--multi-metrics",
        help="If set, enables multiple metrics from one folder",
    ),
    folder: str = typer.Option(
        None, "--folder", help="Path to folder with multiple metrics"
    ),
    max_samples: int = typer.Option(
        5, "--max-samples", help="Maximum number of samples to process"
    ),
):
    """Preview an evaluation with sample data (legacy method)."""
    try:
        if not metric_folder and not folder:
            typer.echo(
                "Either --metric-folder or --folder with --multi-metrics must be specified"
            )
            raise typer.Exit(code=1)

        if multi_metrics and not folder:
            typer.echo("--folder must be specified when using --multi-metrics")
            raise typer.Exit(code=1)

        preview_result = preview_evaluation(
            metric_folders=metric_folder,
            multi_metrics=multi_metrics,
            folder=folder,
            sample_file=sample_file,
            max_samples=max_samples,
        )

        # Display the results
        preview_result.display()

    except Exception as e:
        typer.echo(f"Error previewing evaluation: {str(e)}", err=True)
        raise typer.Exit(code=1)


@app.command("preview-folder")
def preview_folder_cmd(
    evaluator_folder: str = typer.Argument(
        ..., help="Path to the folder containing the evaluator code"
    ),
    sample_file: str = typer.Option(
        ..., "--sample-file", help="Path to sample JSONL file"
    ),
    max_samples: int = typer.Option(
        5, "--max-samples", help="Maximum number of samples to process"
    ),
    multi_metrics: bool = typer.Option(
        None, "--multi-metrics", help="If set, forces multi-metrics mode"
    ),
):
    """Preview an evaluation directly from a folder with sample data."""
    try:
        typer.echo(f"Previewing evaluation from folder: {evaluator_folder}")

        preview_result = preview_folder_evaluation(
            evaluator_folder=evaluator_folder,
            sample_file=sample_file,
            max_samples=max_samples,
            multi_metrics=multi_metrics,
        )

        # Display the results
        preview_result.display()

    except Exception as e:
        typer.echo(f"Error previewing folder evaluation: {str(e)}", err=True)
        raise typer.Exit(code=1)


@app.command("create")
def create_cmd(
    eval_id: str = typer.Argument(..., help="ID for the evaluation to create"),
    metric_folder: List[str] = typer.Option(
        None, "--metric-folder", help="Format as METRIC_NAME=folder_path"
    ),
    multi_metrics: bool = typer.Option(
        False,
        "--multi-metrics",
        help="If set, enables multiple metrics from one folder",
    ),
    folder: str = typer.Option(
        None, "--folder", help="Path to folder with multiple metrics"
    ),
    display_name: str = typer.Option(
        None, "--display-name", help="Display name for the evaluation"
    ),
    description: str = typer.Option(
        None, "--description", help="Description of the evaluation"
    ),
):
    """Create an evaluation (legacy method)."""
    try:
        if not metric_folder and not folder:
            typer.echo(
                "Either --metric-folder or --folder with --multi-metrics must be specified"
            )
            raise typer.Exit(code=1)

        if multi_metrics and not folder:
            typer.echo("--folder must be specified when using --multi-metrics")
            raise typer.Exit(code=1)

        evaluator = create_evaluation(
            evaluator_id=eval_id,
            metric_folders=metric_folder,
            multi_metrics=multi_metrics,
            folder=folder,
            display_name=display_name,
            description=description,
        )

        typer.echo(f"Successfully created evaluator: {evaluator['name']}")

    except Exception as e:
        typer.echo(f"Error creating evaluation: {str(e)}", err=True)
        raise typer.Exit(code=1)


@app.command("deploy-folder")
def deploy_folder_cmd(
    eval_id: str = typer.Argument(..., help="ID for the evaluation to create"),
    evaluator_folder: str = typer.Argument(
        ..., help="Path to the folder containing the evaluator code"
    ),
    display_name: str = typer.Option(
        None, "--display-name", help="Display name for the evaluation"
    ),
    description: str = typer.Option(
        None, "--description", help="Description of the evaluation"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="If set, will overwrite an existing evaluator with the same ID",
    ),
    multi_metrics: bool = typer.Option(
        None, "--multi-metrics", help="If set, forces multi-metrics mode"
    ),
):
    """Deploy an evaluation directly from a folder."""
    try:
        typer.echo(f"Deploying evaluation from folder: {evaluator_folder}")

        evaluator = deploy_folder_evaluation(
            evaluator_id=eval_id,
            evaluator_folder=evaluator_folder,
            display_name=display_name,
            description=description,
            force=force,
            multi_metrics=multi_metrics,
        )

        typer.echo(f"Successfully deployed evaluator: {evaluator['name']}")

    except Exception as e:
        typer.echo(f"Error deploying folder evaluation: {str(e)}", err=True)
        raise typer.Exit(code=1)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
