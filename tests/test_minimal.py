"""
Minimal test for the agent evaluation CLI command.

This is a simple test to verify that the agent evaluation CLI command works
with the minimum required parameters.
"""

import os
import pytest
import tempfile
import json
import subprocess
from pathlib import Path


def test_cli_help():
    """Test that the CLI help message works."""
    result = subprocess.run(
        ["reward-kit", "--help"],
        capture_output=True,
        text=True,
        check=False
    )
    
    # Check that the command ran successfully
    assert result.returncode == 0
    
    # Check that the help message includes the agent-eval command
    assert "agent-eval" in result.stdout


def test_cli_agent_eval_help():
    """Test that the agent-eval help message works."""
    result = subprocess.run(
        ["reward-kit", "agent-eval", "--help"],
        capture_output=True,
        text=True,
        check=False
    )
    
    # Check that the command ran successfully
    assert result.returncode == 0
    
    # Check that the help message includes essential parameters
    help_text = result.stdout
    assert "--task-dir" in help_text or "--dataset" in help_text
    assert "--test-mode" in help_text


def setup_minimal_task_bundle():
    """Create a minimal task bundle for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal tools module
        task_dir = os.path.join(tmpdir, "test_task")
        os.makedirs(task_dir)
        
        # Create a simple tools.py file
        with open(os.path.join(task_dir, "tools.py"), "w") as f:
            f.write("""
from reward_kit.agent import ToolRegistry

# Create tool registry
R = ToolRegistry("test_tools")

@R.tool(description="Echo text", parameters={"text": str})
def echo(text):
    return text
""")
        
        # Create a simple reward.py file
        with open(os.path.join(task_dir, "reward.py"), "w") as f:
            f.write("""
from reward_kit import reward_function, RewardOutput

@reward_function
def evaluate(messages, **kwargs):
    return RewardOutput(
        score=1.0,
        reason="Test evaluation",
        metrics={"test": {"score": 1.0, "reason": "Test"}}
    )
""")
        
        # Create an __init__.py file
        with open(os.path.join(task_dir, "__init__.py"), "w") as f:
            f.write("")
        
        # Create a task.jsonl file
        with open(os.path.join(task_dir, "task.jsonl"), "w") as f:
            f.write(json.dumps({
                "id": "test_task",
                "toolset": "test_task.tools",
                "initial_messages": [
                    {"role": "user", "content": "Hello"}
                ]
            }))
        
        return tmpdir, task_dir


@pytest.mark.skipif(
    os.environ.get("SKIP_CLI_TESTS") == "1",
    reason="CLI tests are disabled"
)
def test_cli_agent_eval_test_mode():
    """Test the agent-eval command in test mode."""
    try:
        tmpdir, task_dir = setup_minimal_task_bundle()
        
        # Add the tmpdir to the Python path
        old_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = f"{tmpdir}:{old_pythonpath}"
        
        # Run the command in test mode
        result = subprocess.run(
            ["reward-kit", "agent-eval", "--task-dir", task_dir, "--test-mode", "--validate-only"],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Check that the command ran successfully
        assert result.returncode == 0
        
        # Check output for expected text
        assert "Validating task bundle" in result.stdout
        
    finally:
        # Restore the Python path
        os.environ["PYTHONPATH"] = old_pythonpath