"""
Basic tests for the agent evaluation framework.
"""

import os
import pytest
import asyncio
from pathlib import Path
import tempfile
import json

from reward_kit.agent import (
    ToolRegistry,
    Database,
    AgentEvaluator,
    load_task_from_file,
)


def test_tool_registry_creation():
    """Test that a tool registry can be created."""
    registry = ToolRegistry("test_tools")
    assert registry.name == "test_tools"
    assert len(registry.tools) == 0


def test_tool_registration():
    """Test that tools can be registered."""
    registry = ToolRegistry("test_tools")

    @registry.tool(description="Test tool", parameters={"param1": str})
    def test_tool(param1):
        return param1.upper()

    assert "test_tool" in registry.tools
    assert registry.tools["test_tool"].description == "Test tool"
    assert "param1" in registry.tools["test_tool"].parameters


def test_openai_tools_format():
    """Test that tools can be formatted for OpenAI."""
    registry = ToolRegistry("test_tools")

    @registry.tool(
        description="Test tool", parameters={"param1": str, "param2": int}
    )
    def test_tool(param1, param2):
        return param1.upper()

    tools = registry.get_openai_tools()
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "test_tool"
    assert "param1" in tools[0]["function"]["parameters"]["properties"]
    assert "param2" in tools[0]["function"]["parameters"]["properties"]
    assert (
        tools[0]["function"]["parameters"]["properties"]["param1"]["type"]
        == "string"
    )
    assert (
        tools[0]["function"]["parameters"]["properties"]["param2"]["type"]
        == "integer"
    )


@pytest.mark.asyncio
async def test_tool_execution():
    """Test that tools can be executed."""
    registry = ToolRegistry("test_tools")

    @registry.tool(description="Test tool", parameters={"text": str})
    def capitalize(text):
        return text.upper()

    result = await registry.execute_tool("capitalize", {"text": "hello"})
    assert result == "HELLO"


@pytest.mark.asyncio
async def test_database_setup():
    """Test database setup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = Database(
            db_path,
            seed_sql="CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT);",
        )
        await db.setup()

        # Use a separate connection for executing the query
        conn = await db.get_connection()
        try:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            )
            rows = await cursor.fetchall()
            # Convert to list of dicts
            columns = [col[0] for col in cursor.description]
            result = [dict(zip(columns, row)) for row in rows]
            table_names = [row["name"] for row in result]
            assert "test" in table_names
        finally:
            await conn.close()


@pytest.mark.asyncio
async def test_agent_evaluator_setup():
    """Test agent evaluator setup with minimal toolset."""
    # Create a minimal test task and toolset
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal tools module
        tools_dir = os.path.join(tmpdir, "test_task")
        os.makedirs(tools_dir)

        # Create a simple tools.py file
        with open(os.path.join(tools_dir, "tools.py"), "w") as f:
            f.write(
                """
from reward_kit.agent import ToolRegistry

# Create tool registry
R = ToolRegistry("test_tools")

@R.tool(description="Echo text", parameters={"text": str})
def echo(text):
    return text
"""
            )

        # Create a simple reward.py file
        with open(os.path.join(tools_dir, "reward.py"), "w") as f:
            f.write(
                """
from reward_kit import reward_function
from reward_kit.models import EvaluateResult, MetricResult

@reward_function
def evaluate(messages, **kwargs):
    return {
        "score": 1.0,
        "reason": "Test evaluation",
        "metrics": {
            "test": {
                "score": 1.0,
                "reason": "Test"
            }
        }
    }
"""
            )

        # Create an __init__.py file
        with open(os.path.join(tools_dir, "__init__.py"), "w") as f:
            f.write("")

        # Create a task.jsonl file
        with open(os.path.join(tools_dir, "task.jsonl"), "w") as f:
            f.write(
                json.dumps(
                    {
                        "id": "test_task",
                        "toolset": "test_task.tools",
                        "initial_messages": [
                            {"role": "user", "content": "Hello"}
                        ],
                    }
                )
            )

        # Add the tmpdir to the Python path so we can import the test task
        import sys

        sys.path.insert(0, tmpdir)

        try:
            # Test the agent evaluator setup
            evaluator = AgentEvaluator(
                task_id="test_task",
                toolset_path="test_task.tools",
                reward_path="test_task.reward",
                base_dir=os.path.join(tmpdir, "runs"),
            )

            # Set up the evaluator
            await evaluator.setup()

            # Verify that the tool registry was loaded
            assert evaluator.tool_registry is not None
            assert evaluator.tool_registry.name == "test_tools"
            assert "echo" in evaluator.tool_registry.tools

            # Create a run
            run_id = "test_run"
            run_db_path = await evaluator.create_run(run_id)
            assert os.path.exists(run_db_path)

            # Try to execute a tool
            result = await evaluator.execute_tool(
                run_id, "echo", {"text": "test"}
            )
            assert result == "test"

            # Try to evaluate
            evaluation = await evaluator.evaluate(
                run_id=run_id, messages=[{"role": "user", "content": "Hello"}]
            )
            # Ensure it's a dictionary with expected structure
            assert isinstance(evaluation, dict)

            # Check the nested structure based on what we've observed
            assert "score" in evaluation

            if isinstance(evaluation["score"], dict):
                # Nested structure case
                nested_result = evaluation["score"]
                assert nested_result["score"] == 1.0
                assert "metrics" in nested_result
                assert "test" in nested_result["metrics"]
                assert nested_result["metrics"]["test"]["score"] == 1.0
            else:
                # Direct score case
                assert evaluation["score"] == 1.0

        finally:
            # Clean up
            sys.path.remove(tmpdir)


def test_load_task_from_file():
    """Test loading tasks from a file."""
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".jsonl", delete=False
    ) as f:
        try:
            # Write a simple task
            f.write(
                json.dumps(
                    {
                        "id": "test_task",
                        "toolset": "test_task.tools",
                        "initial_messages": [
                            {"role": "user", "content": "Hello"}
                        ],
                    }
                )
            )
            f.write("\n")
            f.write(
                json.dumps(
                    {
                        "id": "test_task_2",
                        "toolset": "test_task.tools",
                        "initial_messages": [
                            {"role": "user", "content": "Hello 2"}
                        ],
                    }
                )
            )
            f.flush()

            # Load the tasks
            tasks = load_task_from_file(f.name)
            assert len(tasks) == 2
            assert tasks[0]["id"] == "test_task"
            assert tasks[1]["id"] == "test_task_2"
        finally:
            os.unlink(f.name)
