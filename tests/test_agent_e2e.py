"""End-to-end tests for the agent evaluation framework."""

import os
import json
import asyncio
import tempfile
import pytest
from pathlib import Path

from reward_kit import reward_function, RewardOutput
from reward_kit.agent import ToolRegistry, AgentEvaluator, load_task_from_file


@pytest.fixture
def tool_registry():
    """Create a tool registry with simple math tools."""
    registry = ToolRegistry("test_tools", "Testing tools")

    @registry.tool(
        description="Add two numbers", parameters={"a": int, "b": int}
    )
    def add(a, b):
        return a + b

    @registry.tool(
        description="Subtract two numbers", parameters={"a": int, "b": int}
    )
    def subtract(a, b):
        return a - b

    return registry


@pytest.fixture
def task_bundle():
    """Create a temporary task bundle with math operations."""
    # Create a temporary directory for the task
    temp_dir = tempfile.mkdtemp()
    module_dir = os.path.join(temp_dir, "math_task")
    os.makedirs(module_dir, exist_ok=True)

    # Create __init__.py
    with open(os.path.join(module_dir, "__init__.py"), "w") as f:
        f.write('"""Math task module for agent evaluation."""\n')

    # Create tools.py with tool registry
    with open(os.path.join(module_dir, "tools.py"), "w") as f:
        f.write(
            """from reward_kit.agent import ToolRegistry

# Create tool registry
R = ToolRegistry("math_tools", "Tools for performing math operations")

@R.tool(description="Add two numbers", parameters={"a": int, "b": int})
def add(a, b):
    return a + b

@R.tool(description="Subtract two numbers", parameters={"a": int, "b": int})
def subtract(a, b):
    return a - b
"""
        )

    # Create reward.py with reward function
    with open(os.path.join(module_dir, "reward.py"), "w") as f:
        f.write(
            """from reward_kit import reward_function, RewardOutput, MetricRewardOutput

@reward_function
def evaluate(messages, *, answer=None, **kwargs):
    if not answer:
        return RewardOutput(
            score=0.0,
            metrics={}
        )
    
    # Check if the last message contains the expected answer
    if not messages or messages[-1].role != "assistant":
        return RewardOutput(
            score=0.0,
            metrics={}
        )
    
    last_message = messages[-1].content
    if str(answer) in last_message:
        return RewardOutput(
            score=1.0,
            metrics={"answer_correct": MetricRewardOutput(score=1.0, reason="Answer matched")}
        )
    else:
        return RewardOutput(
            score=0.0,
            metrics={"answer_correct": MetricRewardOutput(score=0.0, reason="Answer did not match")}
        )
"""
        )

    # Create task.jsonl with test tasks
    task_file = os.path.join(module_dir, "task.jsonl")
    with open(task_file, "w") as f:
        # Task 1: Addition
        f.write(
            json.dumps(
                {
                    "id": "math.add.001",
                    "initial_messages": [
                        {"role": "user", "content": "What is 25 + 17?"}
                    ],
                    "toolset": os.path.basename(temp_dir) + ".math_task.tools",
                    "answer": 42,
                }
            )
            + "\n"
        )

        # Task 2: Subtraction
        f.write(
            json.dumps(
                {
                    "id": "math.subtract.001",
                    "initial_messages": [
                        {"role": "user", "content": "What is 50 - 8?"}
                    ],
                    "toolset": os.path.basename(temp_dir) + ".math_task.tools",
                    "answer": 42,
                }
            )
            + "\n"
        )

    # Add the temp dir to Python path
    import sys

    sys.path.insert(0, os.path.dirname(temp_dir))

    # Return paths for cleanup
    return {
        "dir": temp_dir,
        "module_dir": module_dir,
        "task_file": task_file,
        "module_name": os.path.basename(temp_dir),
    }


@pytest.mark.asyncio
async def test_basic_tool_execution():
    """Test basic tool registry and execution."""
    # Create a simple tool registry
    registry = ToolRegistry("test_tools", "Testing tools")

    @registry.tool(
        description="Add two numbers", parameters={"a": int, "b": int}
    )
    def add(a, b):
        return a + b

    @registry.tool(
        description="Subtract two numbers", parameters={"a": int, "b": int}
    )
    def subtract(a, b):
        return a - b

    # Check tool registration
    assert len(registry.tools) == 2, "Expected 2 tools"
    assert "add" in registry.tools, "Add tool not registered"
    assert "subtract" in registry.tools, "Subtract tool not registered"

    # Check the OpenAI format
    tools_spec = registry.get_openai_tools()
    assert len(tools_spec) == 2, "Expected 2 tools in OpenAI format"

    # Execute the tools
    add_result = await registry.execute_tool("add", {"a": 25, "b": 17})
    assert add_result == 42, "Add tool execution failed"

    subtract_result = await registry.execute_tool("subtract", {"a": 50, "b": 8})
    assert subtract_result == 42, "Subtract tool execution failed"


@pytest.mark.asyncio
async def test_agent_with_direct_db():
    """Test agent database interaction using direct DB operations."""
    # Create a temporary directory for database
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")

    # Create a Database instance
    from reward_kit.agent import Database

    # Initialize the database with a schema
    schema = """
    CREATE TABLE items (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        quantity INTEGER NOT NULL
    );
    
    INSERT INTO items (name, quantity) VALUES ('apple', 5);
    INSERT INTO items (name, quantity) VALUES ('banana', 10);
    INSERT INTO items (name, quantity) VALUES ('orange', 7);
    """

    # Create the database with the schema
    db = Database(db_path, seed_sql=schema)
    await db.setup()

    try:
        # Get a connection
        conn = await db.get_connection()

        # Test reading data
        cursor = await conn.execute(
            "SELECT quantity FROM items WHERE name = ?", ("banana",)
        )
        row = await cursor.fetchone()
        assert row is not None, "Item not found"
        assert row[0] == 10, "Unexpected initial quantity"

        # Test updating data
        await conn.execute(
            "UPDATE items SET quantity = ? WHERE name = ?", (20, "banana")
        )
        await conn.commit()

        # Verify the update
        cursor = await conn.execute(
            "SELECT quantity FROM items WHERE name = ?", ("banana",)
        )
        row = await cursor.fetchone()
        assert row[0] == 20, "Update not reflected in database"

        # Close connections
        await conn.close()
        await db.close()

    finally:
        # Clean up
        import shutil

        shutil.rmtree(temp_dir)
