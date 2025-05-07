"""
Tests for the agent evaluation framework.
"""

import os
import pytest
import tempfile
import asyncio
import shutil
import time
from pathlib import Path

import pytest_asyncio
import aiosqlite
from asyncio import TimeoutError

from reward_kit.agent import (
    ToolRegistry,
    Database,
    AgentEvaluator,
    load_task_from_file,
    load_sql_from_file,
)


# Test the ToolRegistry
def test_tool_registry_basic():
    """Test basic ToolRegistry functionality."""
    registry = ToolRegistry("test_tools", "Testing tools")

    # Register a tool
    @registry.tool(
        description="Add two numbers", parameters={"a": int, "b": int}
    )
    def add(a, b):
        return a + b

    # Check that the tool was registered
    assert "add" in registry.tools
    assert registry.tools["add"].description == "Add two numbers"
    assert registry.tools["add"].parameters["a"]["type"] == "integer"
    assert registry.tools["add"].parameters["b"]["type"] == "integer"

    # Check the OpenAI format
    tools_spec = registry.get_openai_tools()
    assert len(tools_spec) == 1
    assert tools_spec[0]["type"] == "function"
    assert tools_spec[0]["function"]["name"] == "add"
    assert tools_spec[0]["function"]["description"] == "Add two numbers"
    assert (
        tools_spec[0]["function"]["parameters"]["properties"]["a"]["type"]
        == "integer"
    )
    assert (
        tools_spec[0]["function"]["parameters"]["properties"]["b"]["type"]
        == "integer"
    )
    assert "a" in tools_spec[0]["function"]["parameters"]["required"]
    assert "b" in tools_spec[0]["function"]["parameters"]["required"]


def test_tool_registry_async():
    """Test ToolRegistry with async functions."""
    registry = ToolRegistry("async_tools", "Testing async tools")

    # Register an async tool
    @registry.tool(description="Async add", parameters={"a": int, "b": int})
    async def async_add(a, b):
        return a + b

    # Check that the tool was registered as async
    assert "async_add" in registry.tools
    assert registry.tools["async_add"].is_async


def test_create_fastapi_app():
    """Test creating a FastAPI app from a ToolRegistry."""
    registry = ToolRegistry("api_tools", "Testing API tools")

    @registry.tool(
        description="Add two numbers", parameters={"a": int, "b": int}
    )
    def add(a, b):
        return a + b

    # Create a FastAPI app
    app = registry.create_fastapi_app()

    # Check that the app was created
    assert app is not None
    assert app.title == "api_tools"
    assert app.description == "Testing API tools"


# We'll avoid using fixtures for now


@pytest.mark.asyncio
async def test_database_setup():
    """Test database setup and basic functionality."""
    # Create a temporary database directly
    temp_dir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(temp_dir, "test.db")

        # Create schema
        schema = """
        CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT, value INTEGER);
        INSERT INTO test (name, value) VALUES ('test1', 100);
        """

        # Create and setup database directly with sqlite3
        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.executescript(schema)
        conn.close()

        # Now verify with our Database class
        db = Database(db_path)

        # Use the sync interface for simplicity in testing
        engine = db.get_sync_engine()
        with engine.connect() as conn:
            # Use sqlalchemy.text for SQL queries
            from sqlalchemy import text

            result = conn.execute(text("SELECT COUNT(*) FROM test"))
            assert result.scalar_one() == 1

            # Insert another row
            conn.execute(
                text("INSERT INTO test (name, value) VALUES ('test2', 200)")
            )
            conn.commit()

            # Verify the insertion
            result = conn.execute(text("SELECT COUNT(*) FROM test"))
            assert result.scalar_one() == 2
    finally:
        shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_database_snapshot():
    """Test creating a database snapshot."""
    # Create temporary directories
    source_dir = tempfile.mkdtemp()
    target_dir = tempfile.mkdtemp()

    try:
        source_db_path = os.path.join(source_dir, "source.db")
        target_db_path = os.path.join(target_dir, "target.db")

        # Simple schema
        schema = "CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT);"

        # Create the source database using standard sqlite3
        import sqlite3

        conn = sqlite3.connect(source_db_path)
        conn.executescript(schema)
        conn.execute("INSERT INTO test (value) VALUES (?)", ("test-value",))
        conn.commit()
        conn.close()

        # Create a Database instance
        db = Database(source_db_path)

        # Create a snapshot
        await db.create_snapshot(target_db_path)

        # Verify the copied database
        conn = sqlite3.connect(target_db_path)
        cursor = conn.execute("SELECT value FROM test")
        row = cursor.fetchone()
        assert row[0] == "test-value"
        conn.close()
    finally:
        shutil.rmtree(source_dir)
        shutil.rmtree(target_dir)


# Test evaluator functionality
@pytest.fixture
def example_task_path():
    """Get the path to the flight task example."""
    # This assumes the tests are run from the project root
    return os.path.join("examples", "flight_task", "task.jsonl")


def test_load_task_from_file(example_task_path):
    """Test loading tasks from a JSONL file."""
    if not os.path.exists(example_task_path):
        pytest.skip(f"Example task file not found: {example_task_path}")

    tasks = load_task_from_file(example_task_path)

    assert len(tasks) > 0
    assert "id" in tasks[0]
    assert "toolset" in tasks[0]
    assert "initial_messages" in tasks[0]


@pytest.mark.asyncio
async def test_agent_evaluator(example_task_path):
    """Test setting up an AgentEvaluator."""
    if not os.path.exists(example_task_path):
        pytest.skip(f"Example task file not found: {example_task_path}")

    tasks = load_task_from_file(example_task_path)
    assert len(tasks) > 0

    # Create a temp directory for the test
    temp_dir = tempfile.mkdtemp()

    try:
        task = tasks[0]
        task_id = task["id"]
        toolset = task["toolset"]

        # Extract reward module path from toolset path
        reward_path = ".".join(toolset.split(".")[:-1] + ["reward"])

        # Check for seed SQL
        seed_sql = task.get("seed_sql")
        seed_file = None

        if seed_sql and seed_sql.startswith("file:"):
            # If seed_sql is a file reference, load it
            seed_file_relative = seed_sql[5:]  # Remove "file:" prefix
            seed_file = os.path.join(
                os.path.dirname(example_task_path), seed_file_relative
            )
            seed_sql = None

        try:
            # Create the evaluator
            evaluator = AgentEvaluator(
                task_id=task_id,
                toolset_path=toolset,
                reward_path=reward_path,
                base_dir=temp_dir,
                seed_file=seed_file,
            )

            # Set up with timeout to prevent hanging
            try:
                await asyncio.wait_for(evaluator.setup(), timeout=10.0)

                # Create a run with timeout
                run_id = "test_run"
                run_db_path = await asyncio.wait_for(
                    evaluator.create_run(run_id), timeout=5.0
                )

                # Verify the run was created
                assert os.path.exists(run_db_path)

            except TimeoutError:
                pytest.fail(
                    "Test timed out - possible issue with database operations"
                )

        except (ImportError, ModuleNotFoundError):
            pytest.skip("Could not import example modules - skipping test")
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
