"""
Tests for the agent evaluation framework.
"""

import os
import pytest # pytest is available globally in test files
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


@pytest.mark.skip(reason="Skipping old agent V1 test during V2 refactor")
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


@pytest.mark.skip(reason="Skipping old agent V1 test during V2 refactor")
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

# Add import shutil at the top of the file if it's used elsewhere and not just in skipped tests.
# For now, the NameError was in the skipped tests. If other tests in this file use shutil,
# it should be imported. Reviewing the file, shutil is used in test_database_setup as well.
# So, an `import shutil` at the top is indeed needed.
# The initial `import pytest` was removed as it's usually not needed explicitly.
# Let's ensure shutil is imported.

# Re-evaluating the first diff block:
# The original file had `import pytest`. My thought to remove it was based on common practice,
# but it's safer to leave it if it was there. The main issue is `shutil`.

# Corrected plan:
# 1. Add `import shutil` at the top.
# 2. Add skip markers to the two functions.
# (No need to touch the `import pytest` line if it was already there and correct)

# Let's re-generate the diff based on this.
# The file content provided shows `import pytest` is already there.
# So, the only needed changes are adding `import shutil` and the skip markers.

# Corrected diff based on provided file content:

# Add `import shutil`
# Add skip markers.

# The provided file content already has `import shutil` (line 5).
# This means the NameError for shutil is unexpected if the provided file content is accurate for what's on disk.
# Let's assume the provided file content is the source of truth.
# If `import shutil` is already there, then the `NameError` for `shutil` in the test output is confusing.
# Perhaps the test runner is picking up an old version of the file?

# Let's assume the test output is correct and `shutil` is NOT defined, meaning it's missing the import.
# The provided file content for `tests/test_agent.py` DOES have `import shutil`.
# This is a contradiction.

# I will proceed by adding the skip markers as requested, and I will also ensure `import shutil` is present.
# If `import shutil` is already there as per the provided file content, the diff for adding it will be a no-op for that line.

# The `NameError: name 'shutil' is not defined` is the primary reason for skipping.
# The user's message implies these tests are failing due to this NameError.
# The file content shown in the prompt for `tests/test_agent.py` *does* include `import shutil`.
# This suggests the error might be from a different state of the file or an environment issue.

# Given the user's request to "skip the old agent test cases" and the specific error,
# I will focus on adding the skip markers.
# If `import shutil` is truly missing from the actual file on disk, the test environment will tell us.
# For now, I will trust the user's error report that `shutil` is not defined at runtime for those tests.
# The skip markers will prevent them from running anyway.

# The `test_database_setup` also uses `shutil.rmtree(temp_dir)`. If `shutil` is not imported, this would also fail.
# The user did not list `test_database_setup` as failing with a NameError.
# This further suggests the `import shutil` might be present in the file but perhaps not effective in the execution scope of the failing tests,
# or the test failures provided are the only ones of concern.

# I will only add the skip markers as the primary action.
# The `shutil` NameError is the *reason* they are being skipped (as per the error log).
# If `import shutil` is missing, it should be added for other tests like `test_database_setup` to pass.
# Let's check `test_database_setup` again. It uses `shutil.rmtree(temp_dir)`.
# If `import shutil` is missing, this test *should* also fail with NameError.
# The user's list of failures does not include it.

# This implies `import shutil` is likely present. The NameError in the *specific failing tests* might be due to
# how pytest isolates tests or some other subtlety.
# Safest action: add skip markers as requested. Assume `import shutil` is present as per file content.
# The skip markers make the `shutil` issue in those specific tests moot.

# The diff should be:
# 1. Add `@pytest.mark.skip(...)` to `test_database_snapshot`
# 2. Add `@pytest.mark.skip(...)` to `test_agent_evaluator`
# (No changes to imports if they are already correct as per the provided file content)
# The provided file content for `tests/test_agent.py` shows `import shutil` on line 6.
# And `import pytest` on line 3. So imports are fine.
