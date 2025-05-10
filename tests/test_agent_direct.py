"""
Direct test script for the agent evaluation framework.

This script tests the agent evaluation framework by running a simple example
without requiring the full CLI.
"""

import os
import asyncio
import json
from pathlib import Path
import sys

from reward_kit.agent import load_task_from_file, AgentEvaluator


async def run_test():
    """Run a simple test of the agent evaluation framework."""
    # Check if a test directory was provided
    if len(sys.argv) > 1:
        task_dir = Path(sys.argv[1])
    else:
        # Default to the flight task example
        task_dir = Path("./examples/flight_task")
    
    # Check that the task directory exists
    if not task_dir.exists():
        print(f"Error: Task directory not found at {task_dir}")
        return
    
    # Check for task.jsonl file
    task_file = task_dir / "task.jsonl"
    if not task_file.exists():
        print(f"Error: Task file not found at {task_file}")
        return
    
    print(f"Testing task bundle: {task_dir}")
    
    # Load the tasks from the task file
    tasks = load_task_from_file(str(task_file))
    if not tasks:
        print(f"Error: No tasks found in {task_file}")
        return
    
    print(f"Loaded {len(tasks)} tasks")
    
    # Process the first task
    task = tasks[0]
    task_id = task.get("id", "test_task")
    toolset = task.get("toolset")
    
    if not toolset:
        print(f"Error: Task {task_id} has no toolset defined")
        return
    
    # Extract reward module path from toolset path
    reward_path = ".".join(toolset.split(".")[:-1] + ["reward"])
    
    # Check for seed SQL
    seed_sql = task.get("seed_sql")
    seed_file = None
    
    if seed_sql and seed_sql.startswith("file:"):
        # If seed_sql is a file reference, load it
        seed_file_relative = seed_sql[5:]  # Remove "file:" prefix
        seed_file = os.path.join(task_dir, seed_file_relative)
        seed_sql = None
    
    print(f"Task ID: {task_id}")
    print(f"Toolset: {toolset}")
    print(f"Reward path: {reward_path}")
    print(f"Seed file: {seed_file}")
    
    # Create evaluator
    print("\nSetting up evaluator...")
    evaluator = AgentEvaluator(
        task_id=task_id,
        toolset_path=toolset,
        reward_path=reward_path,
        base_dir="./runs",
        seed_sql=seed_sql,
        seed_file=seed_file
    )
    
    # Set up the evaluator
    try:
        await evaluator.setup()
        print("Evaluator setup successful")
    except Exception as e:
        print(f"Error setting up evaluator: {str(e)}")
        return
    
    # Create a run
    run_id = "test_run"
    run_db_path = await evaluator.create_run(run_id)
    print(f"Created evaluation run at {run_db_path}")
    
    # Get the tools for this task
    tools_spec = evaluator.tool_registry.get_openai_tools()
    print(f"\nAvailable tools ({len(tools_spec)}):")
    for tool in tools_spec:
        print(f"  - {tool['function']['name']}: {tool['function']['description']}")
    
    # Execute a tool as an example
    if tools_spec:
        first_tool = tools_spec[0]["function"]["name"]
        print(f"\nSample execution of tool: {first_tool}")
        
        # Get sample parameters for the tool
        params = {}
        # Try to determine parameters based on tool name conventions
        if "search" in first_tool.lower() or "list" in first_tool.lower():
            if "flight" in first_tool.lower():
                params = {"origin": "SFO", "dest": "JFK", "date": "2025-05-05"}
            elif "hotel" in first_tool.lower():
                params = {"location": "New York", "checkin": "2025-05-05", "checkout": "2025-05-07"}
            else:
                params = {"query": "test"}
        
        # Execute the tool
        try:
            tool_result = await evaluator.execute_tool(run_id, first_tool, params)
            print(f"Tool execution result: {json.dumps(tool_result, indent=2)}")
        except Exception as e:
            print(f"Tool execution failed: {str(e)}")
    
    print("\nTest completed successfully")


if __name__ == "__main__":
    asyncio.run(run_test())