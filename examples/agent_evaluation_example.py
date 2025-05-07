"""
Example of using the agent evaluation framework with a task bundle.

This example demonstrates how to use the agent evaluation framework to evaluate
an agent model on a flight booking task.
"""

import os
import asyncio
import json
from pathlib import Path

from reward_kit.agent import load_task_from_file, AgentEvaluator


async def run_evaluation():
    """Run the evaluation on the flight booking task."""
    # Get the path to the flight task bundle
    current_dir = Path(__file__).parent
    flight_task_dir = current_dir / "flight_task"
    task_file = flight_task_dir / "task.jsonl"

    # Check that the task file exists
    if not task_file.exists():
        print(f"Error: Task file not found at {task_file}")
        return

    # Load the tasks from the task file
    tasks = load_task_from_file(str(task_file))
    if not tasks:
        print(f"Error: No tasks found in {task_file}")
        return

    print(f"Loaded {len(tasks)} tasks from {task_file}")

    # Process the first task
    task = tasks[0]
    task_id = task.get("id", "flight_task_1")
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
        seed_file = os.path.join(flight_task_dir, seed_file_relative)
        seed_sql = None

    # Create evaluator
    evaluator = AgentEvaluator(
        task_id=task_id,
        toolset_path=toolset,
        reward_path=reward_path,
        base_dir="./runs",
        seed_sql=seed_sql,
        seed_file=seed_file,
    )

    # Set up the evaluator
    await evaluator.setup()

    # Create a run
    run_id = "example_run_1"
    run_db_path = await evaluator.create_run(run_id)

    print(f"Created evaluation run at {run_db_path}")

    # Get the tools for this task
    tools_spec = evaluator.tool_registry.get_openai_tools()
    print(f"Available tools ({len(tools_spec)}):")
    for tool in tools_spec:
        print(
            f"  - {tool['function']['name']}: {tool['function']['description']}"
        )

    # Get the initial messages
    messages = task.get("initial_messages", [])

    # This example doesn't make actual API calls, just demonstrates setup
    print(
        "\nIn a real evaluation, the agent would now use the tools to complete the task."
    )
    print("For this example, we'll just demonstrate tool execution:")

    # Execute a tool as an example
    if tools_spec:
        first_tool = tools_spec[0]["function"]["name"]
        print(f"\nSimulating execution of tool: {first_tool}")

        # Get parameters for the tool
        params = {}
        if first_tool == "search_flights":
            params = {"origin": "SFO", "dest": "JFK", "date": "2025-05-05"}

        # Execute the tool
        try:
            tool_result = await evaluator.execute_tool(
                run_id, first_tool, params
            )
            print(f"Tool execution result: {json.dumps(tool_result, indent=2)}")
        except Exception as e:
            print(f"Tool execution failed: {str(e)}")

    # Evaluate the result
    print("\nEvaluating the (mock) interaction:")
    end_goal_sql = task.get("end_goal_sql")
    eval_kwargs = {"end_goal_sql": end_goal_sql} if end_goal_sql else {}

    # In a real evaluation, we would pass the complete messages including the agent's responses
    # For this example, we'll just use the initial messages
    try:
        evaluation = await evaluator.evaluate(
            run_id=run_id, messages=messages, **eval_kwargs
        )
        print(f"Evaluation result: {json.dumps(evaluation, indent=2)}")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
