#!/usr/bin/env python
"""
Example script showing how to programmatically use the BFCL framework.

This demonstrates running a BFCL task from Python code rather than the command line.
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from framework import run_bfcl_task

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("bfcl_example")


async def main():
    """Run an example BFCL evaluation."""
    # Path to task definition
    task_path = Path(__file__).parent / "tasks" / "file_management_task.yaml"

    # Make sure we have an OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        api_key = input("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key

    # Set which model to use
    model = os.environ.get("MODEL_AGENT", "openai/gpt-4")

    print(f"Running BFCL task with model: {model}")
    print(f"Task file: {task_path}")

    # Run the evaluation
    results = await run_bfcl_task(task_path, model)

    # Display results
    print("\nResults:")
    print(json.dumps(results, indent=2))

    # You can access specific metrics
    if "error" not in results:
        print(f"\nScore: {results.get('score', 0.0):.2f}")
        print(f"Function Call Score: {results.get('function_call_score', 0.0):.2f}")
        print(f"State Match: {results.get('state_match', False)}")
        print(f"\nReason: {results.get('reason', '')}")


if __name__ == "__main__":
    asyncio.run(main())
