#!/usr/bin/env python
"""
BFCL Task Runner

A simplified script for running BFCL agent evaluations using the framework.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from framework import run_bfcl_task

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("bfcl_runner")


async def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run BFCL agent evaluations")
    parser.add_argument("--task", "-t", required=True, help="Path to task YAML file")
    parser.add_argument(
        "--model", "-m", help="Model to use (format: provider/model_name)"
    )
    parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run the task
    logger.info(f"Running BFCL task: {args.task}")
    results = await run_bfcl_task(args.task, args.model)

    # Display the results
    if "error" in results:
        logger.error(f"Task failed: {results['error']}")
    else:
        logger.info(f"Task completed with score: {results.get('score', 0.0)}")
        if "reason" in results:
            logger.info(f"Reason: {results['reason']}")

    # Save the results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    # Pretty print the results to console
    print("\n" + "=" * 50)
    print(" BFCL Task Evaluation Results")
    print("=" * 50)

    if "error" in results:
        print(f"\nERROR: {results['error']}\n")
    else:
        print(f"\nScore: {results.get('score', 0.0):.2f}")
        if "function_call_score" in results:
            print(f"Function Call Score: {results['function_call_score']:.2f}")
        if "state_match" in results:
            print(f"State Match: {results['state_match']}")
        if "reason" in results:
            print(f"\nReason: {results['reason']}")

    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
