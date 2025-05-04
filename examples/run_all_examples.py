#!/usr/bin/env python
"""
Run all examples in the examples directory as end-to-end tests.
This script can be used for manual verification or incorporated into CI/CD.

Usage:
    python run_all_examples.py [options]

Options:
    --skip-deploy     Skip examples that deploy to Fireworks
    --skip-e2b        Skip examples that use E2B
    --only=<example>  Run only this specific example
    --list            List all available examples
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("run_examples")

# Directory containing the examples
EXAMPLES_DIR = Path(__file__).parent.absolute()

# Examples that require Fireworks API key and deployment
DEPLOY_EXAMPLES = [
    "deploy_example.py",
]

# Examples that require E2B
E2B_EXAMPLES = [
    "e2b_reward_example.py",
    "e2b_javascript_example.py",
    "e2b_auto_extract_example.py",
    "e2b_fallback_example.py"
]


def list_examples():
    """List all available example scripts."""
    all_examples = sorted([
        p.name for p in EXAMPLES_DIR.glob("*.py")
        if p.name != "run_all_examples.py" and not p.name.startswith("__")
    ])

    logger.info("Available examples:")
    for example in all_examples:
        example_type = []
        if example in DEPLOY_EXAMPLES:
            example_type.append("DEPLOY")
        if example in E2B_EXAMPLES:
            example_type.append("E2B")

        if example_type:
            logger.info(f"  {example} [{', '.join(example_type)}]")
        else:
            logger.info(f"  {example}")

    return all_examples


def run_example(example_path, env=None):
    """Run a specific example script as a subprocess."""
    example_name = Path(example_path).name
    logger.info(f"Running example: {example_name}")

    try:
        # Create a new environment with current env vars plus any additions
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        # Add special arguments for specific examples
        cmd = [sys.executable, example_path]
        
        # Handle special cases
        if example_name == "folder_based_evaluation_example.py":
            cmd.append("--auto-mode")  # Run in non-interactive mode
        elif example_name == "server_example.py":
            # For the server example, use the test script instead
            cmd[1] = str(Path(example_path).parent / "server_example_test.py")
        
        # Run the example as a subprocess with a timeout
        result = subprocess.run(
            cmd,
            env=run_env,
            check=False,
            capture_output=True,
            text=True,
            timeout=60  # Set a 60-second timeout
        )

        # Log the result
        if result.returncode == 0:
            logger.info(f"✅ {example_name} completed successfully")
            if result.stdout.strip():
                logger.info(f"Output: {result.stdout.strip()}")
            return True
        else:
            msg = f"❌ {example_name} failed with code {result.returncode}"
            logger.error(msg)  # No need to truncate now
            if result.stdout.strip():
                logger.error(f"Output: {result.stdout.strip()}")
            if result.stderr.strip():
                logger.error(f"Error: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(
            f"❌ {example_name} timed out after 60 seconds - possible interactive prompt"
        )
        return False
    except Exception as e:
        logger.error(f"❌ {example_name} failed with exception: {str(e)}")
        return False


def run_all_examples(args):
    """Run all examples or a subset based on the command line arguments."""
    # Get list of all example scripts
    all_examples = [
        p for p in EXAMPLES_DIR.glob("*.py")
        if p.name != "run_all_examples.py" and not p.name.startswith("__")
    ]

    # Filter based on command line arguments
    examples_to_run = []
    for example_path in all_examples:
        example_name = example_path.name

        # Skip if a specific example was requested and this isn't it
        if args.only and args.only != example_name:
            continue

        # Skip deploy examples if requested
        if args.skip_deploy and example_name in DEPLOY_EXAMPLES:
            logger.info(f"Skipping deploy example: {example_name}")
            continue

        # Skip E2B examples if requested
        if args.skip_e2b and example_name in E2B_EXAMPLES:
            logger.info(f"Skipping E2B example: {example_name}")
            continue

        examples_to_run.append(example_path)

    # Sort alphabetically for consistent execution order
    examples_to_run.sort()

    # Set up environment variables
    env = {}
    if not args.skip_deploy:
        # Check if Fireworks API key is available
        if ("FIREWORKS_API_KEY" not in os.environ and
                "DEV_FIREWORKS_API_KEY" not in os.environ):
            logger.warning(
                "FIREWORKS_API_KEY or DEV_FIREWORKS_API_KEY not set. "
                "Deploy examples may fail."
            )
        elif "DEV_FIREWORKS_API_KEY" in os.environ:
            env["FIREWORKS_API_KEY"] = os.environ["DEV_FIREWORKS_API_KEY"]
            env["FIREWORKS_API_BASE"] = "https://dev.api.fireworks.ai"

    # Run each example
    success_count = 0
    fail_count = 0

    for example_path in examples_to_run:
        if run_example(example_path, env):
            success_count += 1
        else:
            fail_count += 1

    # Print summary
    logger.info(
        f"\nSummary: {success_count} succeeded, "
        f"{fail_count} failed, {len(examples_to_run)} total"
    )

    # Return success if all examples succeeded
    return fail_count == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run reward-kit examples as end-to-end tests"
    )
    parser.add_argument(
        "--skip-deploy",
        action="store_true",
        help="Skip examples that deploy to Fireworks"
    )
    parser.add_argument(
        "--skip-e2b",
        action="store_true",
        help="Skip examples that use E2B"
    )
    parser.add_argument(
        "--only",
        type=str,
        help="Run only this specific example"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available examples"
    )

    args = parser.parse_args()

    if args.list:
        list_examples()
        sys.exit(0)

    success = run_all_examples(args)
    sys.exit(0 if success else 1)
