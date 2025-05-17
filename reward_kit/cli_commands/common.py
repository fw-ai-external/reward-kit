"""
Common utility functions for the Reward Kit CLI.
"""

import logging
import os


def setup_logging(verbose=False, debug=False):
    """Setup logging configuration"""
    if debug:
        log_level = logging.DEBUG
        format_str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    elif verbose:
        log_level = logging.INFO
        format_str = "%(levelname)s:%(name)s:%(message)s"
    else:
        log_level = logging.WARNING
        format_str = "%(levelname)s:%(message)s"

    logging.basicConfig(level=log_level, format=format_str)


def check_environment():
    """Check if required environment variables are set for general commands."""
    if not os.environ.get("FIREWORKS_API_KEY"):
        print("Warning: FIREWORKS_API_KEY environment variable is not set.")
        print(
            "This is required for API calls. Set this variable before running the command."
        )
        print("Example: FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY reward-kit [command]")
        return False
    return True


def check_agent_environment(test_mode=False):
    """Check if required environment variables are set for agent evaluation commands."""
    missing_vars = []
    if not os.environ.get("MODEL_AGENT"):
        missing_vars.append("MODEL_AGENT")

    # MODEL_SIM is mentioned in AGENT_ISSUES.md but not checked here in original code.
    # Keeping original logic.

    if test_mode:
        # In test mode, we don't strictly require these environment variables
        if missing_vars:
            print(
                f"Note: The following environment variables are not set: {', '.join(missing_vars)}"
            )
            print(
                "Since you're running in test mode, these are not strictly required for all operations."
            )
        return True  # Return true as it's test mode, warnings are informational

    if missing_vars:
        print(
            f"Warning: The following environment variables are not set: {', '.join(missing_vars)}"
        )
        print(
            "These are typically required for full agent evaluation. Set these variables for full functionality."
        )
        print("Example: MODEL_AGENT=openai/gpt-4o-mini reward-kit agent-eval [args]")
        print(
            "Alternatively, use --test-mode for certain validation tasks without requiring all API keys."
        )
        return False  # Return false if not test_mode and vars are missing
    return True
