# System Architecture Overview

This document provides a high-level overview of the `reward-kit` system architecture. Its purpose is to help contributors understand the main components of the library and how they interact.

## Core Components

The `reward-kit` library is designed around a few key concepts and components:

1.  **Reward Functions (`@reward_function`)**:
    *   The core abstraction for defining custom evaluation logic.
    *   Developers use the `@reward_function` decorator to turn Python functions into evaluators.
    *   These functions typically take conversation `messages` and other optional parameters (like `ground_truth`) and return an `EvaluateResult`.
    *   Located primarily in `reward_kit/rewards/`.

2.  **Data Models (`reward_kit/models.py`)**:
    *   Defines the primary data structures used throughout the library, such as `Message`, `EvaluateResult`, `MetricResult`, etc.
    *   Ensures consistent data handling and clear interfaces.

3.  **Evaluation Pipeline (`reward_kit/evaluation.py`)**:
    *   Manages the process of running reward functions against datasets.
    *   Handles loading data, executing reward functions, and aggregating results.
    *   Powers both local preview (`reward-kit preview`) and deployment.

4.  **Typed Interface (`reward_kit/typed_interface.py`)**:
    *   Provides type definitions and interfaces that reward functions adhere to, ensuring consistency and enabling static analysis.

5.  **CLI (`reward_kit/cli.py`)**:
    *   The command-line interface for interacting with the `reward-kit`.
    *   Provides commands for `preview`, `deploy`, and other utilities.
    *   Uses libraries like Typer or Click for command parsing.

6.  **Authentication (`reward_kit/auth.py`)**:
    *   Handles authentication with external services, particularly the Fireworks AI platform.
    *   Manages API keys and account information.
    *   See [Authentication Details](authentication.md) for more.

7.  **Deployment**:
    *   The system supports deploying reward functions as evaluators to a platform (e.g., Fireworks AI).
    *   This involves packaging the reward function and its dependencies.
    *   See [Deployment Details](deploy_to_server.md) for more information on the server-side deployment aspects.

## Directory Structure Highlights

*   `reward_kit/`: Contains the main library source code.
    *   `rewards/`: Houses the collection of built-in and custom reward functions.
*   `examples/`: Provides practical examples of how to use the library.
*   `tests/`: Contains unit and integration tests for the library.
*   `docs/`: All user and developer documentation.

## Design Principles

*   **Modularity**: Components are designed to be as independent as possible.
*   **Extensibility**: Easy to add new reward functions and integrate with different evaluation scenarios.
*   **Developer Experience**: Simple APIs (like the `@reward_function` decorator) and clear CLI tools.
*   **Type Safety**: Extensive use of type hints to catch errors early and improve code clarity.

## Further Reading

For more detailed information on specific parts of the system, refer to:

*   [Authentication](authentication.md)
*   [Deployment to Server](deploy_to_server.md)
*   The code itself within the `reward_kit/` directory.
*   Developer documentation in `development/CONTRIBUTING.md`.

This overview should provide a good starting point for understanding the `reward-kit` architecture. As the system evolves, this document will be updated.
