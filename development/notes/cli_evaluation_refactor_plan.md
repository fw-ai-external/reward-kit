# Plan: Integrate Advanced Evaluation into Reward Kit CLI

**Objective:** Move the core logic from `examples/math_example/main.py` (dataset loading, system prompt handling, model response generation with caching & API client features, and evaluation orchestration) into the `reward-kit` core library and expose it via an enhanced or new CLI command. The example `main.py` (and similar examples) should become minimal, primarily defining custom reward logic if necessary.

**Phase 1: Core Library Enhancements (`reward_kit/`)**

1.  **New Orchestration Module (e.g., `reward_kit/execution/pipeline.py`):**
    *   This module will house the main orchestration logic.
    *   **Input:** A comprehensive configuration object (Pydantic model or OmegaConf DictConfig) covering:
        *   Prompt Dataset configuration.
        *   System prompt (optional string).
        *   Generation parameters (`enabled`, `model_client_config`, `cache_config`).
        *   Reward function reference (Python import path string).
        *   Reward function parameters.
        *   Output/logging parameters.
    *   **Responsibilities:**
        *   Load prompt dataset (using `reward_kit.datasets.loader`).
        *   For each sample:
            *   Construct initial messages (system prompt + user query).
            *   If generation enabled: Check cache -> (Generate if needed via ModelClient) -> Store to cache.
            *   Else: Retrieve pre-existing model response from dataset.
            *   Dynamically load and call the specified reward function.
            *   Collect results.
        *   Return aggregated results and/or save detailed results.

2.  **Model Client Abstraction (e.g., `reward_kit/generation/clients.py`):**
    *   Base class `ModelClient` (e.g., `async generate(messages) -> Optional[str]`).
    *   Implement `FireworksModelClient(ModelClient)`:
        *   Handles `aiohttp` calls, auth, robust error handling (custom exceptions for Auth, RateLimit, etc.), retries (`tenacity`), rate limiting (token bucket/semaphore).
    *   Extensible for other model providers.

3.  **Caching Logic (e.g., `reward_kit/generation/cache.py`):**
    *   `ResponseCache` class (`get(key)`, `put(key, value)`).
    *   Key generation (sample ID, prompt, model, temp).
    *   File-based JSON storage.

4.  **Configuration Models (e.g., `reward_kit/config_models.py`):**
    *   Pydantic models for pipeline configurations (e.g., `GenerationConfig`, `PipelineConfig`) if not solely relying on Hydra schemas for library internals.

5.  **Dynamic Reward Function Loading Utility (e.g., `reward_kit/utils/module_loader.py`):**
    *   Helper to load a function from its import path string.

**Phase 2: CLI Enhancements (`reward_kit/cli.py` and `reward_kit/cli_commands/`)**

1.  **New CLI Command (e.g., `reward-kit run` or `reward-kit exec-eval`):**
    *   Will be a Hydra application.
    *   Primary config file (e.g., `conf/cli/run_eval_config_schema.yaml` for structure, users provide instances).
    *   **CLI Overrides:** For dataset, model, reward function, system prompt, generation toggle, output file, etc.
    *   The command's function will:
        *   Initialize pipeline configuration from Hydra.
        *   Instantiate and run the core evaluation pipeline.
        *   Display/save results.

2.  **Update `reward_kit/cli.py`:** Add the new command.
3.  **Create `reward_kit/cli_commands/run_eval_cmd.py`** for the command logic.

**Phase 3: Refactor `examples/math_example/`**

1.  **`examples/math_example/main.py` (Minimal):**
    *   If using a standard library reward function (like `math_reward`), this file might be removed or just contain comments pointing to the CLI usage.
    *   If demonstrating a *custom* reward function, it would only define that function:
        ```python
        from reward_kit import reward_function, EvaluateResult, Message
        from reward_kit.rewards.math import math_reward

        @reward_function
        def custom_math_eval(messages: list[Message], ground_truth: str, **kwargs) -> EvaluateResult:
            return math_reward(messages=messages, ground_truth=ground_truth, **kwargs)
        ```
2.  **New Hydra Config for CLI (`examples/math_example/conf/run_math_eval.yaml`):**
    *   This config is for the new `reward-kit run` CLI command.
    *   Specifies: `dataset` (prompt dataset), `reward_function_path`, `system_prompt`, `generation` block, `reward_params`, `output_file`.
3.  **Update `examples/math_example/README.md`:**
    *   Focus on the `reward-kit run` CLI command with `run_math_eval.yaml`.
    *   Explain prompt dataset prep using `convert_dataset.py`.
4.  **Cleanup:** Remove old `examples/math_example/conf/main_config.yaml`.

**Phase 4: Documentation**

1.  Document the new CLI command, its configuration, and usage.
2.  Document the core evaluation pipeline.

This refactoring centralizes complex logic into the framework, making examples cleaner and the CLI more powerful.
