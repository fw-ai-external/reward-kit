# Plan to Reproduce DeepCoder-Style Reward Function

**Overall Goal:** Create a new reward function in `reward-kit` (or adapt an existing one) that closely mimics the sparse, test-case-based reward mechanism described in the DeepCoder paper, using Qwen3 (or a similar available model) and TRL.

**Phase 1: Core Reward Function Implementation (Adapting `fractional_code_reward`)**

1.  **Create a New Reward Function (e.g., `deepcoder_code_reward`):**
    *   **Location:** This could be a new file in `reward_kit/rewards/` or an extension within `reward_kit/rewards/code_execution.py`.
    *   **Base Logic:** Leverage the `_run_test_cases` private utility function from `fractional_code_reward` as a starting point, as it already handles iterating through test cases, injecting input, and capturing output for Python (and JS).
    *   **Parameters:**
        *   `messages`: Standard input.
        *   `language`: (e.g., "python").
        *   `test_cases`: A list of dictionaries, each with `input` (string to be fed to stdin) and `expected_output` (string).
        *   `timeout`: Execution timeout per test case.
        *   `environment`: "local" or "e2b".
        *   `api_key` (for E2B).

2.  **Implement Sparse Reward Logic:**
    *   Modify the scoring. Instead of `passed / total`, the score should be:
        *   `1.0` if **all** test cases pass.
        *   `0.0` if **any** test case fails or if there's an execution error.
    *   A test case "passes" if `compare_outputs(actual, expected)` returns `1.0` (exact match after normalization). The DeepCoder paper implies strict pass/fail per test, not similarity-based. We might need to adjust `compare_outputs` or use a stricter check for this specific reward function. For a minimal version, we can stick to `compare_outputs == 1.0`.

3.  **Code Extraction:**
    *   Use the existing `extract_code_blocks` utility.
    *   The DeepCoder paper mentions a penalty for missing `python [CODE]` tags. For a minimal repro, if `extract_code_blocks` doesn't find a suitable block, the reward should be `0.0`.

4.  **Sandbox and Security:**
    *   **Local:** Rely on the existing `execute_python_code` (which uses `_execute_python_in_subprocess` with its `reliability_guard`). Acknowledge its limitations as noted in the comments ("NOT a security sandbox").
    *   **E2B:** Utilize `execute_code_with_e2b` for a more secure environment if available and configured.
    *   The DeepCoder paper mentions specific timeouts (6-12s). Make this configurable.

5.  **Test Case Input/Output:**
    *   The `_run_test_cases` function already handles injecting `test_case["input"]` via `stdin` and capturing `stdout`. This matches the typical competitive programming setup.

**Phase 2: TRL Integration and Example Script**

1.  **TRL Adapter:**
    *   Ensure the new `deepcoder_code_reward` function can be adapted for TRL using the existing `get_trl_adapter` mechanism in `reward-kit` (similar to how other reward functions are adapted in `examples/trl_integration/trl_adapter.py`).

2.  **Example Script (Minimal DeepCoder Repro):**
    *   **Base:** Adapt an existing TRL example script, like `examples/trl_integration/ppo_example.py` or `grpo_example.py`.
    *   **Model:** Use a Qwen model (e.g., a small Qwen2 variant like `Qwen/Qwen2-1.5B-Instruct` or `Qwen/Qwen2-7B-Instruct` if resources allow, as "Qwen3" might refer to the broader family or a future release). The DeepCoder paper used a 14B model, but for minimal repro, a smaller one is fine.
    *   **Dataset:**
        *   For a *minimal* example, we won't use the large datasets from the paper (TACO, LiveCodeBench, etc.).
        *   Create a small, synthetic dataset of Python coding problems with a few test cases each. Each item should have:
            *   `prompt`: The problem description.
            *   `test_cases`: A list of `{"input": "...", "expected_output": "..."}`.
            *   (Optional) `solution`: A reference solution for initial SFT or reference.
        *   The prompt should instruct the model to produce Python code.
    *   **Reward Configuration:**
        *   Instantiate `deepcoder_code_reward` with the `test_cases` from the dataset for each sample.
    *   **Training:** Run a PPO or GRPO training loop.

**Phase 3: Simplifications for Minimal Viable Product (MVP)**

*   **Test Case Sampling:** For MVP, run *all* provided test cases for a problem. The "sample 15 most challenging" from the paper is an advanced optimization.
*   **Dataset Size:** Start with a very small, illustrative dataset (e.g., 5-10 problems, 2-3 test cases each).
*   **Formatting Penalty:** The primary reward will be based on test case pass/fail. If `extract_code_blocks` fails, it will naturally lead to a 0 reward as no code is run. Explicit penalty for tag format can be a later addition.

**Key Files to Potentially Modify/Create:**

*   `reward_kit/rewards/code_execution.py` (if extending it) OR a new file like `reward_kit/rewards/deepcoder_reward.py`.
*   A new example script in `examples/trl_integration/`, e.g., `minimal_deepcoder_grpo_example.py`.
*   A new small dataset file (e.g., `examples/trl_integration/data/deepcoder_sample_problems.jsonl`).

---
## Current Status (as of 2025-05-09 12:30 AM UTC)

**Completed:**

*   **Step 1 (A, B, C): Sample Data Creation & Processing**
    *   Created `examples/trl_integration/data/simulated_deepcoder_raw_sample.jsonl`.
    *   Created `examples/trl_integration/data_utils.py`.
    *   Generated `examples/trl_integration/data/deepcoder_mvp_transformed_sample.jsonl`.
*   **Step 2: Reward Function Implementation & Testing**
    *   Implemented `deepcoder_code_reward` in `reward_kit/rewards/deepcoder_reward.py`.
    *   Added unit tests in `tests/test_deepcoder_reward.py`. All tests passed successfully.
    *   Updated `reward_kit/rewards/__init__.py` to expose the new function.
*   **Step 3: Develop the TRL Example Script (GRPO)**
    *   Created `examples/trl_integration/minimal_deepcoder_grpo_example.py`.
    *   Refactored script from PPO to use GRPO (`GRPOConfig`, `GRPOTrainer`).
    *   Resolved `GRPOTrainer` initialization issues.
    *   Adapted reward function (`deepcoder_grpo_reward_adapter`).
*   **Step 4: Test and Refine End-to-End MVP Pipeline**
    *   Changed model in `minimal_deepcoder_grpo_example.py` to `Qwen/Qwen3-0.6B`.
    *   Fixed syntax errors and Pylint issues in `reward_kit/rewards/code_execution.py`.
    *   Modified `_run_test_cases` in `code_execution.py` to include a function-calling harness (Mode 1) triggered by `function_to_call` argument, alongside the original stdin/stdout harness (Mode 2).
    *   Attempted to improve function name extraction from prompts in `deepcoder_reward.py` using regex, but it proved unreliable.
    *   **Switched to explicit function name passing:**
        *   Added `target_function` field to `simulated_deepcoder_raw_sample.jsonl`.
        *   Updated `data_utils.py` to read `target_function`.
        *   Updated `deepcoder_reward.py` to accept and use `target_function` argument, removing regex extraction.
        *   Updated `minimal_deepcoder_grpo_example.py` to load and pass `target_function` to the reward adapter.
    *   **Further Refinements (Session 2025-05-09):**
        *   Increased `max_completion_length` in `GRPOConfig` and `max_new_tokens` in `generate_for_comparison` in `minimal_deepcoder_grpo_example.py` (user set these to 4096).
        *   Updated `extract_code_blocks` in `reward_kit/rewards/code_execution.py` to remove common verbose patterns (e.g., `<think>` tags, introductory phrases).
        *   Fixed a Pylint error in `reward_kit/rewards/code_execution.py` related to E2B `sandbox.filesystem.make_dir()` and `sandbox.filesystem.write()`.
        *   Updated prompts in `examples/trl_integration/data_utils.py` to be conditional based on `target_function`, instructing the model to generate ONLY the function definition if specified.
        *   Updated the system prompt in `generate_for_comparison` in `minimal_deepcoder_grpo_example.py` to be more general.
    *   Partially ran the `minimal_deepcoder_grpo_example.py` script.

## Observations from Test Run (2025-05-09)

The `minimal_deepcoder_grpo_example.py` script was run for 3 out of 5 planned steps before being manually interrupted.

*   **Code Extraction:**
    *   The `extract_code_blocks` function's new cleaning logic appeared to work for some cases, but the model sometimes still produced non-code content (comments, example calls) instead of pure function definitions, leading to 0.0 reward.
*   **Code Generation Quality:**
    *   The model sometimes generated correct function definitions that passed all tests.
    *   However, it also frequently:
        *   Failed to produce a function definition, outputting comments or example usage instead.
        *   Generated functions with incorrect logic (e.g., `add_one` concatenating strings).
        *   Generated functions with incorrect signatures (e.g., `get_length(*args, **kwargs)`).
*   **Function Call Harness (`_run_test_cases`):**
    *   A critical issue was observed with the argument parsing in the Python function call harness within `_run_test_cases` (specifically in the `prepare_test_code` sub-function). For inputs intended to be lists (e.g., `'[1, 2, 3]'` for `get_length`), the current parsing mechanism treats the entire input string as a single string argument to the target function, rather than parsing it into a Python list. This caused test failures for `get_length` even when the function logic itself (`len(lst)`) was correct.
*   **Warnings:**
    *   Multiple `DeprecationWarning: legacy_reward_function is deprecated. Use the reward_function decorator instead.` were observed across various reward function files.
    *   Multiple `TOKENIZERS_PARALLELISM` warnings appeared during training.

## Next Steps

1.  **[COMPLETED] Improve Input Parsing in Function Harness (Critical):**
    *   Modified the argument parsing logic within `_run_test_cases` (in `reward_kit/rewards/code_execution.py`, specifically the `prepare_test_code` sub-function for Python when `function_to_call` is active).
    *   The updated logic uses a helper (`refine_evaluated_value`) to intelligently parse initial results from `json.loads` or `ast.literal_eval`. If an evaluated result is a string that appears to be a list, dictionary, or number, it attempts a further conversion. This addresses issues where inputs like `'"[1,2,3]"'` or `'"5"'` were not being correctly typed as lists or integers respectively.
2.  **[COMPLETED] Address Deprecation Warnings:**
    *   Commented out the `warnings.warn` calls related to `legacy_reward_function` in `reward_kit/reward_function.py`. This should suppress the observed deprecation warnings during development. The library structure already promotes the new `@reward_function` decorator from `typed_interface`.
3.  **Run Full Test Script:** After addressing the critical input parsing issue, re-run the `minimal_deepcoder_grpo_example.py` script to completion to get a clearer picture of the training dynamics and model improvement.
4.  **(Optional) Set `TOKENIZERS_PARALLELISM`:** To suppress warnings during development, consider setting the `TOKENIZERS_PARALLELISM=(true|false)` environment variable.
5.  **(Future) Train on Full Dataset:** Once the MVP pipeline is more robust with the sample data, adapt data loading for the full DeepCoder dataset.
