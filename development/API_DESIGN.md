# Reward Function API, Data Format, and Input Handling Refactor

## 1. Overview

This document outlines a significant refactor to the Reward Kit's API concerning reward functions and their associated data formats. The primary goals are:

*   **Standardize Dataset Format**: Establish a clear and unambiguous structure for `.jsonl` datasets used with reward functions.
*   **Clarify Reward Function Inputs**: Define distinct roles for input parameters, with `messages` representing the prompt/context and `ground_truth` representing the complete expected assistant response.
*   **Explicit Type-Hint Driven Pydantic Conversion**: Empower developers to control Pydantic conversion of `messages` and `ground_truth` inputs through explicit type hints in their reward function signatures, managed by the `@reward_function` decorator.
*   **Remove Ambiguity**: Deprecate and remove the `original_messages` concept to simplify the data model.

## 2. Core Concepts

### 2.1. Dataset Structure (`.jsonl` files)

Reward function datasets, typically in `.jsonl` format, will adhere to the following structure for each JSON line (entry):

*   **`messages`: `List[Message]`**
    *   This field contains a list of message objects representing the input prompt or conversation context provided to the language model.
    *   Each message object in the list should conform to a dictionary structure (e.g., `{"role": "user", "content": "Hello"}`).
    *   This would contain the model output
*   **`ground_truth`: Can be `List[Message]` or `str` Any**
    *   This field contains a complete list of message object representing the expected or target response from the assistant for the given `messages` prompt.
    *   Each object includes all relevant attributes of an assistant's turn, such as `role` (which will be "assistant"), `content`, `tool_calls`, `function_call`, etc.
    *   Example: 
        * `{"role": "assistant", "content": "Hi there!", "tool_calls": null}` or 
        * just some simple integer for math questions.
    *   It can also be Anything else, for example for math or coding, it is cleaner to just have one string as the output.
    *   This would **not** be the model output, it will be the ground truth that the model output should be compared against.
*   **`original_messages`: (Deprecated)**
    *   This field is **deprecated** and must be removed from all datasets. 

**Example `.jsonl` line:**
```json
{"messages": [{"role": "user", "content": "What is the capital of France?"}], "ground_truth": [{"role": "assistant", "content": "The capital of France is Paris."}]}
```

### 2.2. Reward Function Signature

The recommended standard signature for a reward function is as follows:

```python
from typing import List, Dict, Any, Union
from reward_kit.models import Message, EvaluateResult
from reward_kit.typed_interface import reward_function

@reward_function
def my_reward_func(
    messages: List[Message],  # Or List[Dict[str, Any]] if no Pydantic conversion is desired
    ground_truth: List[Message],    # Or Any if no Pydantic conversion is desired
    **kwargs: Any
) -> EvaluateResult:
    # Assistant's response content can be accessed via ground_truth.content
    # Prompt/context is in messages
    # ... evaluation logic ...
    pass
```

The type hints for the `messages` and `ground_truth` parameters play a crucial role in how the `@reward_function` decorator processes these inputs (see next section).

### 2.3. `@reward_function` Decorator Behavior (Input Handling)

The `@reward_function` decorator, located in `reward_kit/typed_interface.py`, will manage the Pydantic conversion of the `messages` and `ground_truth` arguments based on the type hints of the decorated function.

*   **`messages` Argument Processing**:
    *   If the decorated function's `messages` parameter is explicitly type-hinted as `List[Message]` (e.g., `messages: List[Message]`), the decorator will attempt to convert the input `messages` (expected to be `List[Dict[str, Any]]` from the dataset) into a list of `reward_kit.models.Message` Pydantic objects.
    *   If the type hint is anything else (e.g., `List[Dict[str, Any]]`, `Any`, or if no type hint is provided that specifically indicates `List[Message]`), the `messages` argument will be passed through to the decorated function as is, without Pydantic conversion.
*   **`ground_truth` Argument Processing**:
    *   If the decorated function's `ground_truth` parameter is explicitly type-hinted as `List[Message]` (e.g., `ground_truth: List[Message]`), the decorator will attempt to convert the input `ground_truth` (expected to be `List[Dict[str, Any]]` from the dataset) into a `List[reward_kit.models.Message]` Pydantic object.
    *   If the type hint is anything else (e.g., `List[Dict[str, Any]]`, `str`, `Any`, or if no type hint is provided that specifically indicates `List[Message]`), the `ground_truth` argument will be passed through as is.
*   **Type Hinting Guidance**:
    *   For maximum clarity and predictable behavior, developers should use explicit type hints for `messages` and `ground_truth` parameters in their reward functions.
    *   Preferred hints:
        *   For `messages`: `List[Message]` (for Pydantic objects) or `List[Dict[str, Any]]` (for raw dictionaries).
        *   For `ground_truth`: `List[Message]` (for a Pydantic object) or `List[Dict[str, Any]]` (for a raw dictionary).
    *   The use of `Union` types for these specific parameters (e.g., `Union[List[Message], List[Dict[str, Any]]]`) should be minimized in new and refactored code to simplify understanding the decorator's behavior. The decorator will prioritize the `Message` / `List[Message]` part of a Union if present for conversion.

## 3. Refactoring Tasks (Breakdown for Parallel Work)

The following tasks are identified to implement this refactor. They can be worked on in parallel where dependencies allow.

**Task 1: Update `@reward_function` Decorator (`reward_kit/typed_interface.py`)**
*   **Description**: Modify the decorator to implement the conditional Pydantic conversion logic described in section 2.3 for `messages` and `ground_truth` arguments based on their type hints in the decorated function.
*   **Key Files**: `reward_kit/typed_interface.py`
*   **Acceptance Criteria**:
    *   Decorator correctly converts `messages` to `List[Message]` if type-hinted as such.
    *   Decorator correctly converts `ground_truth` to `List[Message]` if type-hinted as such.
    *   Inputs are passed as-is if not matching these specific Pydantic type hints.
    *   Comprehensive unit tests for all conversion scenarios (hinted for Pydantic, hinted for raw, no hint) are added and pass.

**Task 2: Dataset Migration Script & Execution**
*   **Description**: Develop a robust Python script to transform all existing `.jsonl` datasets to the new format. The script must:
    1.  Iterate through each line of an input `.jsonl` file.
    2.  For each JSON object, read the `messages` list.
    3.  Identify the last message in this list. If it has `role: "assistant"`, move this entire message object into a new list and assign it to a new top-level field named `ground_truth`. (If `ground_truth` is intended to capture a longer trajectory, this step would involve collecting all relevant assistant and tool messages into the list).
    4.  The `messages` field should then be updated to contain only the preceding messages (the context/prompt).
    5.  Remove the `original_messages` field entirely if it exists.
    6.  Write the modified JSON object to an output file (or overwrite with backup).
*   **Files to Migrate**: A comprehensive list needs to be compiled by searching for `*.jsonl` files within the `examples/` and `development/` directories. Key examples include:
    *   `examples/samples/samples.jsonl`
    *   `development/CODING_DATASET.jsonl`
    *   (Others to be identified)
*   **Acceptance Criteria**:
    *   All specified `.jsonl` datasets are correctly transformed to the new structure.
    *   The script is idempotent or handles already migrated files gracefully.
    *   The script logs its actions and any anomalies encountered.

**Task 3: Refactor Core Reward Functions (`reward_kit/rewards/`)**
*   **Description**: Update all existing reward functions located in the `reward_kit/rewards/` directory to:
    1.  Adapt their signatures to accept `messages` (as the prompt/context) and `ground_truth` (as the assistant's full response object).
    2.  Remove any internal logic that extracts the assistant's response from `messages[-1]`. Instead, use attributes of the `ground_truth` objects (e.g., `ground_truth[0].content`, iterate through the list if it's a trajectory).
    3.  Completely remove any handling or reliance on an `original_messages` parameter.
    4.  Update type hints for `messages` and `ground_truth` to be explicit (e.g., `List[Message]`, `List[Dict[str, Any]]`) to clearly define whether Pydantic conversion is expected for each.
*   **Key Files**: All Python files in `reward_kit/rewards/`.
*   **Acceptance Criteria**:
    *   All core reward functions are refactored to the new input paradigm.
    *   Functions operate correctly with the new data flow.
    *   Corresponding unit tests in the `tests/` directory are updated to reflect these changes and all tests pass.

**Task 4: Update Example Scripts (`examples/`)**
*   **Description**: Review and refactor all Python example scripts in the `examples/` directory that demonstrate the use of reward functions, dataset loading, or dataset processing. These scripts must be updated to align with the new API, data format, and best practices.
*   **Key Files**: All relevant `*.py` files in `examples/`.
*   **Acceptance Criteria**:
    *   Example scripts run correctly using the refactored reward functions and new dataset structure.
    *   Examples clearly demonstrate the new way of defining and using reward functions.

**Task 5: Update TRL Adapter (`reward_kit/reward_function.py`)**
*   **Description**: Review and update the `RewardFunction.get_trl_adapter` method. Ensure it correctly processes `prompts` (which will map to the new `messages` format – the input context) and `solutions` (which will map to the new `ground_truth` format – the target assistant response). The adapter must pass these to the underlying reward function in the expected manner.
*   **Key Files**: `reward_kit/reward_function.py`
*   **Acceptance Criteria**:
    *   The TRL adapter correctly interfaces with refactored reward functions.
    *   TRL integration examples (if any) continue to work or are updated.
    *   Relevant tests for the TRL adapter pass.

**Task 6: Documentation Update (`docs/`, `CONTRIBUTING.md`, etc.)**
*   **Description**: Thoroughly update all project documentation to reflect the changes. This includes:
    *   API reference documentation for `@reward_function`, `Message`, `EvaluateResult`.
    *   Tutorials and guides on creating and using reward functions.
    *   Explanations of the new dataset format.
    *   Updates to `CONTRIBUTING.md` regarding reward function development standards and type hinting.
*   **Key Files**: Files within `docs/`, `README.md`, `development/CONTRIBUTING.md`.
*   **Acceptance Criteria**: All documentation is accurate, consistent, and clearly explains the new API and data structures.

**Task 7: Review `RewardFunction` Class (`reward_kit/reward_function.py`)**
*   **Description**: Examine the `RewardFunction` class, specifically its `__call__` method and how it handles functions loaded via `func_path` that might not be decorated with `@reward_function`. Determine if this class should also implement similar type-hint-based Pydantic conversion for `messages` and `ground_truth` for consistency, or if it should be clearly documented that functions used this way must manage their own input types.
*   **Key Files**: `reward_kit/reward_function.py`
*   **Acceptance Criteria**: The behavior of the `RewardFunction` class with potentially non-decorated functions is well-defined, consistent with the overall design, and clearly documented.

## 4. Code Style and Type Hinting Conventions

*   **Explicit Type Hints**: When defining or refactoring reward functions, use explicit type hints for `messages` and `ground_truth` parameters.
    *   For `messages`: Prefer `List[Message]` if Pydantic objects are desired, or `List[Dict[str, Any]]` if raw dictionaries are preferred.
    *   For `ground_truth`: Prefer `List[Message]` if Pydantic objects are desired, or `List[Dict[str, Any]]` if raw dictionaries are preferred.
*   **Minimize `Union` Types**: Avoid using `Union` types for the `messages` and `ground_truth` parameters in reward function signatures where possible, as the decorator's behavior is more straightforward with explicit, non-union types.
*   **Consistency**: Ensure that the type hints used in function signatures accurately reflect whether Pydantic conversion is expected for those inputs.

## 5. Progress Update (As of 2025-05-15)

**Task 1: Update `@reward_function` Decorator (`reward_kit/typed_interface.py`)**
*   **Status**: COMPLETED
*   **Details**: The decorator in `reward_kit/typed_interface.py` has been modified to implement conditional Pydantic conversion for `messages` (to `List[Message]`) and `ground_truth` (to `List[Message]`) based strictly on whether the decorated function's parameters are type-hinted as `List[Message]`.

**Task 2: Dataset Migration Script & Execution**
*   **Status**: COMPLETED
*   **Details**: A script `scripts/migrate_datasets.py` was created and executed on the following target files:
    *   `development/CODING_DATASET.jsonl`
    *   `examples/samples/samples.jsonl`
    *   `examples/flight_task/task.jsonl`
    *   `examples/trl_integration/data/simulated_deepcoder_raw_sample.jsonl`
    *   `examples/trl_integration/data/simulated_deepcoder_transformed_sample.jsonl`
    The script transformed these datasets to the new format (`messages` for prompt, `ground_truth` as a list containing the assistant's response message, and `original_messages` removed).

**Task 3: Refactor Core Reward Functions (`reward_kit/rewards/`)**
*   **Status**: IN PROGRESS
*   **Completed Sub-tasks**:
    *   `reward_kit/rewards/accuracy.py` and `tests/test_accuracy.py` refactored and tests passing.
    *   `reward_kit/rewards/length.py` and `tests/test_length.py` refactored and tests passing.
    *   `reward_kit/rewards/format.py` and `tests/test_format.py` refactored and tests passing.
    *   `reward_kit/rewards/repetition.py` and `tests/test_repetition.py` refactored and tests passing.
    *   `reward_kit/rewards/json_schema.py` (specifically `json_schema_reward` and `json_schema_reward_with_llm_judge`) and `tests/test_json_schema.py` refactored and tests passing.
    *   `reward_kit/rewards/function_calling.py` (specifically `schema_jaccard_reward`, `llm_judge_reward`, `composite_function_call_reward`, and helper `match_function_call`) and `tests/test_function_calling.py` refactored and tests passing.
    *   `reward_kit/rewards/math.py` and `tests/test_math.py` refactored and tests passing.
    *   `reward_kit/rewards/language_consistency.py` and `tests/test_language_consistency.py` refactored and tests passing.
    *   `reward_kit/rewards/code_execution.py` and `tests/test_code_execution.py` refactored and tests passing.
    *   `reward_kit/rewards/bfcl_reward.py` refactored. (Note: No specific tests found for `bfcl_reward.py` in `tests/` directory).
    *   `reward_kit/rewards/cpp_code.py` and `tests/test_cpp_code.py` refactored and tests passing.
    *   `reward_kit/rewards/deepcoder_reward.py` and `tests/test_deepcoder_reward.py` refactored and tests passing.
    *   `reward_kit/rewards/lean_prover.py` and `tests/test_lean_prover.py` refactored and tests passing.
    *   `reward_kit/rewards/list_comparison_math_reward.py` and `tests/test_list_comparison_math_reward.py` refactored and tests passing.
*   **Remaining Sub-tasks (Example)**:
    *   Refactor `reward_kit/rewards/advanced_math.py` and its tests. (Note: File `reward_kit/rewards/advanced_math.py` not found, likely removed or renamed).
    *   Refactor `reward_kit/rewards/multiple_choice_math_reward.py` and its tests.
    *   Refactor `reward_kit/rewards/reasoning_steps.py` and its tests.
    *   Refactor `reward_kit/rewards/accuracy_length.py` and its tests.
    *   (Review all files in `reward_kit/rewards/` for any other reward functions).

**Task 4: Update Example Scripts (`examples/`)**
*   **Status**: PENDING

**Task 5: Update TRL Adapter (`reward_kit/reward_function.py`)**
*   **Status**: PENDING

**Task 6: Documentation Update (`docs/`, `CONTRIBUTING.md`, etc.)**
*   **Status**: PENDING

**Task 7: Review `RewardFunction` Class (`reward_kit/reward_function.py`)**
*   **Status**: PENDING
