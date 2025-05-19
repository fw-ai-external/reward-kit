# 2. Remove `original_messages` Parameter

**Parent Plan:** [../../IMPROVEMENT_PLAN.md](../../IMPROVEMENT_PLAN.md)

## Original Item Description

*   **Issue:** The `original_messages` parameter is used in some reward functions to provide conversational context separately from the main `messages` list (which typically ends with the assistant's response). The new coding format aims to consolidate this by having `ground_truth` serve as the container for such context if it's a list of messages.
*   **Proposal:**
    *   Remove the `original_messages` parameter from all reward function signatures and internal processing.
    *   Adapt reward functions and their callers (tests, examples) to source any necessary prior context from the `ground_truth` parameter when `ground_truth` is structured as a list of messages, or from the main `messages` parameter if appropriate.
    *   Update all documentation, examples, and tests to reflect this change.
*   **Rationale:** Simplifies the reward function interface, reduces redundancy, and aligns with the evolving standard of using `ground_truth` as a flexible container for various forms of expected data, including conversational history.

## Detailed Plan

1.  **Codebase Analysis - Identify Usages of `original_messages`:**
    *   **Action:** Perform a comprehensive search across the `reward_kit/`, `examples/`, and `tests/` directories for all occurrences of the `original_messages` parameter.
    *   **Tools:** Use `search_files` tool or equivalent IDE search.
    *   **Focus:**
        *   Function signatures in `reward_kit/rewards/**/*.py`.
        *   Internal logic within these functions that consumes `original_messages`.
        *   The `@reward_function` decorator and `reward_kit/typed_interface.py` for any special handling.
        *   How `original_messages` is provided in test setups (`tests/**/*.py`).
        *   How `original_messages` is used in example scripts (`examples/**/*.py`).

2.  **Refactor Reward Functions and Related Code:**
    *   **Action:** For each identified reward function:
        *   Remove `original_messages: Optional[List[Message]] = None` (or similar type hint) from its signature.
        *   Analyze how `original_messages` was used. If it provided conversational history, modify the function to expect this history as part of `ground_truth` (when `ground_truth` is a list of `Message` objects) or from the main `messages` list.
        *   Adjust internal logic to reflect the new source of this contextual information.
    *   **Action:** Update `reward_kit/typed_interface.py` if it has specific logic tied to `original_messages` to remove or adapt this logic.

3.  **Update Test Cases:**
    *   **Action:** Review and modify all test files in `tests/`.
    *   **Action:** For tests that previously supplied `original_messages`, update them:
        *   If the context from `original_messages` is still needed, ensure it's provided via the `ground_truth` parameter (e.g., by making `ground_truth` a list of messages that includes this context).
        *   Adjust assertions and test logic accordingly.

4.  **Update Examples:**
    *   **Action:** Review and modify all example scripts in `examples/`.
    *   **Action:** Remove `original_messages` from any reward function calls.
    *   **Action:** Ensure that if context previously passed via `original_messages` is still required for the example to make sense, it's now correctly passed via `ground_truth` or handled by the updated reward logic.

5.  **Update Documentation:**
    *   **Action:** Review and update all documentation files, including:
        *   `development/CONTRIBUTING.md` (if examples use `original_messages`).
        *   Files in `docs/` (API references, tutorials, developer guides).
        *   Docstrings within the `reward_kit` codebase (for functions, classes, modules).
    *   **Action:** Clearly explain that `original_messages` is deprecated/removed and how to provide similar context using `ground_truth` if it's a list of messages.

6.  **Testing and Validation:**
    *   **Action:** Run `pytest tests/` to ensure all tests pass after the refactoring.
    *   **Action:** Run `mypy reward_kit examples tests` (or a similar comprehensive mypy command) to check for type errors and ensure type consistency.
    *   **Action:** Manually review a few key examples and documentation pages to verify the changes are correctly implemented and clearly communicated.

## Progress

*   [ ] Not Started
*   [ ] In Progress
*   [X] Completed

## Files to Update (Anticipated - based on initial assessment)

*   `reward_kit/rewards/**/*.py` (various reward function definitions)
*   `reward_kit/typed_interface.py` (potentially)
*   `tests/**/*.py` (various test files)
*   `examples/**/*.py` (various example files)
*   `docs/**/*.md, .mdx` (various documentation files)
*   `development/CONTRIBUTING.md`

## Notes

*   **Investigation Finding (2025-05-20):** A thorough search for `original_messages` was conducted across `reward_kit/`, `examples/`, `tests/` directories, as well as documentation files (`docs/`, `development/CONTRIBUTING.md`). No occurrences of the `original_messages` parameter were found in the codebase or documentation. This suggests the parameter may have been removed previously or was never implemented as described.
*   **Validation (2025-05-20):**
    *   `pytest tests/` executed successfully (388 passed, 17 skipped, numerous warnings but no failures).
    *   Initial `mypy reward_kit examples tests` reported 7 errors.
    *   Re-running `pip install -e ".[dev]"` resolved 6 errors by installing missing type stubs (`types-requests`, `types-PyYAML`).
    *   The final remaining mypy error (`tests/test_typed_interface.py:132: error: Dict entry 0 has incompatible type "str": "dict[str, float]"; expected "str": "MetricResult"  [dict-item]`) was resolved by adding a targeted `# type: ignore[dict-item]` comment to the specific line in `tests/test_typed_interface.py`.
    *   Subsequent `mypy reward_kit examples tests` run reported "Success: no issues found".
    *   All mypy errors encountered during validation were unrelated to the `original_messages` parameter.
*   The core change is to shift the responsibility of providing extended conversational context from a dedicated `original_messages` parameter to the more flexible `ground_truth` parameter, particularly when `ground_truth` is used to pass a list of `Message` objects.
*   This refactoring aims to streamline the API for reward functions.
