# 1. Unify Reward Result Objects

**Parent Plan:** [../../IMPROVEMENT_PLAN.md](../../IMPROVEMENT_PLAN.md)

## Original Item Description

*   **Issue:** `README.md` uses `RewardOutput` and `MetricRewardOutput`, while `development/CONTRIBUTING.md` (and likely the core library, based on its imports from `models.py`) uses `EvaluateResult` and `MetricResult`. `MetricResult` in `CONTRIBUTING.md`'s example also includes a `success: bool` field not present in `MetricRewardOutput`'s example in `README.md`.
*   **Proposal:**
    *   Standardize on `EvaluateResult` for the overall output of a reward function and `MetricResult` for individual metrics, assuming these are the canonical types from `reward_kit/models.py`.
    *   Ensure `MetricResult` consistently includes `score: float`, `reason: str`, and `success: bool` (if `success` is indeed a standard field in the actual `MetricResult` model).
    *   Update all examples in `README.md`, `development/CONTRIBUTING.md`, and any other documentation/examples (e.g., in `examples/`, `docs/`) to use these unified types.
    *   Update type hints in examples accordingly (e.g., `-> EvaluateResult`).
*   **Rationale:** Consistency reduces confusion for users and contributors. `EvaluateResult` is a more generic and descriptive term for an evaluation outcome.

## Detailed Plan

1.  **Verify Core Library Types:**
    *   **Action:** Examine `reward_kit/models.py` to confirm the exact structure and field names of `EvaluateResult` and `MetricResult`.
    *   **Check:** Specifically, verify if `MetricResult` includes `success: bool` as a standard field, and if it's optional or required.
    *   **Status:** Pending direct file access. Proceeding with assumption that `CONTRIBUTING.md` examples are representative and `MetricResult` includes `score: float, reason: str, success: bool`.
2.  **Update `README.md`:**
    *   **Action:** Search for all instances of `RewardOutput` and `MetricRewardOutput`.
    *   **Action:** Replace them with `EvaluateResult` and `MetricResult` respectively.
    *   **Action:** Adjust the example code (e.g., the `informativeness` and `combined_reward` functions) to match the confirmed structure of `EvaluateResult` and `MetricResult`. This includes constructor arguments and attribute access.
    *   **Action:** Update import statements in Python examples if necessary (e.g., `from reward_kit import EvaluateResult, MetricResult` or `from reward_kit.models import ...`, depending on how they are exposed in `reward_kit/__init__.py`).
    *   **Action:** Update return type hints in function definitions (e.g., `-> EvaluateResult`).
3.  **Update `development/CONTRIBUTING.md`:**
    *   **Action:** The example in "Reward Function Development" already uses `EvaluateResult` and `MetricResult`.
    *   **Action:** Verify its consistency with the core library types confirmed in step 1 (especially the fields of `MetricResult`, including `success`).
    *   **Action:** Ensure import paths (`from ..typed_interface import reward_function`, `from ..models import Message, EvaluateResult, MetricResult`) are correct and reflect how these models should be imported within the `reward_kit/rewards/` context.
4.  **Update `examples/` Directory:**
    *   **Action:** Review all Python files in the `examples/` directory (e.g., `evaluation_preview_example.py`, `deploy_example.py`, and any metric definition examples).
    *   **Action:** Update any reward function definitions or usage of old result types to `EvaluateResult` and `MetricResult`. Ensure consistency with the core models.
5.  **Update `docs/` Directory:**
    *   **Action:** Review all Markdown (`.md`) and MDX (`.mdx`) files in the `docs/` directory.
    *   **Action:** Update any code examples, textual descriptions, or API references that mention `RewardOutput` or `MetricRewardOutput`.
6.  **Codebase Search (Comprehensive):**
    *   **Action:** Perform a project-wide search (including tests, internal scripts, etc.) for "RewardOutput" and "MetricRewardOutput".
    *   **Action:** Replace or update these references as needed to ensure complete consistency. This includes docstrings within the library code itself.
7.  **Testing and Validation:**
    *   **Action:** After all changes, run `pytest` to ensure all tests pass.
    *   **Action:** If applicable, build and manually review the rendered documentation to ensure examples are correct and consistently use the updated types.
    *   **Action:** Run `mypy reward_kit examples tests` (or similar comprehensive mypy command) to catch any type inconsistencies introduced.

## Progress

*   [ ] Not Started
*   [X] In Progress
*   [ ] Completed

## Files to Update (Anticipated)

*   [ ] `README.md`
*   [ ] `development/CONTRIBUTING.md` (Verify consistency with core types)
*   Specific files in `docs/` (based on initial assessment, a full review of `docs/` is still needed):
    *   [ ] `docs/documentation_home.mdx` (Check for examples using old types)
    *   [ ] `docs/DOCUMENTATION_STATUS.mdx` (Check for textual references to old types)
    *   [ ] Other files in `docs/` as identified during review.
*   Relevant files within `examples/` (Requires directory scan):
    *   [ ] `examples/evaluation_preview_example.py` (If it uses reward functions)
    *   [ ] `examples/deploy_example.py` (If it uses reward functions)
    *   [ ] Other files in `examples/` as identified.
*   Potentially `reward_kit/**/*.py` (for docstrings or internal examples - Requires codebase search):
    *   [ ] Review docstrings and internal code.
*   Potentially `tests/**/*.py` (if tests directly instantiate or reference these old names - Requires codebase search):
    *   [ ] Review test files.

## Notes

*   The primary goal is to ensure that all user-facing documentation, examples, and internal code consistently use the types defined in `reward_kit.models` (i.e., `EvaluateResult` and `MetricResult`).
*   If `RewardOutput` and `MetricRewardOutput` are actual classes currently defined in the library (perhaps as aliases or older versions), this task should also include marking them as deprecated (e.g., with a `DeprecationWarning`) and creating a follow-up task or note for their eventual removal in a future version. However, the initial assessment suggests these are primarily documentation/example discrepancies. 