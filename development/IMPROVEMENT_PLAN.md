# Reward Kit Improvement Plan

This document outlines proposed changes to enhance the usability, clarity, and overall developer/user experience for the Reward Kit library. The focus is on refining documentation (`README.md`, `CONTRIBUTING.md`) and ensuring consistency.

## Approved Action Items

1.  **Unify Reward Result Objects:**
    *   **Issue:** `README.md` uses `RewardOutput` and `MetricRewardOutput`, while `development/CONTRIBUTING.md` (and likely the core library, based on its imports from `models.py`) uses `EvaluateResult` and `MetricResult`. `MetricResult` also includes a `success: bool` field not present in `MetricRewardOutput`'s example.
    *   **Proposal:**
        *   Standardize on `EvaluateResult` for the overall output of a reward function and `MetricResult` for individual metrics.
        *   Ensure `MetricResult` consistently includes `score: float`, `reason: str`, and `success: bool` (where applicable).
        *   Update all examples in `README.md`, `CONTRIBUTING.md`, and any other documentation/examples to use these unified types.
        *   Update type hints in examples accordingly (e.g., `-> EvaluateResult`).
    *   **Rationale:** Consistency reduces confusion for users and contributors. `EvaluateResult` is a more generic and descriptive term for an evaluation outcome.
    *   **Detailed Plan & Progress:** [./improvements/01_unify_reward_result_objects.md](./improvements/01_unify_reward_result_objects.md)

## Proposed Action Items

### I. Project-Wide Consistency & Clarity (Continued)

1.  **Clarify "Metric Folder" Structure:**
    *   **Issue:** The term `--metrics-folders "name=./path/to/folder"` is used frequently, but the expected content and structure of this folder (e.g., what files define the metric, how `__init__.py` is used if it's a Python module) isn't explicitly clear in the `README.md`. `CONTRIBUTING.md` mentions `examples/metrics/word_count`.
    *   **Proposal:**
        *   In `README.md` (Getting Started/CLI sections) and `CONTRIBUTING.md` (Reward Function Development), briefly explain what a "metric folder" should contain.
        *   Provide a simple example of a metric file (e.g., `word_count_metric.py`) and how it's structured, especially if it's just a Python file with a `@reward_function` decorated function.
        *   Explain how the `name` in `name=./path/to/folder` is used (e.g., as an identifier in results or configuration).
    *   **Rationale:** Improves usability by making it clear how to structure custom metrics for use with the CLI and deployment.

2.  **API Key Guidance:**
    *   **Issue:** Instructions to set API keys are present, but lack guidance on *where* users obtain these keys.
    *   **Proposal:**
        *   In `README.md` (Authentication Setup) and `CONTRIBUTING.md` (Required Environment Variables), add a brief note like: `(obtain your API key from your Fireworks AI account dashboard)`.
    *   **Rationale:** Small addition that significantly helps new users get started.

3.  **Documentation Links:**
    *   **Issue:** Links to "full documentation" or "official documentation" sometimes point to a `.mdx` file or the GitHub repository root.
    *   **Proposal:**
        *   If a rendered documentation website exists (e.g., ReadTheDocs, GitHub Pages from the `.mdx` files), all primary documentation links should point to this live site.
        *   If not, links should consistently point to the most relevant Markdown file within the repository (e.g., `docs/documentation_home.mdx` or specific sections in `CONTRIBUTING.md`).
    *   **Rationale:** Provides users with the best and most accessible version of the documentation.

### II. `README.md` Enhancements

Focus: Make the README more user-centric (using the library) and streamlined.

4.  **Refine "Getting Started" Section:**
    *   **Issue:** Placeholder paths like `./path/to/metrics` can be slightly improved. The purpose of `original_messages` and `**kwargs` in simple examples isn't clear.
    *   **Proposal:**
        *   For CLI examples, consider using a path relative to the `examples/` directory if that's the primary source of samples, e.g., `--metrics-folders "word_count=./examples/metrics/word_count"`. Or, clearly state that `./path/to/your_metrics_dir` is a placeholder.
        *   In the `informativeness` example:
            *   Update to use `EvaluateResult` and `MetricResult`.
            *   Add a type hint for the return value: `-> EvaluateResult`.
            *   For `original_messages=None` and `**kwargs`: If these are standard parts of the `@reward_function` signature, add a brief comment (e.g., `# original_messages: Optional[List[Message]] = None - for context if needed` and `# **kwargs: Any - for additional parameters passed by the evaluation system`). If not strictly necessary for basic examples, consider omitting them for simplicity in the *very first* example, introducing them later in "Advanced Usage".
            *   The scoring logic `min(word_count / 100, 1.0)` could benefit from a brief comment, e.g., `# Score normalized to 0-1, assuming 100 words is a good target for this example`.
    *   **Rationale:** Improves clarity and makes the first steps smoother.

5.  **Streamline "Development" Section:**
    *   **Issue:** The "Development" section (Type Checking, Running Tests) is brief and largely duplicates content in `CONTRIBUTING.md`.
    *   **Proposal:**
        *   Reduce this section to a minimum.
        *   Add a prominent link to `development/CONTRIBUTING.md` for comprehensive development and contribution guidelines. Example: "For details on setting up a development environment, running tests, type checking, and contributing to Reward Kit, please see our [Contributing Guidelines](development/CONTRIBUTING.md)."
    *   **Rationale:** Keeps README focused on library usage, directing contributors to the dedicated guide.

6.  **Review `RewardOutput` and `MetricRewardOutput` Usage (Covered by Approved Item 1):**
    *   **Action:** This item is now covered by the approved "Unify Reward Result Objects" task. Once completed, this specific point can be considered resolved.

7.  **CLI Examples Clarity:**
    *   **Issue:** The structure of `--metrics-folders "word_count=./path/to/metrics"` could be briefly explained.
    *   **Proposal:** Add a short note after the first CLI example using it: e.g., "Here, `word_count` is a name you assign to your metric(s) defined in the specified folder."
    *   **Rationale:** Reduces ambiguity for a key CLI argument.

### III. `development/CONTRIBUTING.md` Enhancements

Focus: Improve clarity, reduce redundancy, and ensure examples align with best practices.

8.  **Streamline Setup Sections:**
    *   **Issue:** "Quick Start" and "Development Environment" sections have significant overlap.
    *   **Proposal:**
        *   Consider merging them or making "Quick Start" the primary entry point and linking to more detailed subsections within "Development Environment" if necessary.
        *   Ensure the "Quick Start" is truly quick and covers the essentials: clone, venv, install dev dependencies, run tests/linters.
    *   **Rationale:** Improves readability and reduces cognitive load for new contributors.

9.  **Update Reward Function Example (Related to Approved Item 1):**
    *   **Issue:** The example uses `EvaluateResult` and `MetricResult`, which is good. Ensure it remains the canonical example after project-wide unification.
    *   **Proposal:**
        *   Verify the example aligns perfectly with the chosen `EvaluateResult` and `MetricResult` structure (including the `success` field in `MetricResult`), as per the outcome of Approved Item 1.
        *   The example uses `from ..typed_interface import reward_function` and `from ..models import Message, EvaluateResult, MetricResult`. This is good for internal consistency.
    *   **Rationale:** Provides a clear and correct template for new reward functions.

10. **Testing Example:**
    *   **Issue:** The test example uses `unittest`, while `pytest` is the recommended tool for running tests. Assertions on a `dict` might not be ideal if `EvaluateResult` is a class.
    *   **Proposal:**
        *   Rewrite the test example using `pytest` style (e.g., plain functions, `assert` statements).
        *   If `EvaluateResult` is a class instance (likely, from `models.py`), use attribute access for assertions (e.g., `assert result.score >= 0.0`).
        *   Example:
            ```python
            # tests/test_your_module.py
            from reward_kit.rewards.your_module import your_function
            from reward_kit.models import Message, EvaluateResult # Assuming EvaluateResult is the type

            def test_your_function_basic():
                messages = [
                    {"role": "user", "content": "Test question"},
                    {"role": "assistant", "content": "Test response"}
                ]
                result: EvaluateResult = your_function(messages=messages) # Add type hint for clarity
                assert result is not None
                assert isinstance(result, EvaluateResult) # Or the specific return type
                assert result.score >= 0.0
                assert result.score <= 1.0
                # Add assertions for result.reason and result.metrics as appropriate
            ```
    *   **Rationale:** Aligns examples with tooling and promotes better testing practices.

11. **Refine "Running Tests" Comment:**
    *   **Issue:** The comment "We only care about tests/ folder for now since there are a lot of other repos" is informal.
    *   **Proposal:** Remove this comment or rephrase if necessary (e.g., "All unit and integration tests are located within the `tests/` directory.").
    *   **Rationale:** Maintains a professional tone in documentation.

12. **Clarify `FIREWORKS_ACCOUNT_ID`:**
    *   **Issue:** The comment `# For specific operations` for `FIREWORKS_ACCOUNT_ID` is vague.
    *   **Proposal:** If possible, provide a brief example or context for when this variable is needed (e.g., "Required for operations like deploying evaluators under a specific account namespace."). If it's rarely needed by typical contributors, this can be noted.
    *   **Rationale:** Helps contributors understand the purpose of all environment variables.

13. **Consistency in `black` Commands:**
    *   **Issue:** "Coding Style and Standards" mentions `black reward_kit`. "Contributing Process" step 6 mentions `black reward_kit tests`.
    *   **Proposal:** Use the more comprehensive `black reward_kit tests examples` (if examples should also be formatted) consistently, or clarify (e.g., "Format application code with `black reward_kit` and tests with `black tests`."). A single command `black .` run from the root is often simplest if all Python code should be formatted.
    *   **Rationale:** Avoids minor confusion.

14. **Pre-commit Hooks in Contributing Process:**
    *   **Issue:** `(Once configured) Run pre-commit hooks` can be more direct.
    *   **Proposal:** Change to "Ensure pre-commit hooks pass (they run automatically on commit if installed). You can also run them manually on all files: `pre-commit run --all-files`."
    *   **Rationale:** Clearer instruction.

15. **"Documentation" Section for Contributors:**
    *   **Issue:** "Update relevant files in `docs/`" is general.
    *   **Proposal:** If specific tools (e.g., Sphinx, MkDocs) or a particular workflow (e.g., "MDX files in `docs/` are compiled into our documentation website at [link-to-docs-site]") are used for documentation, briefly mention them.
    *   **Rationale:** Guides contributors on how to update documentation effectively.

### IV. General "Taste" and Usability Considerations

16. **Review `**kwargs` and `original_messages`:**
    *   **Issue:** These parameters appear in function signatures in examples but are often unused.
    *   **Proposal:**
        *   If they are part of a fixed signature that the `@reward_function` decorator or evaluation system expects, add a comment in examples explaining their general purpose, even if unused in that specific example.
        *   If they are truly optional and not always needed, consider omitting them from the most basic "hello world" style examples to reduce initial complexity, introducing them later when their utility can be shown.
    *   **Rationale:** Balances completeness with simplicity for newcomers.

17. **Code Structure Diagram:**
    *   **Action:** Keep the ASCII tree in `CONTRIBUTING.md`; it's helpful. Ensure it stays up-to-date.

This plan aims to make Reward Kit more polished and easier to engage with. Iterative implementation is recommended. 