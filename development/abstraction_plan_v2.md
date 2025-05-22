## Revised Plan for Abstraction and Refactoring (Phased Approach)

This plan is designed to allow parallel work where possible and defers FastAPI server-side components.

**Phase 1: Core Utilities & Foundational Refactoring**

These tasks can largely be worked on independently or with minimal sequential dependency.

*   **Ticket 1.1: Basic JSONL Dataset Loader**
    *   **Description:** Create a utility function `reward_kit.utils.load_jsonl(file_path: str) -> List[Dict[str, Any]]`.
    *   **Details:** This function will read a JSONL file where each line is a valid JSON object and return a list of these objects. Include basic error handling for file not found or invalid JSON lines.
    *   **Collaboration:** Can be picked up by one engineer.
    *   **Impacts:** `examples/math_example/*_eval.py`, `examples/math_example_openr1/*_eval.py`, `examples/math_example/*/trl_grpo_integration.py`.

*   **Ticket 1.2: Math-Specific String Utilities**
    *   **Description:** Migrate math-specific string processing functions to `reward_kit.rewards.math_utils`.
    *   **Details:**
        *   Move `is_multiple_choice_question` (using `MCQ_PATTERN_REGEX`) from `examples/math_example/convert_dataset.py` to `reward_kit.rewards.math_utils.py`.
        *   Move `is_strictly_numeric` (using `_STRICTLY_NUMERIC_COMPILED_REGEX`) from `examples/math_example/convert_dataset.py` to `reward_kit.rewards.math_utils.py`.
        *   Ensure regexes are defined within or alongside these utility functions.
        *   Update `examples/math_example/convert_dataset.py` to use these new utility functions.
    *   **Collaboration:** Can be picked up by one engineer.
    *   **Impacts:** `examples/math_example/convert_dataset.py`.

*   **Ticket 1.3: Refactor `convert_dataset.py` for Modularity**
    *   **Description:** Break down `examples/math_example/convert_dataset.py` into more focused functions.
    *   **Details:**
        *   Isolate the core conversion logic `convert_math_dataset_to_openai_jsonl` but have it call out to smaller, testable functions for steps like:
            *   Filtering (e.g., MCQ filtering, numeric answer filtering).
            *   Answer formatting (e.g., applying `\\boxed{}`).
        *   This is a preparatory step for potentially abstracting some of these smaller functions later if they prove to be highly reusable beyond this specific script. For now, the goal is improved readability and testability within the example.
    *   **Collaboration:** Can be picked up by one engineer, ideally after Ticket 1.2.
    *   **Impacts:** `examples/math_example/convert_dataset.py`.

*   **Ticket 1.4: Client-Side Fireworks API Interaction Utility**
    *   **Description:** Create a basic synchronous/asynchronous client or utility function for Fireworks AI chat completions.
    *   **Details:**
        *   Initial focus on `generate_with_fireworks_inner` logic from `examples/math_example/fireworks_regenerate.py`.
        *   Provide a function like `reward_kit.integrations.fireworks.call_chat_completion(messages: List[Dict], model: str, api_key: str, api_base_url: str, **generation_params) -> Optional[Dict]`.
        *   Include payload construction, headers, and basic error handling for API calls (e.g., `requests.exceptions.RequestException`).
        *   Initially, this can be a synchronous version using `requests`. An async version with `aiohttp` and semaphore (from the original example) can be a follow-up (Ticket 2.2).
    *   **Collaboration:** Can be picked up by one engineer.
    *   **Impacts:** `examples/math_example/fireworks_regenerate.py`, `examples/math_example_openr1/fireworks_regenerate.py`.

*   **Ticket 1.5: Recorded Data Playback Utility for API Mocking**
    *   **Description:** Create a utility to mock API responses by playing back recorded data from a file.
    *   **Details:**
        *   Develop `reward_kit.testing_utils.APIMocker` class or a set of functions.
        *   Method: `load_responses_from_jsonl(file_path: str, prompt_field: str, response_field: str)`.
        *   Method: `get_mock_response(prompt: str) -> Optional[str]`.
        *   This will be used to replace live API calls during tests or local runs.
    *   **Collaboration:** Can be picked up by one engineer.
    *   **Impacts:** `examples/math_example/fireworks_regenerate.py`, `examples/math_example_openr1/fireworks_regenerate.py`. This utility will also be useful for testing the client from Ticket 1.4.

**Phase 2: Enhancements & Integrations**

These tasks build upon Phase 1 and focus on broader usability.

*   **Ticket 2.1: Advanced Dataset Loader for Conversations**
    *   **Description:** Enhance dataset loading with a more structured conversation loader.
    *   **Details:**
        *   Create `reward_kit.data_utils.load_conversation_dataset(file_path: str, user_role: str = "user", assistant_role: str = "assistant", messages_field: str = "messages", ground_truth_field: Optional[str] = None) -> List[Dict]`.
        *   This function would parse JSONL files into a list of dictionaries, where each dictionary represents a sample and contains structured `messages` (e.g., list of `Message` objects or dicts) and an optional `ground_truth`.
        *   This builds on Ticket 1.1.
    *   **Collaboration:** Can be picked up by one engineer.
    *   **Impacts:** All examples that load datasets.

*   **Ticket 2.2: Asynchronous Fireworks API Client with Concurrency Control**
    *   **Description:** Enhance the Fireworks API client (from Ticket 1.4) with asynchronous capabilities and semaphore for concurrency management.
    *   **Details:**
        *   Refactor the client/utility to use `aiohttp` and `asyncio.Semaphore` as seen in `fireworks_regenerate.py`.
        *   Ensure it can be easily used by scripts that need to make many API calls concurrently.
    *   **Collaboration:** Can be picked up by one engineer, depends on Ticket 1.4.
    *   **Impacts:** `examples/math_example/fireworks_regenerate.py`, `examples/math_example_openr1/fireworks_regenerate.py`.

*   **Ticket 2.3: TRL Reward Function Adapter**
    *   **Description:** Create a generic adapter for using `reward-kit` reward functions with TRL.
    *   **Details:**
        *   Develop `reward_kit.integrations.trl.TRLRewardAdapter`.
        *   Constructor: `__init__(self, reward_function: Callable, ground_truth_dataset_field: str = "response", prompt_dataset_field: str = "prompt", completion_dataset_field: Optional[str] = None)`. The `completion_dataset_field` would be if the dataset *already* has completions, otherwise they come from the TRL generation.
        *   `__call__(self, prompts: List[str], completions: List[str], **kwargs_from_dataset) -> List[torch.Tensor]`.
        *   This adapter will handle extracting ground truth from `kwargs_from_dataset` (which TRL passes from the dataset columns) and formatting inputs/outputs for the wrapped `reward_function`.
    *   **Collaboration:** Can be picked up by one engineer.
    *   **Impacts:** `examples/math_example/trl_grpo_integration.py`, `examples/math_example_openr1/trl_grpo_integration.py`.

*   **Ticket 2.4: General Math Answer Formatting Utility**
    *   **Description:** Create a more general utility for formatting math answers if common patterns emerge beyond just `\\boxed{}`.
    *   **Details:**
        *   Based on the refactoring in Ticket 1.3, if the answer formatting logic (e.g., `final_assistant_content` generation in `convert_dataset.py`) seems broadly applicable, abstract it to `reward_kit.rewards.math_utils.format_math_answer(answer_value: Union[float, int, str], original_solution_text: Optional[str] = None, desired_format: str = "boxed_numerical_with_original_solution_fallback") -> str`.
        *   This is more speculative and depends on findings from Ticket 1.3.
    *   **Collaboration:** Can be picked up by one engineer, depends on insights from Ticket 1.3.

**Phase 3: Standardization & Developer Experience**

Focuses on making examples easier to use and maintain.

*   **Ticket 3.1: Standardize Example Configuration**
    *   **Description:** Define and document a consistent approach for managing configuration in examples.
    *   **Details:**
        *   Recommend a pattern (e.g., using a `config.py` file per example, or a clear convention for environment variables like `FIREWORKS_API_KEY`, `MODEL_NAME`).
        *   Update existing examples to follow this pattern.
    *   **Collaboration:** Can be a cross-cutting effort or assigned to one engineer to enforce consistency.
    *   **Impacts:** All examples.

*   **Ticket 3.2: Create Example Templates & Documentation**
    *   **Description:** Develop templates for common example scripts and document a standard example structure.
    *   **Details:**
        *   Create base templates for `local_eval.py`, `fireworks_preview.py` (if still relevant after API client changes), `data_generation_script.py` (like `fireworks_regenerate.py`), and `trl_integration.py`.
        *   Document how to structure an example directory, where to place data, configs, and scripts.
    *   **Collaboration:** Can be picked up by one engineer or a documentation-focused team member.
    *   **Impacts:** Developer documentation, future example creation.

*   **Ticket 3.3: Review and Relocate Development Utilities**
    *   **Description:** Assess `development/utils/subprocess_manager.py` and `development/utils/generate_api_key.py` for broader utility.
    *   **Details:**
        *   If `subprocess_manager.py` (for managing background processes like mock servers or tunnels) and `generate_api_key.py` are deemed useful for general testing or more complex examples that will be part of the main library, consider moving them to `reward_kit.testing_utils` or `reward_kit.utils.dev_helpers`.
        *   The Serveo tunneling logic in `subprocess_manager.py` is quite specific; evaluate if a more generic tunneling helper is needed or if this remains a dev-specific tool.
    *   **Collaboration:** Requires discussion and then one engineer for implementation.
    *   **Impacts:** `examples/remote_eval_demo/run_demo.py`, potentially testing infrastructure.

**Deferred / Lower Priority (Regarding FastAPI Server-Side Components):**

*   **Mock API Server Framework (FastAPI based):**
    *   **Original Idea:** "The `mock_api_service.py` in `remote_eval_demo` is a good, simple FastAPI example. If many reward functions will involve calling external services, providing a lightweight, configurable mock server utility within `reward_kit.testing_utils` could be beneficial."
    *   **Current Status:** This task, specifically creating a reusable FastAPI *server* component within `reward-kit` for mocking reward function dependencies, will be deferred or handled by the other team/effort focusing on FastAPI for reward functions.
    *   **Note:** The client-side utilities for *calling* APIs (Ticket 1.4, 2.2) and the `APIMocker` for *simulating* responses from recorded data (Ticket 1.5) are still valuable and distinct from this.
