# Tool Calling Example

This directory provides examples for evaluating and training models for tool/function calling capabilities, primarily using the `Salesforce/xlam-function-calling-60k` dataset.

## Dataset Setup: Salesforce/xlam-function-calling-60k

This example is configured to use the `Salesforce/xlam-function-calling-60k` dataset from Hugging Face. The setup involves:

1.  **Base Dataset Configuration (`conf/dataset/xlam_fc_source.yaml`)**: Defines how to load the raw dataset from Hugging Face.
2.  **Derived Dataset Configuration (`conf/dataset/xlam_fc_eval_prompts.yaml`)**: Transforms the raw data into the format required for evaluation. This includes:
    *   Parsing JSON strings for `tools` and `answers` columns from the source dataset.
    *   Formatting the `query` into a `messages` list (e.g., `[{"role": "user", "content": query}]`).
    *   Formatting the `answers` (expected tool calls) into a `ground_truth` object (e.g., `{"role": "assistant", "tool_calls": [...]}`).
3.  **Custom Processors (`examples/tool_calling_example/custom_processors.py`)**: Contains Python helper functions (`safe_json_loads`, `format_messages_for_eval`, `format_ground_truth_for_eval`) used by `xlam_fc_eval_prompts.yaml` for data transformation.

The reward function used is `reward_kit.rewards.function_calling.exact_tool_match_reward`, which performs an exact comparison of generated tool calls against the ground truth.

## Running Evaluation

The primary way to run evaluation for this example is using the `reward_kit.cli run` command with the provided Hydra configuration.

*   **Purpose**: Performs model response generation and evaluation for tool calling abilities using the `Salesforce/xlam-function-calling-60k` dataset.
*   **Configuration**: Uses Hydra and `conf/run_xlam_fc_eval.yaml`. This configuration specifies:
    *   The dataset to use (`xlam_fc_eval_prompts`).
    *   The model for generation (e.g., `mistralai/Mistral-7B-Instruct-v0.2` - **please update this to your desired model**).
    *   The reward function (`exact_tool_match_reward`).
*   **How to Run**:
    ```bash
    # Ensure your virtual environment is active
    source .venv/bin/activate
    # Navigate to the repository root
    # cd /path/to/reward-kit

    # Run evaluation using the CLI
    python -m reward_kit.cli run --config-path examples/tool_calling_example/conf --config-name run_xlam_fc_eval.yaml

    # You can override parameters from the command line, e.g., to change the model:
    python -m reward_kit.cli run --config-path examples/tool_calling_example/conf --config-name run_xlam_fc_eval.yaml generation.model_id="your-model-hf-id"
    ```
    Outputs, including evaluation results and preview samples, will be saved to the directory specified in `hydra.run.dir` within `run_xlam_fc_eval.yaml` (e.g., `./outputs/tool_calling_example/xlam_fc_eval/...`).

## Other Scripts

### 1. `local_eval.py` (Alternative/Legacy)
*   **Purpose**: Performs local evaluation if you have a pre-processed dataset in JSONL format. This script does *not* use the full Hydra dataset pipeline (`xlam_fc_source.yaml`, `xlam_fc_eval_prompts.yaml`) directly for data loading but expects a path to a JSONL file.
*   **Configuration**: Uses Hydra and `conf/local_eval_config.yaml`.
*   **Note**: For the `Salesforce/xlam-function-calling-60k` dataset, prefer the `reward_kit.cli run` method described above. `local_eval.py` could be adapted or used if you manually prepare a JSONL file in the expected format.

### 2. `trl_grpo_integration.py`
*   **Purpose**: A scaffold script demonstrating how to fine-tune a model for tool calling using TRL GRPO.
    **Note**: This script currently uses a MOCK model and tokenizer by default for demonstration and requires further implementation to use a real model (see `use_mock_model_tokenizer` in the config or script).
*   **Configuration**: Uses Hydra and `conf/trl_grpo_config.yaml`.
*   **How to Run (with Mock Model)**:
    ```bash
    # Ensure your virtual environment is active
    source .venv/bin/activate
    # Navigate to the repository root
    # cd /path/to/reward-kit

    # Run with default config (uses mock model)
    .venv/bin/python examples/tool_calling_example/trl_grpo_integration.py

    # Override parameters (e.g., dataset, mock model behavior if extended)
    .venv/bin/python examples/tool_calling_example/trl_grpo_integration.py dataset_file_path=my_tool_train.jsonl grpo.num_train_epochs=1
    ```
*   **To Run with a Real Model (Requires Code Changes)**:
    1.  Modify `trl_grpo_integration.py` to load your desired Hugging Face model and tokenizer (comment out or conditionalize the MockModel/MockTokenizer part).
    2.  Ensure your model is suitable for tool calling and the `format_prompt_and_extract_ground_truth` function in the script correctly prepares prompts for it.
    3.  Update `conf/trl_grpo_config.yaml` with the correct `model_name` and any other relevant training parameters.
    4.  Run the script, potentially disabling the mock model via config override if you added such a flag:
        `.venv/bin/python examples/tool_calling_example/trl_grpo_integration.py +use_mock_model_tokenizer=false ...other_overrides`

    Refer to `conf/trl_grpo_config.yaml` for configuration options. Outputs are saved to Hydra's default output directory.

## Dataset Format Notes

The `Salesforce/xlam-function-calling-60k` dataset, after processing by `xlam_fc_eval_prompts.yaml` and `custom_processors.py`, will yield items with the following key structure for evaluation:

*   `messages`: A list containing a single user message, e.g., `[{"role": "user", "content": "User's query string"}]`.
*   `tools`: A list of tool definition objects, parsed directly from the `tools` field of the source dataset. Each tool object typically includes `name`, `description`, and `parameters`.
*   `ground_truth`: An object representing the expected assistant's response, e.g., `{"role": "assistant", "content": null, "tool_calls": [{"name": "tool_name", "arguments": {...}}]}`. The `tool_calls` array is parsed from the `answers` field of the source dataset.

The `dataset.jsonl` file included in this directory is a small sample and is primarily used by `local_eval.py` if run with its default configuration. For the main evaluation path using `reward_kit.cli run`, the data is sourced and processed dynamically from Hugging Face as per the YAML configurations.
