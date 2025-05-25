# Tool Calling Example

This directory provides examples for evaluating and training models for tool/function calling capabilities.

## Scripts

### 1. `local_eval.py`
*   **Purpose**: Performs local evaluation of a model's tool calling abilities against a dataset.
*   **Configuration**: Uses Hydra and `conf/local_eval_config.yaml`.
*   **How to Run**:
    ```bash
    # Ensure your virtual environment is active
    source .venv/bin/activate
    # Navigate to the repository root
    # cd /path/to/reward-kit

    # Run with default config (expects dataset.jsonl in this directory or as specified in config)
    .venv/bin/python examples/tool_calling_example/local_eval.py

    # Override dataset path
    .venv/bin/python examples/tool_calling_example/local_eval.py dataset_file_path=path/to/your/tool_calling_dataset.jsonl
    ```
    Refer to `conf/local_eval_config.yaml`. Outputs are saved to Hydra's default output directory.

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

## Dataset

A sample `dataset.jsonl` is provided. For tool calling, each JSONL entry should typically include:
*   `messages`: A list of message objects representing the conversation history.
*   `tools`: A list of tool definitions available to the model.
*   `ground_truth`: The expected assistant response, which might include tool calls (e.g., `{"role": "assistant", "tool_calls": [...]}`) or a direct content response.

Adapt your datasets to this structure or modify the data loading and processing parts of the scripts accordingly.
