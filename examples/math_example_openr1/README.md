# Math Example (OpenR1 Variant)

This directory contains examples for math problem evaluation and TRL training, specifically tailored for or tested with OpenR1-style datasets or models if applicable. The core functionality is similar to the main `math_example`.

## Scripts

### 1. `local_eval.py`
*   **Purpose**: Performs local evaluation of a model's responses against a dataset (potentially OpenR1 specific format if different).
*   **Configuration**: Uses Hydra and `conf/local_eval_config.yaml`.
*   **How to Run**:
    ```bash
    # Ensure your virtual environment is active
    source .venv/bin/activate
    # Navigate to the repository root
    # cd /path/to/reward-kit

    # Run with default config
    .venv/bin/python examples/math_example_openr1/local_eval.py

    # Override dataset path
    .venv/bin/python examples/math_example_openr1/local_eval.py dataset_file_path=path/to/your/openr1_dataset.jsonl
    ```
    Refer to `conf/local_eval_config.yaml`. Outputs are saved to Hydra's default output directory.

### 2. `trl_grpo_integration.py`
*   **Purpose**: Demonstrates fine-tuning a model for math problem-solving using TRL GRPO, potentially with considerations for OpenR1 data.
*   **Configuration**: Uses Hydra and `conf/trl_grpo_config.yaml`.
*   **How to Run**:
    ```bash
    # Ensure your virtual environment is active
    source .venv/bin/activate
    # Navigate to the repository root
    # cd /path/to/reward-kit

    # Run with default config
    .venv/bin/python examples/math_example_openr1/trl_grpo_integration.py

    # Override parameters
    .venv/bin/python examples/math_example_openr1/trl_grpo_integration.py dataset_file_path=my_openr1_math_train.jsonl model_name=Qwen/Qwen2-1.5B-Instruct grpo.num_train_epochs=2
    ```
    Refer to `conf/trl_grpo_config.yaml`. Outputs are saved to Hydra's default output directory.

## Dataset

A sample `dataset.jsonl` is provided. Ensure your dataset follows the expected JSONL format for these scripts, typically including "messages" and "ground_truth" fields per line.
