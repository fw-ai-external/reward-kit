# Math Example

This directory contains examples demonstrating:
1.  Dataset conversion for math problems.
2.  Local evaluation of models on math problems.
3.  TRL GRPO training for math problem-solving.

## Scripts

### 1. `convert_dataset.py`
*   **Purpose**: Converts a raw dataset (e.g., GSM8K) into the JSONL format expected by other scripts and the reward kit.
*   **Configuration**: Uses `conf/config.yaml`. This script also uses Hydra.
*   **How to Run**:
    ```bash
    # Ensure your virtual environment is active
    source .venv/bin/activate
    # Navigate to the repository root
    # cd /path/to/reward-kit

    # Run with default config (assumes gsm8k dataset is downloaded and paths in config.yaml are correct)
    .venv/bin/python examples/math_example/convert_dataset.py

    # Override parameters (example)
    .venv/bin/python examples/math_example/convert_dataset.py dataset_config_name=gsm8k dataset_usage.split=test dataset_usage.output_file_path=./outputs/my_converted_math_test_data.jsonl
    ```
    Refer to `conf/config.yaml` and `conf/dataset/gsm8k.yaml` (and other dataset configs) for all options.

### 2. `local_eval.py`
*   **Purpose**: Performs local evaluation of a model's responses against a dataset.
*   **Configuration**: Uses Hydra and `conf/local_eval_config.yaml`.
*   **How to Run**:
    ```bash
    # Ensure your virtual environment is active
    source .venv/bin/activate
    # Navigate to the repository root
    # cd /path/to/reward-kit

    # Run with default config (expects dataset.jsonl in the current directory, or as specified in config)
    .venv/bin/python examples/math_example/local_eval.py

    # Override dataset path and other parameters
    .venv/bin/python examples/math_example/local_eval.py dataset_file_path=path/to/your/dataset.jsonl
    ```
    Refer to `conf/local_eval_config.yaml` for configuration options. Outputs are saved to Hydra's default output directory (e.g., `outputs/YYYY-MM-DD/HH-MM-SS/` relative to the execution directory).

### 3. `trl_grpo_integration.py`
*   **Purpose**: Demonstrates fine-tuning a model for math problem-solving using TRL (Transformer Reinforcement Learning) with GRPO (Generative Rejection Policy Optimization).
*   **Configuration**: Uses Hydra and `conf/trl_grpo_config.yaml`.
*   **How to Run**:
    ```bash
    # Ensure your virtual environment is active
    source .venv/bin/activate
    # Navigate to the repository root
    # cd /path/to/reward-kit

    # Run with default config (expects dataset.jsonl, uses default model)
    .venv/bin/python examples/math_example/trl_grpo_integration.py

    # Override dataset path, model, and training parameters
    .venv/bin/python examples/math_example/trl_grpo_integration.py dataset_file_path=my_math_train.jsonl model_name=Qwen/Qwen2-1.5B-Instruct grpo.learning_rate=1e-5 grpo.num_train_epochs=3
    ```
    Refer to `conf/trl_grpo_config.yaml` for all configuration options. Outputs (logs, checkpoints) are saved to Hydra's default output directory.

## Dataset

A sample `dataset.jsonl` is provided, which is typically the output of `convert_dataset.py`. Ensure your dataset for `local_eval.py` and `trl_grpo_integration.py` follows this JSONL format, where each line is a JSON object containing at least "messages" and "ground_truth" (or fields specified in the respective configs).
