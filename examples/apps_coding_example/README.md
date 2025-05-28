# APPS Coding Example

This example demonstrates how to use the `reward-kit run` command to evaluate code generation models on a sample of the `codeparrot/apps` dataset.

## Overview

*   **Dataset:** `codeparrot/apps` - A dataset of programming problems and solutions.
*   **Task:** Given a problem description (question), generate a Python code solution.
*   **Reward Function:** `reward_kit.rewards.apps_coding_reward.evaluate_apps_solution`
    *   **Initial Version:** This reward function performs a basic check to see if the generated Python code is parsable by Python's `ast.parse` module. It scores `1.0` if parsable, and `0.0` otherwise. It does *not* execute the code or check for functional correctness against test cases in this simplified version.
    *   The `ground_truth_for_eval` field (derived from APPS' `input_output`) is provided to the reward function but not used by the initial parsability check.

## Setup

1.  **Environment:** Ensure your Python environment is set up with `reward-kit` and its dependencies installed.
2.  **API Key:** Make sure your `FIREWORKS_API_KEY` is set in your environment or in a `.env` file in the project root, as the default configuration uses a Fireworks model for generation.

## Data Preparation

The example uses a pre-generated sample of 5 prompts from the `codeparrot/apps` dataset, located at `development/apps_sample_prompts.jsonl`.

To regenerate this sample or create a different one:
1.  The script `scripts/convert_apps_to_prompts.py` is used to convert the raw Hugging Face `codeparrot/apps` dataset into the JSONL format expected by the `reward-kit run` pipeline.
2.  The source dataset configuration is defined in `conf/dataset/apps_source.yaml`.
3.  An example command to generate 5 samples from the 'test' split:
    ```bash
    python scripts/convert_apps_to_prompts.py \
        --dataset_name codeparrot/apps \
        --split test \
        --output_file development/apps_sample_prompts.jsonl \
        --max_samples 5 \
        --id_column problem_id \
        --query_column question \
        --ground_truth_column input_output
    ```

The prompt dataset configuration used by the run command is `conf/dataset/apps_prompts.yaml`, which points to `development/apps_sample_prompts.jsonl`.

## Running the Evaluation

The evaluation is configured in `examples/apps_coding_example/conf/run_eval.yaml`.

To run the evaluation:
```bash
reward-kit run --config-path examples/apps_coding_example/conf --config-name run_eval
```

You can override parameters from the command line, for example, to limit the number of samples for a quick test:
```bash
reward-kit run --config-path examples/apps_coding_example/conf --config-name run_eval evaluation_params.limit_samples=2
```

Or to disable generation and test only the reward function with existing cached responses (if any):
```bash
reward-kit run --config-path examples/apps_coding_example/conf --config-name run_eval generation.enabled=false
```

## Expected Output

The command will:
1.  Load prompts from `development/apps_sample_prompts.jsonl`.
2.  (If `generation.enabled=true`) Generate code solutions using the configured model (e.g., `firefunction-v1`). Responses will be cached in `generated_responses_cache_apps/`.
3.  Evaluate each generated solution using `evaluate_apps_solution`.
4.  Print results to the console.
5.  Save detailed results to a JSONL file in a timestamped directory under `./outputs/apps_coding_example/`, e.g., `./outputs/apps_coding_example/YYYY-MM-DD/HH-MM-SS/apps_coding_example_results.jsonl`.

The results will include the score (0 or 1 for parsability) and metrics from the reward function.
