# Math Example

This example demonstrates how to evaluate and train models on math word problems using the GSM8K dataset. It showcases the simplified dataset architecture with on-the-fly conversion from HuggingFace datasets.

## Overview

The math example provides two different approaches to working with math problems:

1. **CLI-based Evaluation** - Use the reward-kit CLI for streamlined evaluation
2. **TRL GRPO Training** - Fine-tune models using reinforcement learning

## Key Features

- **Simplified Dataset Handling**: Direct integration with HuggingFace GSM8K dataset
- **Automatic Format Conversion**: No need for manual dataset preprocessing
- **System Prompt Integration**: Math-specific prompts built into dataset configuration
- **Flexible Evaluation**: Support for different model providers and custom reward functions

## Quick Start

### Prerequisites

```bash
# Ensure your virtual environment is active and dependencies are installed
source .venv/bin/activate
.venv/bin/pip install -e ".[dev]"

# Set up your Fireworks API credentials (if using Fireworks models)
export FIREWORKS_API_KEY="your_api_key"
export FIREWORKS_ACCOUNT_ID="your_account_id"
```

### 1. CLI-based Evaluation (Recommended)

The simplest way to run math evaluation using the reward-kit CLI:

```bash
# Navigate to the repository root
cd /path/to/reward-kit

# Run evaluation with the math configuration
python -m reward_kit.cli run --config-name run_math_eval.yaml --config-path examples/math_example/conf

# Override parameters as needed
python -m reward_kit.cli run --config-name run_math_eval.yaml --config-path examples/math_example/conf \
  generation.model_name="accounts/fireworks/models/llama-v3p1-405b-instruct" \
  evaluation_params.limit_samples=10
```

**What this does:**
- Loads GSM8K dataset directly from HuggingFace
- Applies math-specific system prompt automatically
- Generates model responses using Fireworks API
- Evaluates responses using the math reward function
- Saves detailed evaluation results to `<config_output_name>.jsonl` (e.g., `math_example_results.jsonl`) in a timestamped output directory
- Saves generated prompt/response pairs to `preview_input_output_pairs.jsonl` in the same output directory, suitable for inspection or use with `reward-kit preview`

### 2. TRL GRPO Training

Fine-tune models using reinforcement learning with the math reward function:

```bash
# Run GRPO training with default settings
.venv/bin/python examples/math_example/trl_grpo_integration.py

# Customize training parameters
.venv/bin/python examples/math_example/trl_grpo_integration.py \
  model_name="Qwen/Qwen2-1.5B-Instruct" \
  grpo.learning_rate=1e-5 \
  grpo.num_train_epochs=3
```

Configuration is managed through `conf/trl_grpo_config.yaml`.

## Dataset Configuration

The example uses a **derived dataset** approach that simplifies dataset handling:

### Base Dataset (`conf/dataset/gsm8k.yaml`)
- Defines connection to HuggingFace GSM8K dataset
- Handles column mapping from GSM8K format to standard format
- Supports different splits (train/test)

### Derived Dataset (`conf/dataset/gsm8k_math_prompts.yaml`)
- References the base GSM8K dataset
- Adds math-specific system prompt
- Converts to evaluation format with `user_query` and `ground_truth_for_eval`
- Limits samples for quick testing

**Example system prompt:**
```
Solve the following math problem. Show your work clearly. Put the final numerical answer between <answer> and </answer> tags.
```

## File Structure

```
math_example/
├── README.md                     # This file
├── main.py                       # Core evaluation logic and reward function
├── trl_grpo_integration.py       # TRL training integration
└── conf/                         # Configuration files
    ├── run_math_eval.yaml        # CLI evaluation configuration
    ├── trl_grpo_config.yaml      # TRL training configuration
    └── dataset/                  # Dataset configurations
        ├── base_dataset.yaml     # Base dataset schema
        ├── base_derived_dataset.yaml # Derived dataset schema
        ├── gsm8k.yaml            # GSM8K base dataset config
        └── gsm8k_math_prompts.yaml # GSM8K with math prompts
```

## Key Components

### Evaluation Logic (`main.py`)
- Contains the `evaluate()` function used as the reward function
- Extracts numerical answers from model responses
- Compares against ground truth with configurable tolerance
- Handles various answer formats and edge cases

### Dataset Pipeline
- **Direct HuggingFace Integration**: No need for manual dataset conversion
- **Automatic Format Conversion**: Transforms data to evaluation format on-the-fly
- **System Prompt Integration**: Prompts are part of dataset configuration, not evaluation logic
- **Flexible Column Mapping**: Adapts different dataset formats to standard interface

## Configuration Options

### Common Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `dataset` | Dataset configuration to use | `gsm8k_math_prompts` |
| `generation.model_name` | Model to evaluate | `accounts/fireworks/models/llama-v3p1-8b-instruct` |
| `evaluation_params.limit_samples` | Number of samples to evaluate | `10` |
| `reward.params.tolerance` | Numerical tolerance for answers | `0.001` |

### Dataset Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `base_dataset` | Base dataset to derive from | `gsm8k` |
| `system_prompt` | Prompt added to each query | `"Solve the math problem..."` |
| `derived_max_samples` | Limit samples in derived dataset | `5` |
| `output_format` | Format conversion type | `evaluation_format` |

## Output

The `reward-kit run` command saves its results to a timestamped directory under `outputs/`. Two main files are typically generated:

1.  **Detailed Evaluation Results (`<config_output_name>.jsonl`, e.g., `math_example_results.jsonl`)**:
    This file contains comprehensive information for each processed sample, including the original query, system prompt, generated assistant response, ground truth, the overall evaluation score, reason, and any sub-metric scores.

    Example entry from `math_example_results.jsonl`:
    ```jsonl
    {
      "id": "2389214",
      "user_query": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
      "system_prompt": "Solve the following math problem. Show your work clearly. Put the final numerical answer between <answer> and </answer> tags.",
      "assistant_response": "To find the total number of clips Natalia sold, we need to calculate the number of clips she sold in May and add it to the number of clips she sold in April...",
      "ground_truth_for_eval": "Natalia sold 48/2 = <<48/2=24>>24 clips in May...",
      "evaluation_score": 1.0,
      "evaluation_reason": "This is the eval result for the score used",
      "evaluation_metrics": {
        "accuracy_reward": {"is_score_valid": true, "score": 1.0, "reason": "This is the eval result for result accuracy"},
        "format_reward": {"is_score_valid": true, "score": 0.0, "reason": "This is the eval result for format matching"}
      }
    }
    ```

2.  **Prompt/Response Pairs for Preview (`preview_input_output_pairs.jsonl`)**:
    This file contains the system prompt, user query, and the model's generated assistant response for successfully processed samples, formatted for potential use with the `reward-kit preview` command or other analysis tools.

    Example entry from `preview_input_output_pairs.jsonl`:
    ```jsonl
    {
      "messages": [
        {"role": "system", "content": "Solve the following math problem. Show your work clearly. Put the final numerical answer between <answer> and </answer> tags."},
        {"role": "user", "content": "Natalia sold clips to 48 of her friends in April..."},
        {"role": "assistant", "content": "To find the total number of clips Natalia sold..."}
      ],
      "ground_truth": "Natalia sold 48/2 = <<48/2=24>>24 clips in May...",
      "id": "2389214"
    }
    ```

## Advanced Usage

### Custom Dataset
To use your own math dataset:

1. Create a new dataset config in `conf/dataset/my_dataset.yaml`
2. Update `run_math_eval.yaml` to reference your dataset
3. Ensure your dataset has appropriate column mappings

### Custom Reward Function
To modify the evaluation logic:

1. Edit the `evaluate()` function in `main.py`
2. Adjust parameters in the configuration files
3. Test with a small sample first

### Different Models
The example supports any model accessible through the Fireworks API:

```bash
python -m reward_kit.cli run --config-name run_math_eval.yaml --config-path examples/math_example/conf \
  generation.model_name="accounts/fireworks/models/mixtral-8x7b-instruct"
```

### Using Generated Pairs with `reward-kit preview`

The `preview_input_output_pairs.jsonl` file generated by the `run` command (located in the run's output directory, e.g., `outputs/<timestamp_dir>/preview_input_output_pairs.jsonl`) can be used as input to `reward-kit preview`. This allows you to re-evaluate the generated responses using a different evaluator, perhaps one defined by local metric scripts or a deployed remote evaluator.

```bash
# Example: Using preview with local metric scripts
# (Ensure your_metrics_folder contains a main.py with a reward function)
python -m reward_kit.cli preview \
  --samples ./outputs/<timestamp_dir>/preview_input_output_pairs.jsonl \
  --metrics-folders custom_metric_name=./path/to/your_metrics_folder

# Example: Using preview with a deployed remote evaluator
python -m reward_kit.cli preview \
  --samples ./outputs/<timestamp_dir>/preview_input_output_pairs.jsonl \
  --remote-url <your_deployed_evaluator_url>
```
**Note**: The local evaluation mechanism of `reward-kit preview --metrics-folders` is primarily designed for composing evaluators from simpler, individual metric scripts. To re-evaluate using the exact complex logic from this example's `main.py` (which is used by `reward-kit run`), you would typically point to a deployed version of that logic via `--remote-url` if you have deployed it as an evaluator. The `preview` command's local mode might have different behavior or requirements for how it loads and combines reward functions from `--metrics-folders` compared to the `run` command's single `reward.function_path`.

## Troubleshooting

### Common Issues

1. **Dataset Loading Errors**: Ensure HuggingFace datasets library is installed
2. **API Authentication**: Verify FIREWORKS_API_KEY is set correctly
3. **Memory Issues**: Reduce `limit_samples` or `derived_max_samples`
4. **Timeout Errors**: Increase API timeout settings in configuration

### Debug Mode

Enable detailed logging:

```bash
python -m reward_kit.cli run --config-name run_math_eval.yaml --config-path examples/math_example/conf \
  hydra.verbose=true
```

## Next Steps

- Explore other examples in the `examples/` directory
- Try different base models and compare performance
- Experiment with custom system prompts
- Integrate with your own datasets and reward functions

For more information about the reward kit framework, see the main [README.md](../../README.md) and [documentation](../../docs/).
