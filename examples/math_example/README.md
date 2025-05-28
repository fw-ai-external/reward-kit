# Math Example

This example demonstrates how to evaluate and train models on math word problems using the GSM8K dataset. It showcases the simplified dataset architecture with on-the-fly conversion from HuggingFace datasets.

## Overview

The math example provides three different approaches to working with math problems:

1. **CLI-based Evaluation** - Use the reward-kit CLI for streamlined evaluation
2. **Local Evaluation** - Programmatic evaluation with full control
3. **TRL GRPO Training** - Fine-tune models using reinforcement learning

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
- Saves results to timestamped output directory

### 2. Local Evaluation

For more programmatic control over the evaluation process:

```bash
# Run local evaluation with Hydra configuration
.venv/bin/python examples/math_example/local_eval.py

# Override dataset size and model
.venv/bin/python examples/math_example/local_eval.py \
  dataset_file_path=path/to/custom/dataset.jsonl \
  model_name="accounts/fireworks/models/llama-v3p1-8b-instruct"
```

Configuration is managed through `conf/local_eval_config.yaml`.

### 3. TRL GRPO Training

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
├── local_eval.py                 # Standalone local evaluation script
├── trl_grpo_integration.py       # TRL training integration
├── fireworks_preview.py          # Fireworks-specific preview functionality
├── fireworks_regenerate.py       # Fireworks response regeneration
└── conf/                         # Configuration files
    ├── run_math_eval.yaml        # CLI evaluation configuration
    ├── local_eval_config.yaml    # Local evaluation configuration
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

Results are saved to timestamped directories under `outputs/` with detailed metrics:

```jsonl
{
  "sample_id": "idx_0",
  "user_query": "Natalia sold clips to 48 of her friends...",
  "model_response": "Let me solve this step by step...",
  "ground_truth_for_eval": "72",
  "evaluation_result": {
    "score": 1.0,
    "reason": "Extracted answer 72 matches ground truth 72",
    "is_score_valid": true
  }
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
