# Math with Formatting Example

This example demonstrates how to evaluate models on math word problems, considering both the accuracy of the numerical answer and the adherence to a specific response format. It uses the GSM8K dataset and showcases a multi-metric evaluation approach.

## Overview

The math with formatting example focuses on a CLI-based evaluation that assesses:

1. **Accuracy Reward**: Whether the numerical answer is correct.
2. **Format Reward**: Whether the response follows the `<think>...</think><answer>...</answer>` structure.
The final score is an average of these two rewards.

## Key Features

- **Simplified Dataset Handling**: Direct integration with HuggingFace GSM8K dataset.
- **Automatic Format Conversion**: No need for manual dataset preprocessing.
- **System Prompt Integration**: Math-specific prompts built into dataset configuration, guiding the model to produce the desired format.
- **Multi-Metric Evaluation**: Combines accuracy and formatting scores.
- **Flexible Evaluation**: Support for different model providers.

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

# Run evaluation with the math_with_formatting configuration
python -m reward_kit.cli run --config-name run_math_with_formatting_eval.yaml --config-path examples/math_with_formatting/conf

# Override parameters as needed
python -m reward_kit.cli run --config-name run_math_with_formatting_eval.yaml --config-path examples/math_with_formatting/conf \
  generation.model_name="accounts/fireworks/models/qwen3-235b-a22b" \
  evaluation_params.limit_samples=10
```

**What this does:**
- Loads GSM8K dataset directly from HuggingFace
- Applies math-specific system prompt automatically
- Generates model responses using Fireworks API
- Evaluates responses using the combined accuracy and format reward function
- Saves detailed evaluation results to `<config_output_name>.jsonl` (e.g., `math_with_formatting_example_results.jsonl`) in a timestamped output directory
- Saves generated prompt/response pairs to `preview_input_output_pairs.jsonl` in the same output directory, suitable for inspection or use with `reward-kit preview`

*(TRL GRPO Training is not the primary focus of this specific example, which emphasizes the formatting reward. Refer to the base `math_example` for TRL integration details.)*

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

**Example system prompt (from `gsm8k_math_with_formatting_prompts.yaml`):**
```
Solve the following math problem. Provide your reasoning and then put the final numerical answer between <answer> and </answer> tags.
```

## File Structure

```
math_with_formatting/
├── README.md                     # This file
├── main.py                       # Core evaluation logic (accuracy and format rewards)
└── conf/                         # Configuration files (assuming similar structure)
    ├── run_math_with_formatting_eval.yaml # CLI evaluation configuration
    └── dataset/                  # Dataset configurations
        ├── base_dataset.yaml     # (Likely shared) Base dataset schema
        ├── base_derived_dataset.yaml # (Likely shared) Derived dataset schema
        ├── gsm8k.yaml            # (Likely shared) GSM8K base dataset config
        └── gsm8k_math_with_formatting_prompts.yaml # GSM8K with formatting-specific prompts
```

## Key Components

### Evaluation Logic (`main.py`)
- Contains the `evaluate()` function which serves as the core reward logic.
- Implements `accuracy_reward_fn` to extract and compare numerical answers.
- Implements `format_reward_fn` to check if the response adheres to the `<think>...</think><answer>...</answer>` structure.
- The final `score` in the `evaluate` function is the average of `accuracy_reward` and `format_reward`.
- Returns a detailed dictionary including the overall score, validity, reason, extracted answers, and a nested `metrics` dictionary for individual reward components.

### Dataset Pipeline
- **Direct HuggingFace Integration**: No need for manual dataset conversion
- **Automatic Format Conversion**: Transforms data to evaluation format on-the-fly
- **System Prompt Integration**: Prompts are part of dataset configuration, not evaluation logic
- **Flexible Column Mapping**: Adapts different dataset formats to standard interface

## Configuration Options

### Common Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `dataset` | Dataset configuration to use | `gsm8k_math_with_formatting_prompts` |
| `generation.model_name` | Model to evaluate (default in config is `qwen3-235b-a22b`) | `accounts/fireworks/models/qwen3-235b-a22b` |
| `evaluation_params.limit_samples` | Number of samples to evaluate | `10` (default in config is `2`) |

### Dataset Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `base_dataset` | Base dataset to derive from | `gsm8k` |
| `system_prompt` | Prompt added to each query | `"Solve the math problem..."` |
| `derived_max_samples` | Limit samples in derived dataset | `5` |
| `output_format` | Format conversion type | `evaluation_format` |

## Output

The `reward-kit run` command saves its results to a timestamped directory under `outputs/`. Two main files are typically generated:

1.  **Detailed Evaluation Results (`<config_output_name>.jsonl`, e.g., `math_with_formatting_example_results.jsonl`)**:
    This file contains comprehensive information for each processed sample. The `evaluation_score` is the combined score, and `evaluation_metrics` breaks down the individual accuracy and format rewards.

    Example entry from `math_with_formatting_example_results.jsonl`:
    ```jsonl
    {
      "id": "3124980",
      "user_query": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
      "system_prompt": "Solve the following math problem. Provide your reasoning and then put the final numerical answer between <answer> and </answer> tags.",
      "assistant_response": "<think>\nOkay, let's see. Natalia sold clips to 48 friends in April. Then in May, she sold half as many as April. Hmm, so first, I need to figure out how many she sold in May. If April was 48, then half of that would be 48 divided by 2. Let me do that calculation. 48 divided by 2 is 24. So May's sales are 24 clips.\n\nNow, to find the total for both months, I need to add April's and May's sales together. That would be 48 plus 24. Let me add those numbers. 48 plus 24 equals 72. Wait, is that right? Let me check again. 48 plus 20 is 68, and then plus 4 more is 72. Yeah, that seems correct.\n\nSo altogether, Natalia sold 72 clips in April and May combined. I think that's the answer. Let me just make sure I didn't miss anything. The problem says \"half as many clips in May,\" which I interpreted as half of April's 48, so 24. Then adding them up. Yep, that seems straightforward. I don't see any other steps here. So the total should be 72.\n</think>\n\nNatalia sold 48 clips in April. In May, she sold half as many, which is $ \\frac{48}{2} = 24 $ clips. Adding both months together:  \n$ 48 + 24 = 72 $.  \n\n<answer>72</answer>",
      "ground_truth_for_eval": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
      "evaluation_score": 1.0, // This is (accuracy_reward + format_reward) / 2
      "evaluation_reason": "This is the eval result for the score used",
      "extracted_completion_answer": 72.0,
      "extracted_ground_truth_answer": 72.0,
      "evaluation_metrics": {
        "accuracy_reward": {"is_score_valid": true, "score": 1.0, "reason": "This is the eval result for result accuracy"},
        "format_reward": {"is_score_valid": true, "score": 1.0, "reason": "This is the eval result for format matching"}
      }
    }
    ```
    *(Note: If the format was incorrect, `format_reward` would be 0.0, and `accuracy_reward` would also be 0.0 due to `force_format_reward=True` in `accuracy_reward_fn`.)*

2.  **Prompt/Response Pairs for Preview (`preview_input_output_pairs.jsonl`)**:
    This file contains the system prompt, user query, and the model's generated assistant response for successfully processed samples.

    Example entry from `preview_input_output_pairs.jsonl`:
    ```jsonl
    {
      "messages": [
        {"role": "system", "content": "Solve the following math problem. Provide your reasoning and then put the final numerical answer between <answer> and </answer> tags."},
        {"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"},
        {"role": "assistant", "content": "<think>\nOkay, let's see. Natalia sold clips to 48 friends in April. Then in May, she sold half as many as April. Hmm, so first, I need to figure out how many she sold in May. If April was 48, then half of that would be 48 divided by 2. Let me do that calculation. 48 divided by 2 is 24. So May's sales are 24 clips.\n\nNow, to find the total for both months, I need to add April's and May's sales together. That would be 48 plus 24. Let me add those numbers. 48 plus 24 equals 72. Wait, is that right? Let me check again. 48 plus 20 is 68, and then plus 4 more is 72. Yeah, that seems correct.\n\nSo altogether, Natalia sold 72 clips in April and May combined. I think that's the answer. Let me just make sure I didn't miss anything. The problem says \"half as many clips in May,\" which I interpreted as half of April's 48, so 24. Then adding them up. Yep, that seems straightforward. I don't see any other steps here. So the total should be 72.\n</think>\n\nNatalia sold 48 clips in April. In May, she sold half as many, which is $ \\frac{48}{2} = 24 $ clips. Adding both months together:  \n$ 48 + 24 = 72 $.  \n\n<answer>72</answer>"}
      ],
      "ground_truth": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
      "id": "3124980"
    }
    ```

## Advanced Usage

### Custom Dataset
To use your own math dataset:

1. Create a new dataset config, e.g., in `examples/math_with_formatting/conf/dataset/my_dataset.yaml`
2. Update `run_math_with_formatting_eval.yaml` to reference your dataset
3. Ensure your dataset has appropriate column mappings

### Custom Reward Function
To modify the evaluation logic:

1. Edit the `evaluate()` function in `main.py`
2. Adjust parameters in the configuration files
3. Test with a small sample first

### Different Models
This example supports any model accessible through the Fireworks API:

```bash
python -m reward_kit.cli run --config-name run_math_with_formatting_eval.yaml --config-path examples/math_with_formatting/conf \
  generation.model_name="accounts/fireworks/models/mixtral-8x7b-instruct"
```

### Using Generated Pairs with `reward-kit preview`

The `preview_input_output_pairs.jsonl` file generated by the `run` command (located in the run's output directory, e.g., `outputs/<timestamp_dir>/preview_input_output_pairs.jsonl`) can be used as input to `reward-kit preview`. This allows you to re-evaluate the generated responses using a different evaluator, perhaps one defined by local metric scripts or a deployed remote evaluator.

```bash
# Example: Using preview with local metric scripts
# (Ensure your_metrics_folder contains a main.py with an appropriate reward function)
python -m reward_kit.cli preview \
  --samples ./outputs/<timestamp_dir>/preview_input_output_pairs.jsonl \
  --metrics-folders custom_metric_name=./path/to/your_metrics_folder # This would use a different evaluator

# Example: Using preview with a deployed remote evaluator
python -m reward_kit.cli preview \
  --samples ./outputs/<timestamp_dir>/preview_input_output_pairs.jsonl \
  --remote-url <your_deployed_evaluator_url>
```
**Note**: To re-evaluate using the exact logic from this example's `main.py` (which is used by `reward-kit run`), you would typically point to a deployed version of this logic via `--remote-url` if you have deployed it as an evaluator. The `preview` command's local mode (`--metrics-folders`) is more for composing evaluators from simpler, individual metric scripts, and might behave differently.

## Troubleshooting

### Common Issues

1. **Dataset Loading Errors**: Ensure HuggingFace datasets library is installed
2. **API Authentication**: Verify FIREWORKS_API_KEY is set correctly
3. **Memory Issues**: Reduce `limit_samples` or `derived_max_samples`
4. **Timeout Errors**: Increase API timeout settings in configuration

### Debug Mode

Enable detailed logging:

```bash
python -m reward_kit.cli run --config-name run_math_with_formatting_eval.yaml --config-path examples/math_with_formatting/conf \
  hydra.verbose=true
```

## Next Steps

- Explore other examples in the `examples/` directory
- Try different base models and compare performance
- Experiment with custom system prompts
- Integrate with your own datasets and reward functions

For more information about the reward kit framework, see the main [README.md](../../README.md) and [documentation](../../docs/).
