# HuggingFace Dataset Integration in Reward Kit CLI

This document provides examples of using the HuggingFace dataset integration with the reward-kit CLI tools.

## Prerequisites

First, ensure the necessary dependencies are installed:

```bash
pip install "reward-kit[deepseek]"
```

## Using HuggingFace Datasets in Evaluation Preview

The `preview` command now supports direct integration with HuggingFace datasets, avoiding the need to manually convert them to JSONL files.

### Basic Preview with a HuggingFace Dataset

```bash
reward-kit preview \
  --metrics-folders "quality=./examples/metrics/word_count" \
  --huggingface-dataset "deepseek-ai/DeepSeek-ProverBench" \
  --huggingface-split "test" \
  --max-samples 3
```

### Customizing Field Mappings

If your dataset uses different field names than the defaults, you can specify them:

```bash
reward-kit preview \
  --metrics-folders "quality=./examples/metrics/word_count" \
  --huggingface-dataset "cnn_dailymail" \
  --huggingface-split "test" \
  --huggingface-prompt-key "article" \
  --huggingface-response-key "highlights" \
  --max-samples 2
```

### Advanced Key Mapping

For more complex mappings, you can use the key mapping parameter:

```bash
reward-kit preview \
  --metrics-folders "quality=./examples/metrics/word_count" \
  --huggingface-dataset "squad" \
  --huggingface-split "validation" \
  --huggingface-prompt-key "question" \
  --huggingface-response-key "answers" \
  --huggingface-key-map '{"context": "context", "id": "question_id"}' \
  --max-samples 2
```

## Creating an Evaluator with HuggingFace Datasets

The `deploy` command also supports HuggingFace datasets for creating evaluators:

```bash
reward-kit deploy \
  --id "deepseek-prover-eval" \
  --metrics-folders "quality=./examples/metrics/word_count" \
  --display-name "DeepSeek Prover Evaluation" \
  --description "Evaluates proof quality using the DeepSeek-ProverBench dataset" \
  --huggingface-dataset "deepseek-ai/DeepSeek-ProverBench" \
  --huggingface-split "test" \
  --huggingface-prompt-key "statement" \
  --huggingface-response-key "reference_solution"
```

## Converting a HuggingFace Dataset to JSONL

If you need to manually convert a HuggingFace dataset to JSONL for other purposes, you can use the following Python script:

```python
from reward_kit.evaluation import huggingface_dataset_to_jsonl

# Convert a dataset to JSONL
jsonl_file = huggingface_dataset_to_jsonl(
    dataset_name="squad",
    split="validation",
    output_file="./squad_validation.jsonl",  # Specify output path
    max_samples=100,
    prompt_key="question",
    response_key="answers"
)

print(f"Converted dataset saved to: {jsonl_file}")
```