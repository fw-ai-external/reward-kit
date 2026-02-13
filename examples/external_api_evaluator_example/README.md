# Fireworks Model as Evaluator Example

This example demonstrates how to use a Fireworks (FW) model as both the generator and the evaluator within Reward Kit. The evaluator uses a Fireworks model to perform sentiment analysis on generated responses.

## Overview

- **Generation**: Uses a Fireworks model (e.g., Qwen3-235B) to generate responses to prompts.
- **Evaluation**: Uses the same or another Fireworks model to classify the sentiment of the generated response as positive, negative, or neutral.
- **Reward**: The sentiment is mapped to a score (positive=1.0, neutral=0.5, negative=0.0) and returned as the reward.

## How It Works

1. **FireworksSentimentEvaluator** sends a prompt to the Fireworks API:
   - Prompt: `Classify the sentiment of the following text as positive, negative, or neutral. Respond with only one word: positive, negative, or neutral.`
   - The model's response is parsed and mapped to a score.
2. **Reward function** uses this evaluator to score the generated response.
3. **Reward Kit pipeline** runs as usual, using the Fireworks model for both generation and evaluation.

## Quick Start

1. **Set your Fireworks API key:**
   ```bash
   export FIREWORKS_API_KEY="your-fireworks-api-key"
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the evaluation pipeline:**
   ```bash
   python -m reward_kit.cli run --config-name external_api_eval
   ```

## Files
- `main.py` — Contains the FireworksSentimentEvaluator and reward function.
- `conf/external_api_eval.yaml` — Config for generation and evaluation.
- `test_evaluators.py` — Test script for the evaluator.
- `requirements.txt` — Dependencies.
- `README.md` — This file.

## Example: FireworksSentimentEvaluator

```python
from main import FireworksSentimentEvaluator

evaluator = FireworksSentimentEvaluator()
result = evaluator.evaluate_sentiment("I love this product!")
print(result)  # {'sentiment_label': 'positive', 'sentiment_score': 1.0}
```

## Output
- Results are saved to `outputs/external_api_evaluator/<timestamp>/eval_results.jsonl`.
- Each result includes the generated response and its sentiment-based reward score.

## Notes
- The evaluator uses the Fireworks API for classification. You can use any model available to your Fireworks account.
- The test script demonstrates direct use of the evaluator class.

## Customization
- Change the prompt in `main.py` to evaluate other criteria (e.g., toxicity, factuality).
- Use a different Fireworks model by changing the `model_name` parameter.

## Troubleshooting
- Ensure your `FIREWORKS_API_KEY` is set in the environment before running.
- If you see fallback responses, check your API key and model access. 