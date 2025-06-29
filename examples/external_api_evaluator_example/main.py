"""Hugging Face Inference API evaluator for Reward Kit."""

import requests
import os
from reward_kit.typed_interface import reward_function

class FireworksSentimentEvaluator:
    """Evaluator that uses a Fireworks model for sentiment analysis."""
    def __init__(self, api_key=None, model_name="accounts/fireworks/models/qwen3-235b-a22b"):
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        self.model_name = model_name
        self.api_url = "https://api.fireworks.ai/inference/v1/completions"
        if not self.api_key:
            raise ValueError("Fireworks API key not set. Set FIREWORKS_API_KEY environment variable or pass api_key explicitly.")

    def evaluate_sentiment(self, text: str) -> dict:
        prompt = (
            "Classify the sentiment of the following text as positive, negative, or neutral. "
            "Respond with only one word: positive, negative, or neutral.\n\n"
            f"Text: {text}\nSentiment:"
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            sentiment = result["choices"][0]["text"].strip().lower()
        except Exception as e:
            print(f"Fireworks API call failed: {e}. Using fallback response.")
            sentiment = "neutral"
        # Convert to score
        if sentiment == "positive":
            score = 1.0
        elif sentiment == "negative":
            score = 0.0
        else:
            score = 0.5
        return {"sentiment_label": sentiment, "sentiment_score": score}

def get_message_content(message, key: str) -> str:
    if isinstance(message, dict):
        return message.get(key, "")
    else:
        return getattr(message, key, "")

# Initialize the evaluator
sentiment_evaluator = FireworksSentimentEvaluator()

def sentiment_scorer(input: str, output: str, expected: str = None) -> float:
    """
    Score based on sentiment analysis of the output using Fireworks model.
    """
    sentiment_result = sentiment_evaluator.evaluate_sentiment(output)
    return sentiment_result.get("sentiment_score", 0.5)

@reward_function
def evaluate(messages, ground_truth=None, **kwargs):
    """
    Reward Kit evaluate function using Fireworks model for sentiment analysis.
    """
    if not messages:
        return {"score": 0.0}
    # Find the last assistant message
    last_assistant_message = None
    for message in reversed(messages):
        role = get_message_content(message, "role")
        if role == "assistant":
            last_assistant_message = get_message_content(message, "content")
            break
    if not last_assistant_message:
        return {"score": 0.0}
    # Get the user input (first user message)
    user_input = ""
    for message in messages:
        role = get_message_content(message, "role")
        if role == "user":
            user_input = get_message_content(message, "content")
            break
    score = sentiment_scorer(input=user_input, output=last_assistant_message, expected=ground_truth)
    return {"score": score}

# Example usage with real API tokens
def setup_with_real_apis(hf_token: str):
    """
    Setup evaluators with real Hugging Face API tokens.
    
    Args:
        hf_token: Your Hugging Face API token
    """
    global sentiment_evaluator
    
    sentiment_evaluator = FireworksSentimentEvaluator(api_key=hf_token)
    
    print("External API evaluators configured with real Fireworks API.") 