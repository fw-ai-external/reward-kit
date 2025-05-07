"""
Example demonstrating how to use reward-kit reward functions with TRL's PPO trainer.

This example shows how to:
1. Define a simple reward function in reward-kit
2. Convert it to TRL-compatible format
3. Use it with the PPO trainer
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure reward-kit is in the path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

# Import reward-kit components
from reward_kit.reward_function import RewardFunction, reward_function
from reward_kit.models import (
    RewardOutput,
    MetricRewardOutput,
    EvaluateResult,
    MetricResult,
)

# Try to import TRL components
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
    from trl.core import respond_to_batch
    from datasets import load_dataset

    HAS_TRL = True
except ImportError:
    print(
        "TRL or related packages not installed. Install with: pip install trl datasets"
    )
    HAS_TRL = False


# Define a simple reward function compatible with reward-kit
@reward_function
def helpfulness_reward(
    messages: List[Dict[str, Any]],
    original_messages: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> EvaluateResult:
    """
    Reward function that evaluates helpfulness based on response length and keywords.

    This is a simplified example - a real helpfulness metric would be more sophisticated.

    Args:
        messages: List of conversation messages
        original_messages: Original messages for context

    Returns:
        EvaluateResult with score based on helpfulness
    """
    # Get the assistant's message
    if not messages or len(messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            metrics={
                "helpfulness": MetricResult(
                    score=0.0, success=False, reason="No messages provided"
                )
            },
        )

    # Extract response text
    response = messages[-1]
    if response.get("role") != "assistant" or not response.get("content"):
        return EvaluateResult(
            score=0.0,
            reason="No assistant response found",
            metrics={
                "helpfulness": MetricResult(
                    score=0.0, success=False, reason="No assistant response"
                )
            },
        )

    text = response.get("content", "")

    # Calculate score based on length (simple heuristic)
    word_count = len(text.split())

    # Normalize length score between 0-1 with an ideal range
    if word_count < 10:
        length_score = 0.2  # Too short
    elif word_count < 50:
        length_score = 0.5 + (word_count - 10) * 0.01  # Linear increase
    elif word_count <= 200:
        length_score = 1.0  # Ideal length
    else:
        length_score = (
            1.0 - (word_count - 200) * 0.002
        )  # Gradually decrease for verbosity
        length_score = max(0.3, length_score)  # Don't go below 0.3

    # Check for helpful phrases (simple keyword heuristic)
    helpful_phrases = [
        "here's how",
        "you can",
        "for example",
        "explanation",
        "step",
        "process",
        "method",
        "approach",
        "solution",
        "answer",
        "result",
    ]

    helpful_count = sum(
        1 for phrase in helpful_phrases if phrase.lower() in text.lower()
    )
    helpfulness_score = min(1.0, helpful_count / 5)  # Normalize to 0-1

    # Combine scores (70% length, 30% helpful phrases)
    combined_score = 0.7 * length_score + 0.3 * helpfulness_score

    # Prepare reason text
    reason = (
        f"Length score: {length_score:.2f} ({word_count} words), "
        f"Helpful phrases: {helpfulness_score:.2f} ({helpful_count} phrases)"
    )

    return EvaluateResult(
        score=combined_score,
        reason=reason,
        metrics={
            "length": MetricResult(
                score=length_score,
                success=length_score > 0.7,
                reason=f"Response length: {word_count} words",
            ),
            "helpful_phrases": MetricResult(
                score=helpfulness_score,
                success=helpfulness_score > 0.5,
                reason=f"Helpful phrases: {helpful_count} found",
            ),
        },
    )


def prepare_dataset_for_ppo(dataset_name, split="train", max_samples=None):
    """
    Prepare a HuggingFace dataset for use with PPO.

    Args:
        dataset_name: Name of the HuggingFace dataset
        split: Dataset split to use
        max_samples: Maximum samples to include

    Returns:
        Dataset in PPO-compatible format
    """
    if not HAS_TRL:
        print(
            "TRL or related packages not installed. Install with: pip install trl datasets"
        )
        return None

    # Load dataset
    dataset = load_dataset(dataset_name, split=split)

    # Limit samples if specified
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    # For PPO we just need query/text pairs - default to assume it's a summarization dataset
    def prepare_sample(example):
        return {
            "query": example.get("article", example.get("document", "")),
            "input_ids": None,  # This will be filled by the PPO Trainer
        }

    formatted_dataset = dataset.map(prepare_sample)

    # Keep only the needed columns
    if "query" in formatted_dataset.column_names:
        formatted_dataset = formatted_dataset.remove_columns(
            [c for c in formatted_dataset.column_names if c != "query"]
        )

    return formatted_dataset


def train_with_ppo_example():
    """
    Example of training with PPO using a reward-kit reward function.
    """
    if not HAS_TRL:
        print(
            "TRL or related packages not installed. Install with: pip install trl datasets"
        )
        return

    print("Setting up PPO training with a reward-kit reward function...")

    # 1. Create reward function and adapter
    reward_fn = RewardFunction(func=helpfulness_reward)
    reward_adapter = reward_fn.get_trl_adapter()

    # 2. Prepare dataset (use a simple summarization dataset)
    try:
        print("Preparing dataset...")
        dataset = prepare_dataset_for_ppo(
            dataset_name="cnn_dailymail",
            split="test[:1%]",  # Use a tiny subset for demonstration
            max_samples=5,
        )
        print(f"Dataset prepared: {len(dataset)} samples")
    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")
        return

    # 3. Set up model (would be used in real training)
    if False:  # Skip for example purposes
        print("Setting up model...")
        # Load model and tokenizer
        model_name = "gpt2"  # Use a small model for example
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Configure PPO
        ppo_config = PPOConfig(
            learning_rate=1e-5,
            batch_size=1,
            mini_batch_size=1,
            gradient_accumulation_steps=1,
            optimize_cuda_cache=True,
            early_stopping=False,
            target_kl=0.1,
            ppo_epochs=4,
            seed=42,
        )

        # Initialize PPO trainer
        ppo_trainer = PPOTrainer(
            config=ppo_config, model=model, tokenizer=tokenizer, dataset=dataset
        )

        # Generate responses and compute rewards
        for _ in range(1):  # In real training, you'd do more iterations
            # Generate model responses
            query_tensors = ppo_trainer.prepare_sample(
                dataset["query"], truncation=True, max_length=256
            )
            response_tensors = respond_to_batch(
                ppo_trainer.model,
                query_tensors,
                ppo_trainer.tokenizer,
                max_new_tokens=64,
            )

            # Decode responses and format for reward function
            responses = [
                ppo_trainer.tokenizer.decode(r.squeeze())
                for r in response_tensors
            ]
            texts = [{"role": "assistant", "content": r} for r in responses]

            # Compute rewards using our adapter
            rewards = reward_adapter(texts)

            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            print(f"Rewards: {rewards}")
            print(f"Stats: {stats}")

    print(
        "\nExample completed successfully. In a real scenario, the PPO training would now run."
    )
    print(
        "This example shows how a reward-kit reward function can be adapted for TRL's PPO trainer."
    )

    # Show how the reward function would be called
    print("\nReward function test on sample data:")
    sample_messages = [
        {"role": "user", "content": "Explain how to make cookies"},
        {
            "role": "assistant",
            "content": "Here's how to make chocolate chip cookies: First, you'll need flour, sugar, butter, eggs, and chocolate chips. The process has several steps. Start by creaming together the butter and sugar, then add eggs. Next, combine flour with baking soda and salt, then mix into the wet ingredients. Finally, fold in chocolate chips and bake at 375°F for 10-12 minutes. For the best results, let them cool for 5 minutes before enjoying.",
        },
    ]

    reward_result = helpfulness_reward(sample_messages)
    print(
        f"Helpfulness reward score: {reward_result.score} - {reward_result.reason}"
    )

    # Show how the adapter formats for TRL
    adapted_reward = reward_adapter([sample_messages])
    print(f"TRL adapter converted reward: {adapted_reward}")


if __name__ == "__main__":
    train_with_ppo_example()
