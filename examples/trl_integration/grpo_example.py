"""
Example demonstrating how to use reward-kit reward functions with TRL's GRPO trainer.

This example shows how to:
1. Define reward functions in reward-kit
2. Convert them to TRL-compatible format
3. Use them with the GRPO trainer
"""

import os
import sys
import re
from pathlib import Path
import torch
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
    from trl import GRPOConfig, GRPOTrainer
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model

    HAS_TRL = True
except ImportError:
    print(
        "TRL or related packages not installed. Install with: pip install trl peft datasets"
    )
    HAS_TRL = False


# Define reward functions compatible with reward-kit
@reward_function
def format_reward(
    messages: List[Dict[str, Any]],
    original_messages: Optional[List[Dict[str, Any]]] = None,
    think_tag: str = "<think>",
    answer_tag: str = "<answer>",
    **kwargs,
) -> EvaluateResult:
    """
    Reward function that checks if the completion has the GRPO specific format.

    Args:
        messages: List of conversation messages
        original_messages: Original messages for context
        think_tag: Tag to use for reasoning (default: "<think>")
        answer_tag: Tag to use for answers (default: "<answer>")

    Returns:
        EvaluateResult with score based on format compliance
    """
    # Get the assistant's message
    if not messages or len(messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            metrics={
                "format": MetricResult(
                    score=0.0, success=False, reason="No messages provided"
                )
            },
        )

    # Extract response text from last message (assistant's response)
    response = messages[-1]
    if response.get("role") != "assistant" or not response.get("content"):
        return EvaluateResult(
            score=0.0,
            reason="No assistant response found",
            metrics={
                "format": MetricResult(
                    score=0.0, success=False, reason="No assistant response"
                )
            },
        )

    text = response.get("content", "")

    # Check for think/answer tags
    think_pattern = (
        f"{re.escape(think_tag)}(.*?){re.escape(think_tag.replace('<', '</'))}"
    )
    answer_pattern = f"{re.escape(answer_tag)}(.*?){re.escape(answer_tag.replace('<', '</'))}"

    think_match = re.search(think_pattern, text, re.DOTALL)
    answer_match = re.search(answer_pattern, text, re.DOTALL)

    has_think = bool(think_match)
    has_answer = bool(answer_match)

    # Check for correct order (think should come before answer)
    correct_order = True
    if has_think and has_answer:
        think_pos = text.find(think_tag)
        answer_pos = text.find(answer_tag)
        correct_order = think_pos < answer_pos

    # Calculate score based on format compliance
    if has_think and has_answer and correct_order:
        score = 1.0
        reason = "Format is compliant with think/answer tags in correct order"
    elif has_think and has_answer:
        score = 0.5
        reason = "Has both think and answer tags but in incorrect order"
    elif has_think:
        score = 0.3
        reason = "Has think tag but missing answer tag"
    elif has_answer:
        score = 0.2
        reason = "Has answer tag but missing think tag"
    else:
        score = 0.0
        reason = "Missing both think and answer tags"

    # Create metrics
    metrics = {
        "has_think": MetricResult(
            score=1.0 if has_think else 0.0,
            success=has_think,
            reason=f"{'Has' if has_think else 'Missing'} think tag",
        ),
        "has_answer": MetricResult(
            score=1.0 if has_answer else 0.0,
            success=has_answer,
            reason=f"{'Has' if has_answer else 'Missing'} answer tag",
        ),
        "correct_order": MetricResult(
            score=1.0 if correct_order else 0.0,
            success=correct_order,
            reason=f"Tags are in {'correct' if correct_order else 'incorrect'} order",
        ),
    }

    return EvaluateResult(score=score, reason=reason, metrics=metrics)


@reward_function
def math_accuracy_reward(
    messages: List[Dict[str, Any]],
    original_messages: Optional[List[Dict[str, Any]]] = None,
    solution: Optional[str] = None,
    **kwargs,
) -> EvaluateResult:
    """
    Reward function that checks if the math solution is correct.

    Args:
        messages: List of conversation messages
        original_messages: Original messages for context
        solution: Expected solution/answer

    Returns:
        EvaluateResult with score based on solution accuracy
    """
    # In a real implementation, we would:
    # 1. Extract the answer from the text
    # 2. Compare it with the expected solution
    # 3. Calculate a similarity score

    # For this example, we'll use a simplified implementation
    if not messages or len(messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            metrics={
                "accuracy": MetricResult(
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
                "accuracy": MetricResult(
                    score=0.0, success=False, reason="No assistant response"
                )
            },
        )

    text = response.get("content", "")

    # Simplified implementation: check if solution appears in the response
    if solution and solution.lower() in text.lower():
        score = 1.0
        reason = f"Solution '{solution}' found in response"
        success = True
    else:
        # Normally we would do a more sophisticated comparison
        # For the example, we're using a random score
        import random

        score = random.uniform(0.3, 0.8)
        reason = f"Solution '{solution}' not directly found, partial match assessment"
        success = score > 0.7

    return EvaluateResult(
        score=score,
        reason=reason,
        metrics={
            "accuracy": MetricResult(
                score=score, success=success, reason=reason
            )
        },
    )


def combine_rewards(reward_functions, weights=None):
    """
    Combine multiple reward functions into a single TRL-compatible function.

    Args:
        reward_functions: List of RewardFunction instances
        weights: Optional weights for each reward function (normalized if not summing to 1)

    Returns:
        A callable function compatible with TRL
    """
    # Normalize weights if provided
    if weights:
        if len(weights) != len(reward_functions):
            raise ValueError(
                "Number of weights must match number of reward functions"
            )
        weight_sum = sum(weights)
        if weight_sum != 1.0:
            weights = [w / weight_sum for w in weights]
    else:
        # Equal weights for all reward functions
        weights = [
            1.0 / len(reward_functions) for _ in range(len(reward_functions))
        ]

    # Create adapters for each reward function
    adapters = [rf.get_trl_adapter() for rf in reward_functions]

    def combined_adapter(batch_input, batch_orig_input=None, **adapter_kwargs):
        """Combined adapter function that works with TRL."""
        # Collect scores from all reward functions
        all_scores = []
        for i, adapter in enumerate(adapters):
            scores = adapter(batch_input, batch_orig_input, **adapter_kwargs)
            all_scores.append(scores)

        # Combine weighted scores for each sample
        combined_scores = []
        for i in range(len(all_scores[0])):
            weighted_sum = sum(
                scores[i] * weight
                for scores, weight in zip(all_scores, weights)
            )
            combined_scores.append(weighted_sum)

        return combined_scores

    return combined_adapter


def prepare_dataset_for_trl(
    dataset_name,
    split="train",
    prompt_key=None,
    response_key=None,
    system_prompt=None,
    max_samples=None,
):
    """
    Prepare a HuggingFace dataset for use with TRL.

    Args:
        dataset_name: Name of the HuggingFace dataset
        split: Dataset split to use
        prompt_key: Key for the prompt content
        response_key: Key for the response content
        system_prompt: Optional system prompt to prepend
        max_samples: Maximum samples to include

    Returns:
        Dataset in TRL-compatible format
    """
    if not HAS_TRL:
        print(
            "TRL or related packages not installed. Install with: pip install trl peft datasets"
        )
        return None

    # Load dataset
    dataset = load_dataset(dataset_name, split=split)

    # Limit samples if specified
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    # Default keys for prompt and response
    if prompt_key is None:
        # Try to guess based on common patterns
        for key in ["problem", "question", "input", "prompt"]:
            if key in dataset.features:
                prompt_key = key
                break
        if prompt_key is None:
            raise ValueError(
                "Could not determine prompt key. Please specify prompt_key."
            )

    if response_key is None:
        # Try to guess based on common patterns
        for key in ["solution", "answer", "output", "response"]:
            if key in dataset.features:
                response_key = key
                break
        if response_key is None:
            raise ValueError(
                "Could not determine response key. Please specify response_key."
            )

    # Prepare GRPO style system prompt
    if system_prompt is None:
        system_prompt = (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
            "The assistant first thinks about the reasoning process in the mind and then provides the user "
            "with the answer. The reasoning process and answer are enclosed within <think> </think> and "
            "<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>"
            "<answer> answer here </answer>"
        )

    # Create the dataset in the format expected by TRL
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example[prompt_key]},
            ],
            "solution": example[response_key],
        }

    formatted_dataset = dataset.map(make_conversation)

    # Remove unnecessary columns but keep solution for reward function
    cols_to_remove = [
        col
        for col in formatted_dataset.column_names
        if col not in ["prompt", "solution"]
    ]
    if cols_to_remove:
        formatted_dataset = formatted_dataset.remove_columns(cols_to_remove)

    return formatted_dataset


def train_with_grpo_example():
    """
    Example of training with GRPO using reward-kit reward functions.
    """
    if not HAS_TRL:
        print(
            "TRL or related packages not installed. Install with: pip install trl peft datasets"
        )
        return

    print("Setting up GRPO training with reward-kit reward functions...")

    # 1. Create reward functions
    format_reward_fn = RewardFunction(func=format_reward)
    accuracy_reward_fn = RewardFunction(func=math_accuracy_reward)

    # 2. Prepare dataset
    try:
        print("Preparing dataset...")
        dataset = prepare_dataset_for_trl(
            dataset_name="AI-MO/NuminaMath-TIR",
            split="train[:1%]",  # Use a small subset for demonstration
            prompt_key="problem",
            response_key="solution",
            max_samples=10,
        )
        print(f"Dataset prepared: {len(dataset)} samples")
    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")
        return

    # 3. Load model (would be done for actual training)
    if False:  # Skip model loading for example
        print("Loading model...")
        model_id = "Qwen/Qwen2-0.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )

        # Configure LoRA for efficient fine-tuning
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)

    # 4. Configure GRPO training
    training_args = GRPOConfig(
        output_dir="./trl-output",
        learning_rate=1e-5,
        remove_unused_columns=False,  # Keep solution column for reward function
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        max_completion_length=64,
        num_generations=4,
        max_prompt_length=128,
        report_to=["tensorboard"],
        logging_steps=10,
        push_to_hub=False,
        save_strategy="steps",
        save_steps=10,
    )

    # 5. Combine reward functions for TRL
    print("Creating combined reward function...")
    combined_reward = combine_rewards(
        [format_reward_fn, accuracy_reward_fn],
        weights=[0.3, 0.7],  # Format is 30%, accuracy is 70%
    )

    # 6. Create and run trainer (would be done for actual training)
    if False:  # Skip actual training
        print("Creating GRPO trainer...")
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[combined_reward],
            args=training_args,
            train_dataset=dataset,
        )

        print("Starting training...")
        trainer.train()

        print("Training complete!")

    print(
        "\nExample completed successfully. In a real scenario, the training would now run."
    )
    print(
        "This example shows how reward-kit reward functions can be adapted for TRL's GRPO trainer."
    )

    # Print dataset sample to show the format
    print("\nDataset format example:")
    print(dataset[0])

    # Show how reward functions would be called
    print("\nReward function test on sample data:")
    sample_messages = [
        {"role": "system", "content": "Solve the following math problem."},
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": "<think>To solve 2+2, I need to add the numbers. 2+2=4</think><answer>4</answer>",
        },
    ]

    format_result = format_reward(sample_messages)
    accuracy_result = math_accuracy_reward(sample_messages, solution="4")

    print(
        f"Format reward score: {format_result.score} - {format_result.reason}"
    )
    print(
        f"Accuracy reward score: {accuracy_result.score} - {accuracy_result.reason}"
    )

    # Show combined reward calculation
    combined_score = 0.3 * format_result.score + 0.7 * accuracy_result.score
    print(f"Combined reward score: {combined_score}")


if __name__ == "__main__":
    train_with_grpo_example()
