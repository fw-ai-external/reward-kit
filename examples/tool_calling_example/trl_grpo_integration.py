"""
Example of TRL GRPO integration for function calling.
This script is a scaffold and requires further implementation for model loading,
dataset processing, and the TRL training loop.
"""

import json
import os
import sys
from typing import Any, Dict, List

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# TRL and Transformers imports (assuming they are installed)
try:
    from transformers import (  # Or sequence-to-sequence if appropriate
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from trl import (  # Or other relevant collator
        DataCollatorForCompletionOnlyLM,
        GRPOConfig,
        GRPOTrainer,
    )
except ImportError:
    print(
        "Please install transformers and trl: pip install transformers trl peft bitsandbytes"
    )
    sys.exit(1)

from reward_kit.rewards.function_calling import composite_function_call_reward

# from reward_kit.models import Message # If needed for strict type handling

# --- Configuration ---
MODEL_NAME = (
    "Salesforce/codegen25-7b-multi"  # Example model, replace with a suitable one
)
OUTPUT_DIR = "./grpo_function_calling_output"
DATASET_PATH = "examples/tool_calling_example/dataset.jsonl"
MAX_SAMPLES_TRAIN = 100  # Limit samples for example run
MAX_SAMPLES_EVAL = 20

# GRPO Configuration (example values, tune as needed)
grpo_config = GRPOConfig(
    model_name=MODEL_NAME,
    reward_adapter_name="reward_adapter",  # Name for the reward adapter
    beta=0.1,  # Weight for the KL divergence term
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=1,
    eval_steps=10,
    logging_steps=1,
    save_steps=50,
    max_length=1024,  # Adjust based on model and data
    max_prompt_length=512,  # Adjust
    remove_unused_columns=False,  # Important for custom datasets
    # reward_baseline=0.5, # Optional: baseline for rewards
)


def load_raw_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Loads the raw dataset from a .jsonl file."""
    dataset = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            # if i >= MAX_SAMPLES_TRAIN + MAX_SAMPLES_EVAL: # Limit total loaded for speed
            #     break
            dataset.append(json.loads(line))
    return dataset


def format_prompt_and_extract_ground_truth(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Formats a dataset item into a prompt for the model and extracts ground truth.
    This needs careful implementation based on how the chosen model expects input for function calling.
    It should linearize the 'tools' definitions and 'messages' history.
    """
    # Example Linearization (very basic, needs significant improvement):
    tools_str = "\n".join(
        [
            f"Tool: {t['function']['name']}: {t['function']['description']}"
            for t in item.get("tools", [])
        ]
    )
    messages_str = "\n".join(
        [
            f"{m['role']}: {m.get('content', '')}"
            + (
                f" (called: {m['tool_calls'][0]['function']['name']})"
                if m.get("tool_calls")
                else ""
            )
            for m in item.get("messages", [])
            if m.get("role") != "assistant"
        ]
    )  # Up to user turn

    # This prompt format is highly dependent on the base model and how it was trained for tool use.
    prompt = f"Available Tools:\n{tools_str}\n\nConversation:\n{messages_str}\nAssistant (should generate tool calls if appropriate):"

    # The 'response' here would be the ground truth assistant message, including tool_calls
    # For GRPO, we often provide the prompt and the trainer handles generation.
    # The ground_truth_for_reward is what our reward function needs.
    return {
        "prompt": prompt,
        "ground_truth_for_reward": item.get(
            "ground_truth"
        ),  # This is the dict for our reward function
        "original_messages_for_reward": item.get(
            "messages"
        ),  # Pass original messages for reward context
    }


def main():
    # 1. Load Model and Tokenizer
    # TODO: Add model loading with PEFT/LoRA if needed, quantization, etc.
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    print(
        f"Note: Model loading and tokenizer setup is currently a placeholder in this script."
    )
    print(f"Please implement model loading for '{MODEL_NAME}' or your chosen model.\n")

    # 2. Load and Prepare Dataset
    raw_dataset = load_raw_dataset(DATASET_PATH)
    # This mapping needs to be carefully designed.
    # TRL GRPOTrainer expects specific column names like 'prompt' and often 'response' or 'label'.
    # We also need to pass necessary info to our reward function.

    # For GRPOTrainer, we typically provide prompts, and the model generates responses.
    # The reward function then scores these (prompt, generated_response) pairs.

    # This is a simplified dataset preparation.
    # In practice, you'd use Hugging Face `datasets.Dataset` and `map`.
    processed_dataset = [
        format_prompt_and_extract_ground_truth(item) for item in raw_dataset
    ]

    # Split into train and eval - simplistic split for example
    train_dataset = processed_dataset[:MAX_SAMPLES_TRAIN]
    eval_dataset = processed_dataset[
        MAX_SAMPLES_TRAIN : MAX_SAMPLES_TRAIN + MAX_SAMPLES_EVAL
    ]

    print(
        f"Loaded and processed {len(train_dataset)} training samples and {len(eval_dataset)} eval samples."
    )
    if not train_dataset:
        print("No training data. Exiting.")
        sys.exit(1)

    # print("\nExample Training Prompt:")
    # print(train_dataset[0]['prompt'])
    # print("\nExample Ground Truth for Reward:")
    # print(json.dumps(train_dataset[0]['ground_truth_for_reward'], indent=2))

    # 3. Define Reward Function for GRPOTrainer
    # The GRPOTrainer will call this function with generated text and other metadata.
    # We need to adapt our `composite_function_call_reward` to fit the expected signature.
    # GRPOTrainer typically passes: List[str] (prompts), List[str] (responses), List[Dict[str,Any]] (metadata from dataset)

    # This is a conceptual wrapper. The actual signature and data flow within GRPOTrainer
    # need to be matched precisely.
    def grpo_reward_fn(
        prompts: List[str], responses: List[str], metadata: List[Dict[str, Any]]
    ) -> List[float]:
        rewards = []
        for i in range(len(prompts)):
            # Reconstruct the 'messages' list for our reward function
            # The 'responses[i]' is the text generated by the LLM.
            # We need to parse it into the assistant message format, potentially extracting tool calls.

            # This is a placeholder for parsing the raw model output string `responses[i]`
            # into the structured assistant message format that our reward function expects.
            # This parsing is CRITICAL and model-dependent.

            # Conceptual parsing:
            generated_assistant_message = {"role": "assistant", "content": responses[i]}
            # If model outputs structured tool_calls (e.g. via special tokens or JSON mode):
            #   tool_calls = parse_model_output_for_tool_calls(responses[i])
            #   generated_assistant_message["tool_calls"] = tool_calls
            # If tool calls are embedded in text (like <tool_call>...</tool_call>):
            #   Our reward function's `eval_tool_call` already handles parsing from 'content'.

            # The 'metadata' should carry the original messages and ground_truth for this item.
            # We assume 'original_messages_for_reward' and 'ground_truth_for_reward' were passed in metadata.
            original_user_messages = metadata[i].get("original_messages_for_reward", [])
            # We need to take messages up to the point of generation
            context_messages = [
                m for m in original_user_messages if m.get("role") != "assistant"
            ]

            messages_for_eval = context_messages + [generated_assistant_message]
            ground_truth = metadata[i].get("ground_truth_for_reward")

            eval_result = composite_function_call_reward(
                messages=messages_for_eval, ground_truth=ground_truth
            )
            rewards.append(eval_result.score)
        return rewards

    print(
        f"\nNote: GRPOTrainer setup and reward function adaptation are conceptual in this script."
    )
    print(
        f"The actual implementation of 'grpo_reward_fn' and data collation needs careful alignment with TRL's GRPOTrainer.\n"
    )

    # 4. Initialize GRPOTrainer
    # TODO: Implement the actual GRPOTrainer initialization
    # This requires a fully set up model, tokenizer, and datasets formatted correctly.
    # Data collator might be needed, e.g., DataCollatorForCompletionOnlyLM.

    # Example (conceptual, will not run without model/tokenizer/proper dataset objects):
    # trainer = GRPOTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     config=grpo_config,
    #     reward_function=grpo_reward_fn,
    #     train_dataset=train_dataset_hf, # This should be a Hugging Face Dataset object
    #     eval_dataset=eval_dataset_hf,   # This should be a Hugging Face Dataset object
    #     # data_collator=... ,
    # )

    # 5. Train
    # TODO: Implement training call
    # trainer.train()

    # 6. Save model (optional)
    # trainer.save_model(os.path.join(OUTPUT_DIR, "final_checkpoint"))
    # tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_checkpoint"))

    print("TRL GRPO integration script scaffold created.")
    print(
        "This script requires significant further implementation for actual training."
    )
    print(
        "Key areas to implement: model loading, tokenizer setup, dataset formatting for TRL,"
    )
    print("and precise GRPOTrainer initialization and reward function wrapping.")


if __name__ == "__main__":
    main()
