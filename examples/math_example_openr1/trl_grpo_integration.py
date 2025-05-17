import os
import sys
import json
import torch
from typing import List, Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
)
from trl import (
    GRPOConfig,
    GRPOTrainer,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from reward_kit.rewards.math import math_reward
from reward_kit.models import Message

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"  # Using the same model
OUTPUT_DIR = "math_grpo_trainer_output_openr1_qwen"  # Specific output dir for OpenR1
DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "dataset.jsonl"
)  # Will point to openr1 dataset

# GRPO Configuration
grpo_config = GRPOConfig(
    learning_rate=1.41e-5,
    beta=0.1,
    num_train_epochs=1,
    max_steps=(1 if os.environ.get("TEST_MODE_TRL") == "true" else 5),
    logging_steps=1,
    output_dir=OUTPUT_DIR,
    remove_unused_columns=False,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    no_cuda=(True if os.environ.get("TEST_MODE_TRL") == "true" else False),
    # Ensure other necessary GRPOConfig fields are defaulted or set if needed
    # For example, max_completion_length for generation during training
    max_completion_length=50,  # Explicitly set, was in generation_kwargs before
    top_k=0.0,  # from generation_kwargs
    top_p=1.0,  # from generation_kwargs
    do_sample=True,  # from generation_kwargs
)


# --- Helper Functions ---
def load_jsonl_dataset(file_path: str):
    """Loads data from a JSONL file, expecting 'messages' field."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            user_msg = next(
                (m["content"] for m in item["messages"] if m["role"] == "user"), None
            )
            # For GRPO, the 'response' field in the dataset is the ground_truth_answer_from_column
            # which is what math_reward's ground_truth parameter expects.
            # The convert_dataset.py script formats ground_truth_answer_from_column as a boxed answer.
            assistant_msg_as_ground_truth = item.get("ground_truth_answer_from_column")

            if (
                user_msg and assistant_msg_as_ground_truth is not None
            ):  # Ensure ground_truth is present
                data.append(
                    {
                        "prompt": user_msg,
                        "response": assistant_msg_as_ground_truth,  # This will be used as ground_truth by adapted_math_reward
                    }
                )
            elif (
                user_msg
            ):  # Handle cases where ground_truth might be missing, though ideally it shouldn't be for math
                print(
                    f"Warning: Missing 'ground_truth_answer_from_column' for prompt: {user_msg[:50]}..."
                )
                data.append(
                    {
                        "prompt": user_msg,
                        "response": "",  # Provide empty string if ground_truth is missing
                    }
                )

    return data


# --- Reward Function for TRL ---
def adapted_math_reward(
    prompts: List[str], completions: List[str], **kwargs
) -> List[torch.Tensor]:
    rewards = []
    ground_truth_responses = kwargs.get(
        "response", []
    )  # This comes from the 'response' column of the dataset

    if len(ground_truth_responses) != len(prompts):
        print(
            f"Warning: Length mismatch between ground_truth_responses ({len(ground_truth_responses)}) and prompts ({len(prompts)}). Rewards may be incorrect."
        )

    for i in range(len(completions)):
        user_query_str = prompts[i]
        generated_completion_str = completions[i]

        ground_truth_answer_str = ""
        if i < len(ground_truth_responses):
            ground_truth_answer_str = ground_truth_responses[i]
        else:
            print(
                f"Warning: No ground truth response found for prompt index {i}: {user_query_str}"
            )

        if not ground_truth_answer_str:
            print(
                f"Warning: Empty ground_truth_answer_str for prompt: {user_query_str}. Assigning 0 reward."
            )
            rewards.append(torch.tensor(0.0, dtype=torch.float32))
            continue

        messages_for_eval = [
            Message(role="user", content=user_query_str),
            Message(role="assistant", content=generated_completion_str),
        ]

        try:
            eval_result = math_reward(
                messages=messages_for_eval,
                original_messages=messages_for_eval,  # For math_reward, this can be same as messages
                ground_truth=ground_truth_answer_str,  # This is the crucial ground truth
            )
            rewards.append(torch.tensor(eval_result.score, dtype=torch.float32))
        except Exception as e:
            print(f"Error in math_reward during TRL for prompt '{user_query_str}': {e}")
            rewards.append(torch.tensor(0.0, dtype=torch.float32))
    return rewards


# --- Main Script ---
def main():
    # 1. Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_implementation_config = None
    if os.environ.get("TEST_MODE_TRL") == "true":
        print(
            "TRL Test Mode (OpenR1): Forcing CPU, default dtype, and eager attention for model loading."
        )  # Modified
        device_map_config = "cpu"
        torch_dtype_config = None
        attn_implementation_config = "eager"
    else:
        device_map_config = "auto"
        torch_dtype_config = torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype_config,
        device_map=device_map_config,
        attn_implementation=attn_implementation_config,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    policy_model = get_peft_model(base_model, lora_config)
    policy_model.print_trainable_parameters()

    raw_dataset_data = load_jsonl_dataset(DATASET_PATH)
    if not raw_dataset_data:
        print(f"No data loaded from {DATASET_PATH}. Exiting.")
        return

    dataset = Dataset.from_list(raw_dataset_data)

    def preprocess_function(examples):
        return tokenizer(
            examples["prompt"], truncation=True, padding="max_length", max_length=512
        )

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "prompt", "response"],
    )

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    grpo_trainer = GRPOTrainer(
        model=policy_model,
        args=grpo_config,
        train_dataset=tokenized_dataset,
        reward_funcs=[adapted_math_reward],
    )

    print(
        "Starting GRPO training loop for OpenR1 Math Example with PEFT..."
    )  # Modified

    grpo_trainer.train()

    print("\nGRPO training loop completed for OpenR1 Math Example.")  # Modified


if __name__ == "__main__":
    try:
        import torch
        import transformers
        import trl
        import datasets
        import peft
    except ImportError:
        print("Error: PyTorch, Transformers, TRL, Datasets, or PEFT library not found.")
        print("Please install them: pip install torch transformers trl datasets peft")
        sys.exit(1)

    main()
