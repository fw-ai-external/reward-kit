import logging
import os
import sys
from typing import Any, Dict, List

import hydra  # Added
import torch
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf  # Added
from peft import LoraConfig, get_peft_model
from transformers import (  # Removed DataCollatorWithPadding
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import GRPOConfig, GRPOTrainer

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from reward_kit.common_utils import load_jsonl
from reward_kit.models import Message
from reward_kit.rewards.math import math_reward

logger = logging.getLogger(__name__)  # Added


# --- Helper Functions ---
# This load_jsonl_dataset is specific to openr1 example
def load_jsonl_dataset(file_path: str):
    """Loads data from a JSONL file, expecting 'messages' and 'ground_truth', and processes it for TRL."""
    raw_data = load_jsonl(file_path)
    if not raw_data:
        logger.warning(f"No data loaded from {file_path}. Returning empty list.")
        return []

    processed_trl_data = []
    for item in raw_data:
        messages = item.get("messages")
        if not messages:
            logger.warning(f"Skipping item due to missing 'messages': {item}")
            continue

        user_msg_content = next(
            (m.get("content") for m in messages if m.get("role") == "user"), None
        )

        # For this OpenR1 example, the ground truth is specifically in this field
        ground_truth_response = item.get("ground_truth")

        if user_msg_content and ground_truth_response is not None:
            processed_trl_data.append(
                {
                    "prompt": user_msg_content,
                    "response": ground_truth_response,  # This is the ground truth for reward
                    "messages": messages,
                }
            )
        elif user_msg_content:  # Handle cases where ground_truth might be missing
            logger.warning(
                f"Missing 'ground_truth' for prompt: {user_msg_content[:50]}... using empty response."
            )
            processed_trl_data.append(
                {
                    "prompt": user_msg_content,
                    "response": "",
                    "messages": messages,
                }
            )
        else:  # Missing user_msg_content
            logger.warning(f"Skipping item due to missing user message: {item}")

    return processed_trl_data


# --- Reward Function for TRL ---
# (adapted_math_reward function remains the same as in the previous file, using logger)
def adapted_math_reward(
    prompts: List[str], completions: List[str], **kwargs
) -> List[torch.Tensor]:
    rewards = []
    ground_truth_responses = kwargs.get("response", [])

    if len(ground_truth_responses) != len(prompts):
        logger.warning(
            f"Length mismatch between ground_truth_responses ({len(ground_truth_responses)}) and prompts ({len(prompts)}). Rewards may be incorrect."
        )

    for i in range(len(completions)):
        user_query_str = prompts[i]
        generated_completion_str = completions[i]
        ground_truth_answer_str = ""

        if i < len(ground_truth_responses):
            ground_truth_answer_str = ground_truth_responses[i]
        else:
            logger.warning(
                f"No ground truth response found for prompt index {i}: {user_query_str}"
            )

        if not ground_truth_answer_str:
            logger.warning(
                f"Empty ground_truth_answer_str for prompt: {user_query_str}. Assigning 0 reward."
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
                ground_truth=ground_truth_answer_str,
            )
            rewards.append(torch.tensor(eval_result.score, dtype=torch.float32))
        except Exception as e:
            logger.error(
                f"Error in math_reward for prompt '{user_query_str}': {e}",
                exc_info=True,
            )
            rewards.append(torch.tensor(0.0, dtype=torch.float32))
    return rewards


# --- Main Script ---
@hydra.main(config_path="conf", config_name="trl_grpo_config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info(f"Hydra configuration (OpenR1):\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(
        f"Hydra output directory: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )

    MODEL_NAME = cfg.model_name
    DATASET_PATH = hydra.utils.to_absolute_path(cfg.dataset_file_path)

    max_steps_config = 1 if cfg.test_mode_trl else cfg.grpo.max_steps

    grpo_config = GRPOConfig(
        output_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        learning_rate=cfg.grpo.learning_rate,
        beta=cfg.grpo.beta,
        num_train_epochs=cfg.grpo.num_train_epochs,
        max_steps=max_steps_config,
        logging_steps=cfg.grpo.logging_steps,
        remove_unused_columns=False,
        per_device_train_batch_size=cfg.grpo.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.grpo.gradient_accumulation_steps,
        no_cuda=cfg.test_mode_trl,
        max_completion_length=cfg.grpo.max_completion_length,
        top_k=cfg.grpo.top_k,
        top_p=cfg.grpo.top_p,
        do_sample=cfg.grpo.do_sample,
    )
    logger.info(f"GRPOConfig prepared (OpenR1): {grpo_config}")
    # 1. Load Tokenizer and Model
    logger.info(f"Loading tokenizer for model (OpenR1): {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(
            f"Tokenizer pad_token was None, set to eos_token (OpenR1): {tokenizer.eos_token}"
        )

    logger.info(f"Loading base model (OpenR1): {MODEL_NAME}")
    attn_implementation_config = None
    if cfg.test_mode_trl:
        logger.info(
            "TRL Test Mode (OpenR1): Forcing CPU, default dtype, and eager attention for model loading."
        )
        device_map_config = "cpu"
        torch_dtype_config = None
        attn_implementation_config = "eager"
    else:
        device_map_config = "auto"
        torch_dtype_config = torch.float16
        logger.info(
            f"Using device_map (OpenR1): {device_map_config}, torch_dtype: {torch_dtype_config}, attn_implementation: {attn_implementation_config}"
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype_config,
        device_map=device_map_config,
        attn_implementation=attn_implementation_config,
    )
    logger.info(f"Base model {MODEL_NAME} loaded (OpenR1).")

    lora_config = LoraConfig(  # These should ideally come from cfg.lora
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    logger.info(f"LoRA config (OpenR1): {lora_config}")

    policy_model = get_peft_model(base_model, lora_config)
    trainable_params, all_param = policy_model.get_nb_trainable_parameters()
    logger.info(
        f"Trainable PEFT params (OpenR1): {trainable_params}. All params: {all_param}. Percentage: {100 * trainable_params / all_param:.2f}%"
    )

    logger.info(f"Loading dataset from (OpenR1): {DATASET_PATH}")
    raw_dataset_data = load_jsonl_dataset(
        DATASET_PATH
    )  # Uses the specific load_jsonl_dataset for OpenR1
    if not raw_dataset_data:
        logger.error(f"No data loaded from {DATASET_PATH} (OpenR1). Exiting.")
        return

    dataset = Dataset.from_list(raw_dataset_data)
    logger.info(f"Dataset loaded with {len(dataset)} samples (OpenR1).")

    def preprocess_function(examples):
        return tokenizer(
            examples["prompt"], truncation=True, padding="max_length", max_length=512
        )

    logger.info("Tokenizing dataset (OpenR1)...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "prompt", "response"],
    )
    logger.info("Dataset tokenized and formatted (OpenR1).")

    grpo_trainer = GRPOTrainer(
        model=policy_model,
        args=grpo_config,
        train_dataset=tokenized_dataset,
        reward_funcs=[adapted_math_reward],
    )
    logger.info("GRPOTrainer instantiated (OpenR1).")

    logger.info("Starting GRPO training loop (OpenR1)...")
    grpo_trainer.train()
    logger.info("GRPO training loop completed (OpenR1).")

    tokenizer_save_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    tokenizer.save_pretrained(tokenizer_save_path)
    logger.info(f"Tokenizer saved to {tokenizer_save_path} (OpenR1)")


if __name__ == "__main__":
    try:
        import accelerate
        import datasets
        import peft
        import torch
        import transformers
        import trl
    except ImportError as e:
        print(f"Import error: {e}. Missing libraries for OpenR1 TRL example.")
        print("Install: pip install torch transformers trl datasets peft accelerate")
        sys.exit(1)
    main()
