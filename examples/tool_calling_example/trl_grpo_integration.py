"""
Example of TRL GRPO integration for function calling.
This script is a scaffold and requires further implementation for model loading,
dataset processing, and the TRL training loop.
"""

import json
import logging  # Added
import os
import sys
from typing import Any, Dict, List

import hydra  # Added
from omegaconf import DictConfig, OmegaConf  # Added

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# TRL and Transformers imports
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import (  # Removed DataCollatorForCompletionOnlyLM for now
        GRPOConfig,
        GRPOTrainer,
    )
except ImportError:
    # This print will be replaced by logger if Hydra initializes logging first
    print(
        "Please install transformers and TRL dependencies: pip install 'reward-kit[trl]' transformers bitsandbytes"
    )
    sys.exit(1)

from reward_kit.rewards.function_calling import composite_function_call_reward

logger = logging.getLogger(__name__)  # Added


# --- Helper Functions ---
def load_raw_dataset(
    file_path: str, max_samples: int = -1
) -> List[Dict[str, Any]]:  # Added max_samples
    """Loads the raw dataset from a .jsonl file."""
    dataset = []
    logger.info(f"Loading raw dataset from: {file_path}")
    try:
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                if max_samples > 0 and i >= max_samples:
                    logger.info(f"Loaded {max_samples} samples, stopping as per limit.")
                    break
                dataset.append(json.loads(line))
        logger.info(f"Successfully loaded {len(dataset)} raw samples.")
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {file_path}")
    except Exception as e:
        logger.error(f"Error loading dataset from {file_path}: {e}", exc_info=True)
    return dataset


def format_prompt_and_extract_ground_truth(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Formats a dataset item into a prompt for the model and extracts ground truth.
    This needs careful implementation based on how the chosen model expects input for function calling.
    """
    # Basic Linearization (placeholder, needs model-specific improvement)
    tools_str = "\n".join(
        [
            f"Tool: {t['function']['name']}: {t['function']['description']}"
            for t in item.get("tools", [])
        ]
    )
    messages_history = item.get("messages", [])
    # Filter out assistant messages that are only ground truth for previous turns
    # The prompt should end with the user's last message, or system messages setting up the scenario.
    prompt_messages = [m for m in messages_history if m.get("role") != "assistant"]

    messages_str = "\n".join(
        [
            f"{m['role']}: {m.get('content', '')}"
            + (
                f" (called: {m['tool_calls'][0]['function']['name']})"
                if m.get("tool_calls")
                and m.get("role") != "tool"  # Avoid printing tool results as calls
                else ""
            )
            for m in prompt_messages
        ]
    )

    prompt = f"Available Tools:\n{tools_str}\n\nConversation History:\n{messages_str}\n\nAssistant (generate tool calls or a direct answer):"

    return {
        "prompt": prompt,
        "ground_truth_for_reward": item.get("ground_truth"),
        "original_messages_for_reward": messages_history,  # Pass all original messages
    }


# --- Main Script ---
@hydra.main(config_path="conf", config_name="trl_grpo_config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info(f"Hydra configuration (Tool Calling TRL):\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(
        f"Hydra output directory: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )

    MODEL_NAME = cfg.model_name
    DATASET_PATH = hydra.utils.to_absolute_path(cfg.dataset_file_path)
    MAX_SAMPLES_TRAIN = cfg.get("max_samples_train", 100)  # Get from cfg or default
    MAX_SAMPLES_EVAL = cfg.get("max_samples_eval", 20)  # Get from cfg or default

    max_steps_config = 1 if cfg.test_mode_trl else cfg.grpo.max_steps

    # GRPO Configuration from Hydra
    grpo_config = GRPOConfig(
        output_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        beta=cfg.grpo.beta,
        per_device_train_batch_size=cfg.grpo.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.grpo.gradient_accumulation_steps,
        learning_rate=cfg.grpo.learning_rate,
        num_train_epochs=cfg.grpo.num_train_epochs,
        logging_steps=cfg.grpo.logging_steps,
        max_completion_length=cfg.grpo.max_completion_length,
        max_prompt_length=cfg.grpo.get("max_prompt_length", 512),
        remove_unused_columns=False,
        no_cuda=cfg.test_mode_trl,
        top_k=cfg.grpo.top_k,
        top_p=cfg.grpo.top_p,
        do_sample=cfg.grpo.do_sample,
        max_steps=max_steps_config,
    )
    logger.info(f"GRPOConfig prepared (Tool Calling): {grpo_config}")

    import torch  # For MockModel
    from datasets import Dataset  # For converting list to Dataset object

    # 1. Load Model and Tokenizer (Placeholder - Requires Implementation)
    logger.warning(
        f"Attempting to load tokenizer for model: {MODEL_NAME} (Tool Calling)"
    )
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) # Needs to be implemented
    # if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    logger.warning(f"Attempting to load base model: {MODEL_NAME} (Tool Calling)")
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, ...) # Needs to be implemented

    # For this scaffold, we'll simulate a tokenizer and model to allow GRPOTrainer to instantiate
    # This is NOT a functional training setup without real model/tokenizer.
    class MockTokenizer:
        def __init__(self, model_name):
            self.model_name = model_name
            self.pad_token_id = 0
            self.eos_token_id = 1  # Added eos_token_id

        def __call__(self, text, **kwargs):
            return {
                "input_ids": [[0, 1, 2]],
                "attention_mask": [[1, 1, 1]],
            }  # Dummy tokenization

        def save_pretrained(self, path):
            logger.info(f"MockTokenizer saved to {path}")

    class MockModel(torch.nn.Module):
        def __init__(self, model_name):
            super().__init__()
            self.model_name = model_name
            # GRPOTrainer checks for model.config.is_encoder_decoder and model.config.model_type
            self.config = type(
                "config",
                (),
                {
                    "is_encoder_decoder": False,
                    "model_type": "mock",
                    "pad_token_id": 0,
                    "eos_token_id": 1,
                },
            )()  # Added pad/eos

        def forward(
            self, input_ids, attention_mask=None, labels=None, **kwargs
        ):  # Added attention_mask, labels
            # Dummy logits: (batch_size, sequence_length, vocab_size)
            return (torch.randn(input_ids.shape[0], input_ids.shape[1], 10),)

        def save_pretrained(self, path):
            logger.info(f"MockModel saved to {path}")

        # GRPOTrainer needs generate method
        def generate(
            self, input_ids, attention_mask=None, **kwargs
        ):  # Added attention_mask
            # Dummy generation: just return input_ids extended by one dummy token
            dummy_completion = torch.full(
                (input_ids.shape[0], 1),
                3,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            return torch.cat([input_ids, dummy_completion], dim=1)

        def get_input_embeddings(self):
            return torch.nn.Embedding(10, 10)

        def prepare_inputs_for_generation(
            self, input_ids, **kwargs
        ):  # Needed by generate
            return {"input_ids": input_ids}

    if cfg.get("use_mock_model_tokenizer", True):
        logger.warning(
            "Using MOCK Tokenizer and MOCK Model for Tool Calling TRL example. THIS WILL NOT TRAIN A REAL MODEL."
        )
        tokenizer = MockTokenizer(MODEL_NAME)
        model = MockModel(MODEL_NAME)
    else:
        logger.error(
            "Real model/tokenizer loading is not implemented in this scaffold. Set use_mock_model_tokenizer=true to run."
        )
        return

    # 2. Load and Prepare Dataset
    logger.info(f"Loading and preparing dataset from: {DATASET_PATH} (Tool Calling)")
    raw_dataset = load_raw_dataset(
        DATASET_PATH, max_samples=(MAX_SAMPLES_TRAIN + MAX_SAMPLES_EVAL)
    )

    processed_data_for_trl = [
        format_prompt_and_extract_ground_truth(item) for item in raw_dataset
    ]

    valid_processed_data = [
        d for d in processed_data_for_trl if d.get("prompt") is not None
    ]
    if not valid_processed_data:
        logger.error(
            "No valid data after processing for TRL. Check format_prompt_and_extract_ground_truth."
        )
        return

    full_dataset_hf = Dataset.from_list(valid_processed_data)

    if len(full_dataset_hf) < MAX_SAMPLES_TRAIN + MAX_SAMPLES_EVAL:
        logger.warning(
            f"Total samples ({len(full_dataset_hf)}) is less than requested train+eval ({MAX_SAMPLES_TRAIN + MAX_SAMPLES_EVAL}). Adjusting split."
        )
        train_dataset_hf = full_dataset_hf.select(
            range(min(MAX_SAMPLES_TRAIN, len(full_dataset_hf)))
        )
        if len(full_dataset_hf) > MAX_SAMPLES_TRAIN:
            eval_dataset_hf = full_dataset_hf.select(
                range(MAX_SAMPLES_TRAIN, len(full_dataset_hf))
            )
        else:
            eval_dataset_hf = Dataset.from_list([])
    else:
        train_dataset_hf = full_dataset_hf.select(range(MAX_SAMPLES_TRAIN))
        eval_dataset_hf = full_dataset_hf.select(
            range(MAX_SAMPLES_TRAIN, MAX_SAMPLES_TRAIN + MAX_SAMPLES_EVAL)
        )

    logger.info(
        f"Prepared {len(train_dataset_hf)} training samples and {len(eval_dataset_hf)} eval samples (Tool Calling)."
    )

    if len(train_dataset_hf) == 0:  # Check length, not object itself
        logger.error("No training data after split. Exiting.")
        return

    def tokenize_prompts(examples):
        # Ensure max_prompt_length is used from grpo_config
        return tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=grpo_config.max_prompt_length,
            padding=(
                "max_length" if cfg.get("pad_prompts_to_max_length", False) else True
            ),
        )

    train_dataset_tokenized = train_dataset_hf.map(tokenize_prompts, batched=True)
    # Keep necessary columns for reward function metadata and trainer
    columns_to_keep = [
        "input_ids",
        "attention_mask",
        "prompt",
        "ground_truth_for_reward",
        "original_messages_for_reward",
    ]
    train_dataset_tokenized = train_dataset_tokenized.remove_columns(
        [
            col
            for col in train_dataset_tokenized.column_names
            if col not in columns_to_keep
        ]
    )
    train_dataset_tokenized.set_format(
        type="torch", columns=columns_to_keep
    )  # Specify columns for set_format

    eval_dataset_tokenized = None
    if len(eval_dataset_hf) > 0:  # Check length
        eval_dataset_tokenized = eval_dataset_hf.map(tokenize_prompts, batched=True)
        eval_dataset_tokenized = eval_dataset_tokenized.remove_columns(
            [
                col
                for col in eval_dataset_tokenized.column_names
                if col not in columns_to_keep
            ]
        )
        eval_dataset_tokenized.set_format(type="torch", columns=columns_to_keep)

    # 3. Define Reward Function for GRPOTrainer
    def grpo_reward_fn(
        prompts: List[str], responses: List[str], metadata: List[Dict[str, Any]]
    ) -> List[float]:
        rewards = []
        for i in range(len(prompts)):
            generated_assistant_message = {"role": "assistant", "content": responses[i]}
            original_all_messages = metadata[i].get("original_messages_for_reward", [])
            context_messages = [
                m for m in original_all_messages if m.get("role") != "assistant"
            ]
            messages_for_eval = context_messages + [generated_assistant_message]
            ground_truth = metadata[i].get("ground_truth_for_reward")

            if ground_truth is None:
                logger.warning(
                    f"Missing ground_truth_for_reward for prompt: {prompts[i][:100]}... Assigning 0 reward."
                )
                rewards.append(0.0)
                continue
            try:
                eval_result = composite_function_call_reward(
                    messages=messages_for_eval, ground_truth=ground_truth
                )
                rewards.append(eval_result.score)
            except Exception as e:
                logger.error(
                    f"Error in reward function for prompt '{prompts[i][:100]}...': {e}",
                    exc_info=True,
                )
                rewards.append(0.0)
        return rewards

    # 4. Initialize GRPOTrainer
    logger.info("Initializing GRPOTrainer (Tool Calling)...")

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=grpo_config,
        reward_fn=grpo_reward_fn,
        train_dataset=train_dataset_tokenized,
        eval_dataset=eval_dataset_tokenized,  # Pass None if empty
    )
    logger.info("GRPOTrainer initialized.")

    # 5. Train
    if len(train_dataset_tokenized) > 0:  # Check length
        logger.info("Starting TRL GRPO training (Tool Calling)...")
        try:
            trainer.train()
            logger.info("TRL GRPO training completed.")

            final_save_path = os.path.join(
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
                "final_model",
            )
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(final_save_path)  # MockModel has it
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(final_save_path)
            logger.info(
                f"Final model and tokenizer potentially saved to {final_save_path}"
            )

        except Exception as e:
            logger.error(f"Error during GRPOTrainer training: {e}", exc_info=True)
            # If using mock model, some errors might be expected if TRL tries deep interactions
            logger.warning(
                "If using mock model, some errors during trainer.train() might be due to mock limitations."
            )
    else:
        logger.warning("No training data available, skipping GRPOTrainer.train().")

    logger.info("Tool Calling TRL GRPO integration script finished.")
    if cfg.get("use_mock_model_tokenizer", True):
        logger.warning(
            "This script used a MOCK model and tokenizer. Real training requires implementing model/tokenizer loading and disabling mock mode."
        )


if __name__ == "__main__":
    try:
        import accelerate
        import bitsandbytes
        import datasets
        import peft
        import torch
        import transformers
        import trl
    except ImportError as e:
        print(
            f"Import error: {e}. Some libraries missing for Tool Calling TRL example."
        )
        print("Install: pip install 'reward-kit[trl]' transformers bitsandbytes")
        sys.exit(1)
    main()
