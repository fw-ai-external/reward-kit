"""
Minimal example demonstrating the DeepCoder-style reward function
with TRL's GRPO trainer.
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG) # Changed to DEBUG to see more logs
logger = logging.getLogger(__name__)

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset # To convert our list of dicts to HuggingFace Dataset
    from peft import LoraConfig, get_peft_model
    HAS_TRL_AND_TRANSFORMERS = True
except ImportError as e:
    print(f"TRL/Transformers/PEFT/Datasets not installed. Install with: pip install trl transformers torch peft datasets accelerate. Error: {e}")
    HAS_TRL_AND_TRANSFORMERS = False

# Import reward-kit components
from reward_kit.reward_function import RewardFunction # Not strictly needed if using deepcoder_code_reward directly
from reward_kit.rewards import deepcoder_code_reward
from reward_kit.models import Message # For constructing messages for the reward function

# Import data processing utility
from data_utils import process_deepcoder_sample

# Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B" # Small Qwen model for example
DATASET_PATH = Path(__file__).parent / "data/simulated_deepcoder_raw_sample.jsonl"
LANGUAGE = "python"
ENVIRONMENT = "local" # "e2b" if configured
TIMEOUT = 10 # seconds for code execution

def load_and_prepare_dataset(raw_data_path: Path) -> Optional[Dataset]:
    """Loads and prepares the DeepCoder-style dataset into HuggingFace Dataset format."""
    if not raw_data_path.exists():
        logger.error(f"Dataset file not found at {raw_data_path}")
        return None

    processed_data_list = process_deepcoder_sample(str(raw_data_path))
    if not processed_data_list:
        logger.error("No data processed from raw file.")
        return None
    
    # Convert list of dicts to HuggingFace Dataset
    # Ensure all required columns are present. GRPOTrainer needs 'prompt'.
    # Our reward function will need 'test_cases'.
    try:
        hf_dataset = Dataset.from_list(processed_data_list)
        logger.info(f"Dataset loaded and prepared: {len(hf_dataset)} samples. Columns: {hf_dataset.column_names}")
        if "prompt" not in hf_dataset.column_names:
            logger.error("Dataset must contain a 'prompt' column for GRPOTrainer.")
            return None
        if "test_cases" not in hf_dataset.column_names:
            logger.error("Dataset must contain a 'test_cases' column for the reward function.")
            return None
        # Also check for the new target_function column
        if "target_function" not in hf_dataset.column_names:
             logger.error("Dataset must contain a 'target_function' column for the reward function.")
             return None
        return hf_dataset
    except Exception as e:
        logger.error(f"Error converting data to HuggingFace Dataset: {e}")
        return None


def deepcoder_grpo_reward_adapter(
    prompts: List[str],
    completions: List[str],
    # original_data: List[Dict[str, Any]], # Removed from explicit params
    **kwargs # original_data should be in here
) -> List[float]:
    """
    Adapter function to make deepcoder_code_reward compatible with GRPOTrainer.
    GRPOTrainer expects a function that takes:
    - prompts (List[str]): list of prompts used for generation
    - completions (List[str]): list of generated texts
    - original_data (List[Dict[str, Any]]): list of original dataset items (passed via kwargs)
    and returns a list of reward scores (float).
    """
    logger.debug(f"deepcoder_grpo_reward_adapter called. Prompts: {len(prompts)}, Completions: {len(completions)}. kwargs keys: {list(kwargs.keys())}")

    # GRPOTrainer passes other columns from the dataset directly in kwargs.
    # Our dataset now has 'prompt', 'test_cases', and 'target_function'.
    batch_test_cases = kwargs.get("test_cases")
    batch_target_functions = kwargs.get("target_function") # Extract target functions

    if batch_test_cases is None or batch_target_functions is None:
        logger.error(f"'test_cases' ({batch_test_cases is not None}) or 'target_function' ({batch_target_functions is not None}) not found in reward function kwargs. Returning 0.0 for all.")
        logger.error(f"Full kwargs received: {kwargs}") # Log for debugging
        return [0.0] * len(completions)

    rewards = []
    num_samples = len(prompts)

    # Basic check for consistent lengths
    if len(completions) != num_samples or len(batch_test_cases) != num_samples or len(batch_target_functions) != num_samples:
        logger.warning(
            f"Mismatch in lengths of prompts ({len(prompts)}), "
            f"completions ({len(completions)}), batch_test_cases ({len(batch_test_cases)}), "
            f"and batch_target_functions ({len(batch_target_functions)}). Using min length."
        )
        num_samples = min(len(prompts), len(completions), len(batch_test_cases), len(batch_target_functions))
    
    for i in range(num_samples):
        prompt_text = prompts[i]
        completion_text = completions[i]
        test_cases = batch_test_cases[i] # Get test_cases for the current sample
        target_function = batch_target_functions[i] # Get target_function for the current sample

        if test_cases is None: # Target function can be None, handled by deepcoder_code_reward
            logger.warning(f"Sample {i} missing 'test_cases'. Assigning 0 reward.")
            rewards.append(0.0)
            continue

        messages_for_reward = [
            Message(role="user", content=prompt_text),
            Message(role="assistant", content=completion_text)
        ]
        
        try:
            reward_output = deepcoder_code_reward( # Call the core reward function directly
                messages=messages_for_reward,
                language=LANGUAGE,
                test_cases=test_cases,
                environment=ENVIRONMENT,
                timeout=TIMEOUT,
                target_function=target_function # Pass the target function name
            )
            # The reward_output from a function decorated with @reward_function (from typed_interface)
            # is a dictionary (due to model_dump()). Access score via key.
            if isinstance(reward_output, dict) and "score" in reward_output:
                rewards.append(reward_output['score'])
            elif hasattr(reward_output, 'score'): # Fallback if it's an object (e.g. direct EvaluateResult)
                rewards.append(reward_output.score)
            else:
                logger.error(f"Sample {i} - Reward output is not a dict with 'score' or an object with .score attribute: {type(reward_output)}")
                rewards.append(0.0) # Default on unexpected reward format

            # Log more details from reward_output for debugging
            logger.debug(f"Sample {i} - Prompt: {prompt_text[:50]}... Completion: {completion_text[:50]}... RewardOutput: {reward_output}")
        except Exception as e:
            logger.error(f"Error calculating reward for completion {i}: {e}", exc_info=True)
            rewards.append(0.0) # Assign 0 score on error
            
    # Log some stats about rewards if needed
    if rewards:
        # Use INFO level for this summary as it's key to seeing if rewards are non-zero
        logger.info(f"Batch rewards calculated. Count: {len(rewards)}, Min: {min(rewards)}, Max: {max(rewards)}, Avg: {sum(rewards)/len(rewards):.2f}")
    return rewards


def generate_for_comparison(model, tokenizer, prompt_text: str, device) -> str:
    """Helper to generate a response from the model for comparison."""
    # Use a more general system prompt that encourages following user instructions.
    # The user prompt (prompt_text) will contain specific formatting instructions from data_utils.py.
    system_prompt = (
        "You are a helpful assistant that writes Python code. "
        "Follow the user's instructions carefully to produce the required code output. "
        "Be concise and generate only the requested code. Avoid any conversational fluff or explanations outside the code block."
    )
    messages_for_generation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text} # The user prompt contains specific instructions
    ]
    prompt_for_model = tokenizer.apply_chat_template(
        messages_for_generation,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt_for_model, return_tensors="pt").to(device)
    generation_kwargs = {
        "max_new_tokens": 4000, # Increased from 250
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": True,
        "top_k": 10,
        "top_p": 0.95,
        "temperature": 0.7,
    }
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        response_text = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error during comparison generation: {e}")
        response_text = f"Error generating: {e}"
    return response_text


def main():
    if not HAS_TRL_AND_TRANSFORMERS:
        return

    logger.info("Starting Minimal DeepCoder GRPO Example...")

    # 1. Initialize Model and Tokenizer
    logger.info(f"Loading model and tokenizer: {MODEL_NAME}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16) if torch.cuda.is_available() else torch.float32,
            # device_map="auto" # Usually good, but can be problematic with small models / CPU
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.chat_template is None:
            # A basic chat template for Qwen2-Instruct if not set
            tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info(f"Using device: {device}")

        # Configure LoRA for efficient fine-tuning
        logger.info("Configuring LoRA...")
        # Adjust target_modules based on the model being used. For Qwen2-0.5B:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16, # Often 2*r
            lora_dropout=0.05, # Reduced dropout
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Common for Qwen2
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    except Exception as e:
        logger.error(f"Error loading model/tokenizer or configuring LoRA: {e}", exc_info=True)
        return

    # 2. Load and Prepare Dataset
    logger.info(f"Loading dataset from: {DATASET_PATH}")
    train_dataset = load_and_prepare_dataset(DATASET_PATH)
    if train_dataset is None:
        return
    
    # For GRPO, the dataset should be a HuggingFace Dataset object
    # The load_and_prepare_dataset function now returns this.

    # 3. Configure GRPO Training
    logger.info("Configuring GRPO training...")
    # Reduce batch size and steps for a quick test
    training_args = GRPOConfig(
        output_dir="./grpo_deepcoder_output",
        per_device_train_batch_size=2, # Adjusted to be divisible by num_generations
        gradient_accumulation_steps=1, # Keep small
        learning_rate=1e-5, # GRPO often uses smaller LRs
        num_train_epochs=1, # Minimal epochs for testing
        max_steps=5, # Run very few steps for a quick test
        remove_unused_columns=False, # We need 'test_cases' and 'target_function' for the reward
        logging_steps=1,
        report_to="none", # No wandb/tensorboard for this minimal example
        max_prompt_length=4000, # Max length of prompt
        max_completion_length=4000, # Max length of completion
        num_generations=2, # Number of completions to generate per prompt
        beta=0.1, # GRPO specific: KL divergence weight
        # bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        # fp16=torch.cuda.is_available() and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        # Using fp32 for wider compatibility in this example, can enable bf16/fp16 if desired
    )

    # Select a sample prompt for before/after comparison
    sample_prompt_for_comparison = train_dataset[0]["prompt"] if len(train_dataset) > 0 else "Write a Python function to add two numbers."

    # Generate before training
    logger.info("\n--- Generating with model BEFORE training ---")
    pre_train_response = generate_for_comparison(model, tokenizer, sample_prompt_for_comparison, device)
    logger.info(f"Prompt: {sample_prompt_for_comparison[:100]}...")
    logger.info(f"Response (before): {pre_train_response[:200]}...")


    # 4. Create and run GRPOTrainer
    try:
        logger.info("Initializing GRPOTrainer...")
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            # tokenizer=tokenizer, # Removed: GRPOTrainer likely infers tokenizer from model or args
            train_dataset=train_dataset,
            reward_funcs=[deepcoder_grpo_reward_adapter], # Pass the adapter
            # peft_config=lora_config, # Already applied with get_peft_model
        )

        logger.info("Starting GRPO training...")
        trainer.train()
        logger.info("GRPO training completed.")

    except Exception as e:
        logger.error(f"Error during GRPOTrainer initialization or training: {e}", exc_info=True)
        return

    # Generate after training
    logger.info("\n--- Generating with model AFTER training ---")
    # If using PEFT, ensure model is in eval mode or merged for inference if needed
    # model.eval() # Good practice, though generate might handle it
    post_train_response = generate_for_comparison(model, tokenizer, sample_prompt_for_comparison, device)
    logger.info(f"Prompt: {sample_prompt_for_comparison[:100]}...")
    logger.info(f"Response (after): {post_train_response[:200]}...")
    
    logger.info("\nMinimal DeepCoder GRPO Example finished.")

if __name__ == "__main__":
    if HAS_TRL_AND_TRANSFORMERS:
        main()
    else:
        print("TRL/Transformers/PEFT/Datasets not found. Please install them to run this example: pip install trl transformers torch peft datasets accelerate")
