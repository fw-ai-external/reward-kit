"""
Core evaluation execution pipeline for reward-kit.
This module orchestrates dataset loading, model response generation (optional),
and evaluation using specified reward functions.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import aiohttp  # Added import
import hydra  # For hydra.utils.instantiate if used for dataset loading
from datasets import Dataset, DatasetDict  # Added missing imports
from hydra.errors import InstantiationException  # For specific error handling
from omegaconf import DictConfig, OmegaConf  # For config handling

from reward_kit.auth import get_fireworks_api_key  # For Fireworks client
from reward_kit.datasets.loader import load_and_process_dataset  # Direct import
from reward_kit.generation.cache import ResponseCache
from reward_kit.generation.clients import (  # Assuming Fireworks for now
    FireworksModelClient,
    ModelClient,
)
from reward_kit.models import Message  # For constructing messages for reward function
from reward_kit.utils.module_loader import load_function as load_reward_function

# from ..config_models import PipelineConfig # If using Pydantic models for config structure

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    def __init__(self, pipeline_cfg: DictConfig):
        self.cfg = pipeline_cfg  # Root config for this pipeline run

        # Initialize components based on config
        logger.info("Initializing EvaluationPipeline...")

        # Dataset config is expected under self.cfg.dataset
        # Generation config under self.cfg.generation
        # Reward function config under self.cfg.reward
        # Output config under self.cfg.output

        self.model_client: Optional[ModelClient] = None
        if self.cfg.generation.enabled:
            # Currently hardcoded to FireworksModelClient
            # TODO: Make client type configurable
            api_key = get_fireworks_api_key()
            if not api_key:
                # If generation is enabled, API key is critical.
                # The CLI command should check this earlier or this pipeline should raise.
                logger.error("Fireworks API key not found, but generation is enabled.")
                raise ValueError(
                    "API key required for Fireworks model client when generation is enabled."
                )
            self.model_client = FireworksModelClient(
                client_config=self.cfg.generation,  # Pass the generation sub-config
                api_key=api_key,
            )
            logger.info(
                f"Initialized FireworksModelClient for model: {self.cfg.generation.model_name}"
            )

        self.cache = ResponseCache(self.cfg.generation.cache)  # Pass cache sub-config
        logger.info("ResponseCache initialized.")

        self.reward_function = load_reward_function(self.cfg.reward.function_path)
        logger.info(f"Loaded reward function from: {self.cfg.reward.function_path}")

    async def _process_single_sample(
        self,
        sample: Dict[str, Any],
        http_session: Optional[aiohttp.ClientSession],  # Required if generation is on
        original_index: Optional[int] = None,  # Added original_index
    ) -> Optional[Dict[str, Any]]:
        """
        Processes a single sample: generates response (if needed) and evaluates.
        """
        # Use original_index for a more stable fallback ID if 'id' is not in sample
        sample_id_fallback = (
            f"idx_{original_index}"
            if original_index is not None
            else "unknown_id_" + os.urandom(4).hex()
        )
        sample_id = sample.get("id", sample_id_fallback)
        user_query = sample.get("user_query")  # From standardized prompt dataset
        ground_truth_for_eval = sample.get("ground_truth_for_eval")

        if user_query is None or ground_truth_for_eval is None:
            logger.warning(
                f"Skipping sample {sample_id} due to missing 'user_query' or 'ground_truth_for_eval'."
            )
            return None

        messages_for_generation: List[Dict[str, str]] = []

        # Check for system prompt in sample data first, then fall back to config
        system_prompt = sample.get("system_prompt") or self.cfg.get("system_prompt")
        if system_prompt:
            messages_for_generation.append({"role": "system", "content": system_prompt})
        messages_for_generation.append({"role": "user", "content": user_query})

        assistant_response_content: Optional[str] = None

        if self.cfg.generation.enabled:
            if (
                not self.model_client or not http_session
            ):  # http_session check is important
                logger.error(
                    f"Model client or HTTP session not available for generation for sample {sample_id}."
                )
                # This case should ideally be prevented by checks in run() or __init__
                return {"id": sample_id, "error": "Generation client not configured"}

            assistant_response_content = None
            if self.cfg.generation.cache.enabled:
                # Check cache first
                assistant_response_content = self.cache.get(
                    sample_id=sample_id,
                    system_prompt=self.cfg.get("system_prompt"),
                    user_query=user_query,
                    model_name=self.model_client.model_name,
                    temperature=self.model_client.temperature,
                    top_p=self.model_client.top_p,  # Pass top_p
                    top_k=self.model_client.top_k,  # Pass top_k
                    min_p=self.model_client.min_p,  # Pass min_p
                    max_tokens=self.model_client.max_tokens,  # Pass max_tokens
                    reasoning_effort=self.model_client.reasoning_effort,  # Pass reasoning_effort
                )
            if assistant_response_content:
                logger.info(f"Using cached response for sample {sample_id}")

            if (
                not assistant_response_content
            ):  # If not found in cache OR cache was disabled for GET
                # The "Generating response for sample..." log was moved to the semaphore wrapper for better timing.
                # logger.info(f"Generating response for sample {sample_id}...") # This line is intentionally removed/commented
                try:
                    assistant_response_content = await self.model_client.generate(
                        messages=messages_for_generation,
                        session=http_session,  # Pass the session
                    )
                    if assistant_response_content and self.cfg.generation.cache.enabled:
                        if (
                            self.model_client.temperature == 0.0
                        ):  # Keep existing temp check
                            self.cache.put(
                                sample_id=sample_id,
                                system_prompt=self.cfg.get("system_prompt"),
                                user_query=user_query,
                                model_name=self.model_client.model_name,
                                temperature=self.model_client.temperature,
                                response=assistant_response_content,
                                top_p=self.model_client.top_p,  # Pass top_p
                                top_k=self.model_client.top_k,  # Pass top_k
                                min_p=self.model_client.min_p,  # Pass min_p
                                max_tokens=self.model_client.max_tokens,  # Pass max_tokens
                                reasoning_effort=self.model_client.reasoning_effort,  # Pass reasoning_effort
                            )
                        logger.info(f"Cached new response for sample {sample_id}")
                # except ModelAuthError as e: # Example of specific error handling
                #     logger.error(f"Authentication error for sample {sample_id}: {e}")
                #     raise # Re-raise to be caught by the main gather loop
                except Exception as e:  # Catch other generation errors
                    logger.error(
                        f"Failed to generate response for sample {sample_id}: {e}",
                        exc_info=True,
                    )
                    # Store error information in the result for this sample
                    return {
                        "id": sample_id,
                        "user_query": user_query,
                        "ground_truth_for_eval": ground_truth_for_eval,
                        "error": f"Generation failed: {str(e)}",
                        "evaluation_score": 0.0,
                        "evaluation_reason": "Generation failed",
                    }

            if not assistant_response_content:
                logger.warning(
                    f"No response generated or retrieved from cache for sample {sample_id}."
                )
                return {
                    "id": sample_id,
                    "error": "No response generated/cached",
                    "evaluation_score": 0.0,
                    "evaluation_reason": "No response",
                }
        else:  # Generation disabled
            # Generation disabled.
            # 1. Try to get response from a pre-existing column in the input sample.
            assistant_response_col_name = self.cfg.dataset.get(
                "column_mapping", {}
            ).get("assistant_response_column", "assistant_response")
            assistant_response_content = sample.get(assistant_response_col_name)

            if assistant_response_content:
                logger.info(
                    f"Using pre-existing response from input sample {sample_id} (column: {assistant_response_col_name})"
                )
            else:  # If not in input sample, try to get from cache.
                # Ensure model_name and temperature are available from config for cache key generation.
                # These should be present in cfg.generation even if generation.enabled is false.
                # Ensure model_name and temperature are available from config for cache key generation.
                # These should be present in cfg.generation even if generation.enabled is false.
                gen_cfg = self.cfg.generation
                model_name_for_cache = gen_cfg.get("model_name", "unknown_model")
                temperature_for_cache = gen_cfg.get("temperature", 0.0)
                top_p_for_cache = gen_cfg.get("top_p", 0.95)
                top_k_for_cache = gen_cfg.get("top_k", 20)
                min_p_for_cache = gen_cfg.get("min_p", 0.0)
                max_tokens_for_cache = gen_cfg.get("max_tokens", 1024)
                reasoning_effort_for_cache = gen_cfg.get("reasoning_effort", None)

                logger.debug(
                    f"Attempting cache lookup for sample {sample_id} with generation disabled."
                )
                assistant_response_content = self.cache.get(
                    sample_id=sample_id,
                    system_prompt=self.cfg.get("system_prompt"),
                    user_query=user_query,
                    model_name=model_name_for_cache,
                    temperature=temperature_for_cache,
                    top_p=top_p_for_cache,
                    top_k=top_k_for_cache,
                    min_p=min_p_for_cache,
                    max_tokens=max_tokens_for_cache,
                    reasoning_effort=reasoning_effort_for_cache,
                )
                if assistant_response_content:
                    logger.info(
                        f"Using cached response for sample {sample_id} (generation disabled)."
                    )

            if not assistant_response_content:
                logger.warning(
                    f"Generation disabled. No pre-existing response in input sample (column: {assistant_response_col_name}) and no cache hit for sample {sample_id}."
                )
                return {
                    "id": sample_id,
                    "error": "No pre-existing or cached response",
                    "evaluation_score": 0.0,
                    "evaluation_reason": "No response available (generation disabled)",
                }

        # Construct final messages list for the reward function
        final_messages_for_eval: List[Message] = []
        if self.cfg.get("system_prompt"):
            final_messages_for_eval.append(
                Message(role="system", content=self.cfg.system_prompt)
            )
        final_messages_for_eval.append(Message(role="user", content=user_query))
        final_messages_for_eval.append(
            Message(role="assistant", content=assistant_response_content)
        )

        # Call reward function
        try:
            eval_params = self.cfg.reward.get("params", {})
            eval_result = self.reward_function(
                messages=final_messages_for_eval,
                ground_truth=ground_truth_for_eval,
                **eval_params,
            )
            logger.info(
                f"Sample ID: {sample_id}, Score: {eval_result.score:.2f}, Reason: {eval_result.reason}"
            )
            return {
                "id": sample_id,
                "user_query": user_query,
                "system_prompt": system_prompt,  # Use the resolved system_prompt variable
                "assistant_response": assistant_response_content,
                "ground_truth_for_eval": ground_truth_for_eval,
                "evaluation_score": eval_result.score,
                "evaluation_reason": eval_result.reason,
                "evaluation_metrics": (
                    {k: v.model_dump() for k, v in eval_result.metrics.items()}
                    if eval_result.metrics
                    else {}
                ),
            }
        except Exception as e:
            logger.error(
                f"Error during reward function execution for sample {sample_id}: {e}",
                exc_info=True,
            )
            return {
                "id": sample_id,
                "user_query": user_query,
                "ground_truth_for_eval": ground_truth_for_eval,
                "assistant_response": assistant_response_content,
                "error": f"Reward function failed: {str(e)}",
                "evaluation_score": 0.0,
                "evaluation_reason": "Reward function error",
            }

    async def run(self) -> List[Dict[str, Any]]:
        logger.info("Starting evaluation pipeline run...")

        # 1. Load prompt dataset using Hydra instantiation for dataset config
        try:
            # cfg.dataset is the sub-config for dataset loading (path_or_name, source_type, etc.)
            prompt_dataset_config = self.cfg.dataset
            prompt_dataset = hydra.utils.instantiate(prompt_dataset_config)

            if isinstance(prompt_dataset, DatasetDict):
                split_name = prompt_dataset_config.get("split", "train")
                if split_name in prompt_dataset:
                    prompt_dataset = prompt_dataset[split_name]
                else:
                    logger.error(
                        f"Split '{split_name}' not found. Available: {list(prompt_dataset.keys())}"
                    )
                    return []
            elif not isinstance(prompt_dataset, Dataset):
                logger.error(
                    f"Loaded dataset is not a Hugging Face Dataset. Type: {type(prompt_dataset)}"
                )
                return []
            # Log dataset info with fallback for derived datasets
            dataset_source = getattr(
                prompt_dataset_config,
                "path_or_name",
                getattr(prompt_dataset_config, "base_dataset", "dataset"),
            )
            logger.info(f"Loaded {len(prompt_dataset)} samples from {dataset_source}.")
        except InstantiationException as ie:
            # Check for the specific nested ValueError from fsspec
            final_cause = ie
            while final_cause.__cause__ is not None:
                final_cause = final_cause.__cause__

            if (
                isinstance(final_cause, ValueError)
                and str(final_cause)
                == "Invalid pattern: '**' can only be an entire path component"
            ):
                # prompt_dataset_config is the config for load_derived_dataset.
                # Its 'base_dataset' field holds the *name* of the base dataset config (e.g., "xlam_fc_source").
                base_dataset_config_name = prompt_dataset_config.get(
                    "base_dataset", "UnknownBaseDatasetConfig"
                )

                # The actual Hugging Face path (e.g., "Salesforce/xlam-function-calling-60k")
                # is within the base dataset config file (e.g., conf/dataset/xlam_fc_source.yaml).
                # For the error message, we'll refer to the base_dataset_config_name and guide the user.

                dataset_display_name = base_dataset_config_name  # This is what the user sees in their derived_dataset config.

                helpful_message = (
                    f"Failed to load the base dataset specified as '{dataset_display_name}' in your derived dataset configuration. "
                    f"This occurred due to an internal error in the 'datasets' library (via fsspec): '{str(final_cause)}'.\n"
                    "The error message \"Invalid pattern: '**' can only be an entire path component\" often indicates issues with "
                    "how the 'datasets' library is resolving the path to the dataset, potential Hugging Face Hub connectivity/authentication problems, or a corrupted local cache.\n\n"
                    "Please try the following troubleshooting steps:\n"
                    "1. Verify Hugging Face Hub Token: Ensure your token is correctly configured (e.g., run `huggingface-cli login`). "
                    "An authentication issue can sometimes lead to unexpected path resolution errors.\n"
                    "2. Clear Datasets Cache: The Hugging Face datasets cache (typically at `~/.cache/huggingface/datasets/`) might be corrupted. "
                    f"Try removing the subdirectory related to the actual Hugging Face dataset path/name (you'll find this path inside your '{dataset_display_name}.yaml' config file, usually under a 'path_or_name' key).\n"
                    "3. Update Libraries: Ensure `datasets`, `huggingface_hub`, and `fsspec` are up-to-date: "
                    "`pip install --upgrade datasets huggingface_hub fsspec`.\n"
                    "4. Test Direct Loading: Try loading the dataset directly in a separate Python script. "
                    f"First, find the actual Hugging Face path/name from your '{dataset_display_name}.yaml' configuration file (it's the 'path_or_name' value, e.g., 'Salesforce/xlam-function-calling-60k'). "
                    "Then use it in a script like this:\n"
                    "   ```python\n"
                    "   from datasets import load_dataset\n"
                    '   actual_hf_path = "PASTE_ACTUAL_HF_PATH_HERE"  # e.g., "Salesforce/xlam-function-calling-60k"\n'
                    "   split_to_load = \"{prompt_dataset_config.get('split', 'train')}\" # Or the relevant split from your config\n"
                    "   try:\n"
                    f"       ds = load_dataset(actual_hf_path, split=split_to_load)\n"
                    "       print(f\"Dataset '{{actual_hf_path}}' loaded successfully:\", ds)\n"
                    "   except Exception as e_script:\n"
                    "       print(f\"Error loading dataset '{{actual_hf_path}}' directly: {{e_script}}\")\n"
                    "   ```\n"
                    f"Original InstantiationException details: {ie}"
                )
                logger.error(helpful_message, exc_info=False)  # Log the helpful message
                # Optionally, re-raise a custom, more informative error or just the original one
                # For now, just log and let the original flow return []
            else:
                # Handle other InstantiationExceptions or log normally
                logger.error(f"Failed to load prompt dataset: {ie}", exc_info=True)
            return []
        except Exception as e:  # Catch any other exceptions
            logger.error(
                f"An unexpected error occurred during dataset loading: {e}",
                exc_info=True,
            )
            return []

        all_results: List[Dict[str, Any]] = []

        limit_samples = self.cfg.evaluation_params.get("limit_samples", None)
        samples_to_process_count = len(prompt_dataset)
        if limit_samples and limit_samples > 0:
            samples_to_process_count = min(len(prompt_dataset), limit_samples)

        logger.info(f"Processing {samples_to_process_count} samples.")

        # Concurrency and task management
        # Ensure http_session is created only if generation is enabled
        http_session: Optional[aiohttp.ClientSession] = None
        if self.cfg.generation.enabled and self.model_client:
            http_session = aiohttp.ClientSession()

        try:
            tasks = []
            semaphore = asyncio.Semaphore(
                self.cfg.generation.api_params.get("max_concurrent_requests", 5)
            )

            async def process_with_semaphore_wrapper(
                sample_idx: int, sample_data: Dict[str, Any]
            ):
                # Construct a preliminary sample_id for logging before _process_single_sample does its own.
                prelim_sample_id = sample_data.get("id", f"idx_{sample_idx}")
                async with semaphore:
                    # Log that a semaphore slot has been acquired and processing for this sample is starting.
                    logger.info(
                        f"Concurrency slot acquired for sample '{prelim_sample_id}', attempting to process."
                    )
                    # Pass original_index for more robust ID generation if 'id' is missing in sample_data
                    return await self._process_single_sample(
                        sample_data, http_session, original_index=sample_idx
                    )

            for i in range(samples_to_process_count):
                tasks.append(process_with_semaphore_wrapper(i, prompt_dataset[i]))

            # Process tasks, potentially in batches for logging
            batch_size_for_logging = self.cfg.logging_params.get(
                "batch_log_interval", 10
            )
            for i_outer in range(0, len(tasks), batch_size_for_logging):
                batch_tasks = tasks[i_outer : i_outer + batch_size_for_logging]
                batch_results = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )

                for res_idx, res_or_exc in enumerate(batch_results):
                    # if isinstance(res_or_exc, ModelAuthError): # Example specific error
                    #     logger.critical(f"Critical Model Auth Error: {res_or_exc}. Halting pipeline.")
                    #     # Potentially re-raise or handle to stop all further processing
                    #     if http_session: await http_session.close()
                    #     return all_results # Return what has been processed so far
                    if isinstance(res_or_exc, Exception):
                        logger.error(
                            f"Task for sample index {i_outer + res_idx} failed: {res_or_exc}"
                        )
                        # Optionally add error placeholder to results
                        all_results.append(
                            {
                                "id": prompt_dataset[i_outer + res_idx].get(
                                    "id", "unknown"
                                ),
                                "error": str(res_or_exc),
                            }
                        )
                    elif res_or_exc is not None:
                        all_results.append(res_or_exc)
                logger.info(
                    f"Completed batch up to sample {i_outer + len(batch_tasks)}. Total results/errors: {len(all_results)}"
                )

        finally:
            if http_session:
                await http_session.close()

        # Save results if output_file is specified
        output_file_path = self.cfg.output.get("results_file", None)
        if output_file_path:
            if (
                not os.path.isabs(output_file_path) and self.cfg.hydra_output_dir
            ):  # Assuming hydra_output_dir is passed in cfg
                output_file_path = os.path.join(
                    self.cfg.hydra_output_dir, output_file_path
                )
            try:
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                with open(output_file_path, "w", encoding="utf-8") as f:
                    for result_item in all_results:
                        f.write(json.dumps(result_item) + "\n")
                logger.info(
                    f"Detailed results saved to: {os.path.abspath(output_file_path)}"
                )
            except Exception as e:
                logger.error(f"Failed to save results to {output_file_path}: {e}")

        # Save input/output pairs for preview command
        preview_pairs_file_path = self.cfg.output.get(
            "preview_pairs_file", "preview_input_output_pairs.jsonl"
        )
        if preview_pairs_file_path:
            if not os.path.isabs(preview_pairs_file_path) and self.cfg.hydra_output_dir:
                preview_pairs_file_path = os.path.join(
                    self.cfg.hydra_output_dir, preview_pairs_file_path
                )

            preview_data_to_save = []
            for result_item in all_results:
                if (
                    "error" in result_item
                    or not result_item.get("user_query")
                    or not result_item.get("assistant_response")
                ):
                    # Skip items with errors or missing critical fields for preview
                    continue

                messages = []
                if result_item.get("system_prompt"):
                    messages.append(
                        {"role": "system", "content": result_item["system_prompt"]}
                    )
                messages.append({"role": "user", "content": result_item["user_query"]})
                messages.append(
                    {"role": "assistant", "content": result_item["assistant_response"]}
                )

                pair_item = {"messages": messages}
                if result_item.get("ground_truth_for_eval"):
                    pair_item["ground_truth"] = result_item["ground_truth_for_eval"]

                # Optionally, carry over the sample ID if present
                if result_item.get("id"):
                    pair_item["id"] = result_item["id"]

                preview_data_to_save.append(pair_item)

            if preview_data_to_save:
                try:
                    os.makedirs(os.path.dirname(preview_pairs_file_path), exist_ok=True)
                    with open(preview_pairs_file_path, "w", encoding="utf-8") as f:
                        for item in preview_data_to_save:
                            f.write(json.dumps(item) + "\n")
                    logger.info(
                        f"Input/output pairs for preview saved to: {os.path.abspath(preview_pairs_file_path)}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to save preview pairs to {preview_pairs_file_path}: {e}"
                    )

        logger.info("Evaluation pipeline run finished.")
        return all_results
