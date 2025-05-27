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

import aiohttp # Added import
import hydra # For hydra.utils.instantiate if used for dataset loading
from omegaconf import DictConfig, OmegaConf # For config handling
from datasets import Dataset, DatasetDict # Added missing imports

from reward_kit.datasets.loader import load_and_process_dataset # Direct import
from reward_kit.generation.clients import ModelClient, FireworksModelClient # Assuming Fireworks for now
from reward_kit.generation.cache import ResponseCache
from reward_kit.utils.module_loader import load_function as load_reward_function
from reward_kit.models import Message # For constructing messages for reward function
from reward_kit.auth import get_fireworks_api_key # For Fireworks client

# from ..config_models import PipelineConfig # If using Pydantic models for config structure

logger = logging.getLogger(__name__)

class EvaluationPipeline:
    def __init__(self, pipeline_cfg: DictConfig):
        self.cfg = pipeline_cfg # Root config for this pipeline run
        
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
                raise ValueError("API key required for Fireworks model client when generation is enabled.")
            self.model_client = FireworksModelClient(
                client_config=self.cfg.generation, # Pass the generation sub-config
                api_key=api_key
            )
            logger.info(f"Initialized FireworksModelClient for model: {self.cfg.generation.model_name}")

        self.cache = ResponseCache(self.cfg.generation.cache) # Pass cache sub-config
        logger.info("ResponseCache initialized.")

        self.reward_function = load_reward_function(self.cfg.reward.function_path)
        logger.info(f"Loaded reward function from: {self.cfg.reward.function_path}")


    async def _process_single_sample(
        self, 
        sample: Dict[str, Any], 
        http_session: Optional[aiohttp.ClientSession] # Required if generation is on
    ) -> Optional[Dict[str, Any]]:
        """
        Processes a single sample: generates response (if needed) and evaluates.
        """
        sample_id = sample.get("id", "unknown_id_" + os.urandom(4).hex())
        user_query = sample.get("user_query") # From standardized prompt dataset
        ground_truth_for_eval = sample.get("ground_truth_for_eval")

        if user_query is None or ground_truth_for_eval is None:
            logger.warning(f"Skipping sample {sample_id} due to missing 'user_query' or 'ground_truth_for_eval'.")
            return None

        messages_for_generation: List[Dict[str, str]] = []
        if self.cfg.get("system_prompt"):
            messages_for_generation.append({"role": "system", "content": self.cfg.system_prompt})
        messages_for_generation.append({"role": "user", "content": user_query})

        assistant_response_content: Optional[str] = None

        if self.cfg.generation.enabled:
            if not self.model_client or not http_session: # http_session check is important
                logger.error(f"Model client or HTTP session not available for generation for sample {sample_id}.")
                # This case should ideally be prevented by checks in run() or __init__
                return {"id": sample_id, "error": "Generation client not configured"}


            # Check cache first
            assistant_response_content = self.cache.get(
                sample_id=sample_id,
                system_prompt=self.cfg.get("system_prompt"),
                user_query=user_query,
                model_name=self.model_client.model_name, # Get from client instance
                temperature=self.model_client.temperature # Get from client instance
            )
            if assistant_response_content:
                logger.info(f"Using cached response for sample {sample_id}")
            else:
                logger.info(f"Generating response for sample {sample_id}...")
                try:
                    assistant_response_content = await self.model_client.generate(
                        messages=messages_for_generation,
                        session=http_session # Pass the session
                    )
                    if assistant_response_content and self.model_client.temperature == 0.0:
                        self.cache.put(
                            sample_id=sample_id,
                            system_prompt=self.cfg.get("system_prompt"),
                            user_query=user_query,
                            model_name=self.model_client.model_name,
                            temperature=self.model_client.temperature,
                            response=assistant_response_content
                        )
                        logger.info(f"Cached new response for sample {sample_id}")
                # except ModelAuthError as e: # Example of specific error handling
                #     logger.error(f"Authentication error for sample {sample_id}: {e}")
                #     raise # Re-raise to be caught by the main gather loop
                except Exception as e: # Catch other generation errors
                    logger.error(f"Failed to generate response for sample {sample_id}: {e}", exc_info=True)
                    # Store error information in the result for this sample
                    return {
                        "id": sample_id, "user_query": user_query, "ground_truth_for_eval": ground_truth_for_eval,
                        "error": f"Generation failed: {str(e)}", "evaluation_score": 0.0, "evaluation_reason": "Generation failed"
                    }


            if not assistant_response_content:
                logger.warning(f"No response generated or retrieved from cache for sample {sample_id}.")
                return {"id": sample_id, "error": "No response generated/cached", "evaluation_score": 0.0, "evaluation_reason": "No response"}
        else: # Generation disabled
            # Expect pre-generated assistant response in the input sample
            assistant_response_content = sample.get(self.cfg.dataset.get("column_mapping",{}).get("assistant_response_column", "assistant_response"))
            if not assistant_response_content:
                logger.warning(f"Generation disabled and no pre-existing assistant response found for sample {sample_id} (expected column: {self.cfg.dataset.get('column_mapping',{}).get('assistant_response_column', 'assistant_response')}).")
                return {"id": sample_id, "error": "No pre-existing response", "evaluation_score": 0.0, "evaluation_reason": "No response"}

        # Construct final messages list for the reward function
        final_messages_for_eval: List[Message] = []
        if self.cfg.get("system_prompt"):
            final_messages_for_eval.append(Message(role="system", content=self.cfg.system_prompt))
        final_messages_for_eval.append(Message(role="user", content=user_query))
        final_messages_for_eval.append(Message(role="assistant", content=assistant_response_content))
        
        # Call reward function
        try:
            eval_params = self.cfg.reward.get("params", {})
            eval_result = self.reward_function(
                messages=final_messages_for_eval,
                ground_truth=ground_truth_for_eval,
                **eval_params
            )
            logger.info(f"Sample ID: {sample_id}, Score: {eval_result.score:.2f}, Reason: {eval_result.reason}")
            return {
                "id": sample_id, "user_query": user_query, 
                "system_prompt": self.cfg.get("system_prompt"),
                "assistant_response": assistant_response_content, 
                "ground_truth_for_eval": ground_truth_for_eval,
                "evaluation_score": eval_result.score,
                "evaluation_reason": eval_result.reason,
                "evaluation_metrics": {k: v.model_dump() for k, v in eval_result.metrics.items()} if eval_result.metrics else {}
            }
        except Exception as e:
            logger.error(f"Error during reward function execution for sample {sample_id}: {e}", exc_info=True)
            return {
                "id": sample_id, "user_query": user_query, "ground_truth_for_eval": ground_truth_for_eval,
                "assistant_response": assistant_response_content, 
                "error": f"Reward function failed: {str(e)}", "evaluation_score": 0.0, "evaluation_reason": "Reward function error"
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
                    logger.error(f"Split '{split_name}' not found. Available: {list(prompt_dataset.keys())}")
                    return []
            elif not isinstance(prompt_dataset, Dataset):
                 logger.error(f"Loaded dataset is not a Hugging Face Dataset. Type: {type(prompt_dataset)}")
                 return []
            logger.info(f"Loaded {len(prompt_dataset)} samples from {prompt_dataset_config.path_or_name}.")
        except Exception as e:
            logger.error(f"Failed to load prompt dataset: {e}", exc_info=True)
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
            semaphore = asyncio.Semaphore(self.cfg.generation.api_params.get("max_concurrent_requests", 5))

            async def process_with_semaphore_wrapper(sample_data: Dict[str, Any]):
                async with semaphore:
                    return await self._process_single_sample(sample_data, http_session)

            for i in range(samples_to_process_count):
                tasks.append(process_with_semaphore_wrapper(prompt_dataset[i]))
            
            # Process tasks, potentially in batches for logging
            batch_size_for_logging = self.cfg.logging_params.get("batch_log_interval", 10)
            for i_outer in range(0, len(tasks), batch_size_for_logging):
                batch_tasks = tasks[i_outer : i_outer + batch_size_for_logging]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for res_idx, res_or_exc in enumerate(batch_results):
                    # if isinstance(res_or_exc, ModelAuthError): # Example specific error
                    #     logger.critical(f"Critical Model Auth Error: {res_or_exc}. Halting pipeline.")
                    #     # Potentially re-raise or handle to stop all further processing
                    #     if http_session: await http_session.close()
                    #     return all_results # Return what has been processed so far
                    if isinstance(res_or_exc, Exception):
                        logger.error(f"Task for sample index {i_outer + res_idx} failed: {res_or_exc}")
                        # Optionally add error placeholder to results
                        all_results.append({"id": prompt_dataset[i_outer + res_idx].get("id", "unknown"), "error": str(res_or_exc)})
                    elif res_or_exc is not None:
                        all_results.append(res_or_exc)
                logger.info(f"Completed batch up to sample {i_outer + len(batch_tasks)}. Total results/errors: {len(all_results)}")

        finally:
            if http_session:
                await http_session.close()
        
        # Save results if output_file is specified
        output_file_path = self.cfg.output.get("results_file", None)
        if output_file_path:
            if not os.path.isabs(output_file_path) and self.cfg.hydra_output_dir: # Assuming hydra_output_dir is passed in cfg
                 output_file_path = os.path.join(self.cfg.hydra_output_dir, output_file_path)
            try:
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                with open(output_file_path, "w", encoding="utf-8") as f:
                    for result_item in all_results:
                        f.write(json.dumps(result_item) + "\n")
                logger.info(f"Detailed results saved to: {os.path.abspath(output_file_path)}")
            except Exception as e:
                logger.error(f"Failed to save results to {output_file_path}: {e}")

        logger.info("Evaluation pipeline run finished.")
        return all_results
