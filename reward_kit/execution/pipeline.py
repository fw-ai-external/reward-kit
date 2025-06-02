"""
Core evaluation execution pipeline for reward-kit.
This module orchestrates dataset loading, model response generation (optional),
and evaluation using specified reward functions.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import aiohttp
import hydra  # For hydra.utils.instantiate if used for dataset loading
from datasets import Dataset, DatasetDict
from hydra.errors import InstantiationException  # For specific error handling
from mcp import types as mcp_types  # Added for tool discovery types
from omegaconf import DictConfig, OmegaConf  # For config handling

from reward_kit.auth import get_fireworks_api_key  # For Fireworks client
from reward_kit.datasets.loader import load_and_process_dataset  # Direct import
from reward_kit.generation.cache import ResponseCache
from reward_kit.generation.clients import (  # Assuming Fireworks for now
    FireworksModelClient,
    ModelClient,
)
from reward_kit.mcp.clients import IntermediaryMCPClient  # Added import
from reward_kit.models import Message  # For constructing messages for reward function
from reward_kit.utils.module_loader import load_function as load_reward_function

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    def __init__(self, pipeline_cfg: DictConfig):
        self.cfg = pipeline_cfg  # Root config for this pipeline run

        # Initialize components based on config
        logger.info("Initializing EvaluationPipeline...")

        self.model_client: Optional[ModelClient] = None
        if self.cfg.generation.enabled:
            api_key = get_fireworks_api_key()
            if not api_key:
                logger.error("Fireworks API key not found, but generation is enabled.")
                raise ValueError(
                    "API key required for Fireworks model client when generation is enabled."
                )
            self.model_client = FireworksModelClient(
                client_config=self.cfg.generation,
                api_key=api_key,
            )
            logger.info(
                f"Initialized FireworksModelClient for model: {self.cfg.generation.model_name}"
            )

        self.cache = ResponseCache(self.cfg.generation.cache)
        logger.info("ResponseCache initialized.")

        self.reward_function = load_reward_function(self.cfg.reward.function_path)
        logger.info(f"Loaded reward function from: {self.cfg.reward.function_path}")

        self.mcp_intermediary_client: Optional[IntermediaryMCPClient] = None
        if self.cfg.get("agent") and self.cfg.agent.get("type") == "mcp_agent":
            if not self.cfg.agent.get("intermediary_server_url"):
                raise ValueError(
                    "agent.intermediary_server_url must be configured for mcp_agent type."
                )
            logger.info(
                f"Pipeline configured for mcp_agent. IntermediaryMCPClient will be initialized in run()."
            )

    async def _process_single_sample(
        self,
        sample: Dict[str, Any],
        http_session: Optional[aiohttp.ClientSession],
        original_index: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        sample_id_fallback = (
            f"idx_{original_index}"
            if original_index is not None
            else "unknown_id_" + os.urandom(4).hex()
        )
        sample_id = sample.get("id", sample_id_fallback)
        user_query = sample.get("user_query")
        ground_truth_for_eval = sample.get("ground_truth_for_eval")

        if user_query is None or ground_truth_for_eval is None:
            logger.warning(
                f"Skipping sample {sample_id} due to missing 'user_query' or 'ground_truth_for_eval'."
            )
            return None

        messages_for_generation: List[Dict[str, str]] = []
        # system_prompt will be constructed later if MCP agent is active
        original_system_prompt = sample.get("system_prompt") or self.cfg.get(
            "system_prompt"
        )

        assistant_response_content: Optional[str] = None
        discovered_tools_for_llm_prompt: List[Dict[str, Any]] = (
            []
        )  # Store tool dicts for prompt

        # MCP Agent specific logic: Initialize session and discover tools first
        if self.mcp_intermediary_client and self.cfg.agent.type == "mcp_agent":
            rk_session_id_for_tools: Optional[str] = None
            try:
                mcp_backend_ref_for_tools = self.cfg.agent.get("mcp_backend_ref")
                if not mcp_backend_ref_for_tools:
                    raise ValueError(
                        "agent.mcp_backend_ref must be configured for mcp_agent tool discovery."
                    )

                backend_requests_for_tools = [
                    {"backend_name_ref": mcp_backend_ref_for_tools, "num_instances": 1}
                ]
                logger.info(
                    f"Sample {sample_id}: Pre-generation: Initializing MCP session for tool discovery (backend: '{mcp_backend_ref_for_tools}')..."
                )
                init_response_for_tools = (
                    await self.mcp_intermediary_client.initialize_session(
                        backend_requests_for_tools
                    )
                )

                if init_response_for_tools.get("error"):
                    raise RuntimeError(
                        f"MCP session initialization for tool discovery failed: {init_response_for_tools.get('error_details', init_response_for_tools['error'])}"
                    )

                rk_session_id_for_tools = init_response_for_tools.get("rk_session_id")
                initialized_backends_for_tools = init_response_for_tools.get(
                    "initialized_backends", []
                )

                if not rk_session_id_for_tools or not initialized_backends_for_tools:
                    raise RuntimeError(
                        f"MCP session initialization for tool discovery returned malformed response: {init_response_for_tools}"
                    )

                for backend_info in initialized_backends_for_tools:
                    current_backend_name_ref = backend_info.get("backend_name_ref")
                    instances_info = backend_info.get("instances", [])
                    if not current_backend_name_ref or not instances_info:
                        continue
                    for inst_info_dict in instances_info:
                        current_instance_id = inst_info_dict.get("instance_id")
                        if not current_instance_id:
                            continue
                        logger.info(
                            f"Sample {sample_id}: Pre-generation: Discovering tools for backend '{current_backend_name_ref}', instance '{current_instance_id}'..."
                        )
                        list_tools_result = (
                            await self.mcp_intermediary_client.list_backend_tools(
                                rk_session_id=rk_session_id_for_tools,
                                instance_id=current_instance_id,
                                backend_name_ref=current_backend_name_ref,
                            )
                        )
                        if list_tools_result and list_tools_result.tools:
                            for tool_obj in list_tools_result.tools:
                                discovered_tools_for_llm_prompt.append(
                                    tool_obj.model_dump(exclude_none=True)
                                )
                logger.info(
                    f"Sample {sample_id}: Pre-generation: Discovered {len(discovered_tools_for_llm_prompt)} tools."
                )
                # We keep rk_session_id_for_tools to clean up this specific session later if generation fails before main MCP block.
                # The main MCP block will re-initialize its own session. This is slightly inefficient but safer for now.
                # A more advanced approach would re-use this session or pass discovered tools.
            except Exception as e_tool_discovery:
                logger.error(
                    f"Sample {sample_id}: Error during pre-generation tool discovery: {e_tool_discovery}",
                    exc_info=True,
                )
                # Proceed without tools if discovery fails, but log it.
                # Cleanup the session if it was partially initialized
                if rk_session_id_for_tools and self.mcp_intermediary_client:
                    try:
                        logger.warning(
                            f"Sample {sample_id}: Cleaning up MCP session '{rk_session_id_for_tools}' due to error in pre-generation tool discovery."
                        )
                        await self.mcp_intermediary_client.cleanup_session(
                            rk_session_id_for_tools
                        )
                    except Exception as e_cleanup_discovery:
                        logger.error(
                            f"Sample {sample_id}: Error cleaning up discovery session '{rk_session_id_for_tools}': {e_cleanup_discovery}",
                            exc_info=True,
                        )
                discovered_tools_for_llm_prompt = []  # Ensure it's empty on failure

        # Construct system prompt and prepare tools for ModelClient
        system_prompt_content = original_system_prompt
        openai_formatted_tools: Optional[List[Dict[str, Any]]] = None

        if (
            self.mcp_intermediary_client
            and self.cfg.agent.type == "mcp_agent"
            and discovered_tools_for_llm_prompt
        ):
            openai_formatted_tools = []
            for mcp_tool_dict in discovered_tools_for_llm_prompt:
                # Ensure inputSchema is present, default to empty object if not (though MCP spec implies it's required)
                input_schema = mcp_tool_dict.get("inputSchema", {})
                openai_formatted_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": mcp_tool_dict.get("name", "unknown_tool"),
                            "description": mcp_tool_dict.get("description", ""),
                            "parameters": input_schema,
                        },
                    }
                )

            # Simplify system prompt if tools are being passed via the 'tools' parameter
            # The FireworksModelClient will attempt to use the 'tools' parameter.
            # If the underlying API doesn't support it, it might ignore it, or we might need a fallback.
            # For now, assume we want a cleaner prompt if 'tools' are provided.
            if original_system_prompt:
                system_prompt_content = f"{original_system_prompt}\n\nYou have access to tools. Use them if appropriate."
            else:
                system_prompt_content = "You are a helpful assistant with access to tools. Use them if appropriate to fulfill the user's request."

        if system_prompt_content:
            messages_for_generation.append(
                {"role": "system", "content": system_prompt_content}
            )
        messages_for_generation.append({"role": "user", "content": user_query})

        # Standard response generation/caching logic
        if self.cfg.generation.enabled:
            if not self.model_client or not http_session:
                logger.error(
                    f"Model client or HTTP session not available for generation for sample {sample_id}."
                )
                return {"id": sample_id, "error": "Generation client not configured"}

            # Cache key should ideally include a hash of tools if they affect generation,
            # or we assume that for the same (original_system_prompt, user_query), if tools are available,
            # the generation path is different and caching might need to be more sophisticated.
            # For now, using original_system_prompt for cache key.
            if self.cfg.generation.cache.enabled:
                assistant_response_content = self.cache.get(
                    sample_id=sample_id,
                    system_prompt=original_system_prompt,
                    user_query=user_query,
                    model_name=self.model_client.model_name,
                    temperature=self.model_client.temperature,
                    top_p=self.model_client.top_p,
                    top_k=self.model_client.top_k,
                    min_p=self.model_client.min_p,
                    max_tokens=self.model_client.max_tokens,
                    reasoning_effort=self.model_client.reasoning_effort,
                )
                if assistant_response_content:
                    logger.info(f"Using cached response for sample {sample_id}")

            if not assistant_response_content:
                try:
                    generation_result = await self.model_client.generate(
                        messages=messages_for_generation,
                        session=http_session,
                        tools=openai_formatted_tools,
                    )

                    if generation_result.tool_calls:
                        # LLM decided to call tool(s)
                        logger.info(
                            f"Sample {sample_id}: LLM responded with tool_calls: {generation_result.tool_calls}"
                        )
                        # For logging/preview, represent tool_calls as string. Actual execution is next.
                        assistant_response_content = json.dumps(
                            [tc.model_dump() for tc in generation_result.tool_calls]
                        )
                        # TODO: Implement actual tool execution logic here.
                        # For now, we just log that tools were called and proceed to state capture.
                        # The reward function will see that assistant_response_content is a tool call JSON.
                    elif generation_result.content:
                        assistant_response_content = generation_result.content
                    else:
                        assistant_response_content = (
                            None  # No content and no tool calls
                        )

                    if (
                        assistant_response_content
                        and self.cfg.generation.cache.enabled
                        and self.model_client.temperature == 0.0
                    ):
                        # Caching decision: if it was a tool call, do we cache the tool call, or wait for its result?
                        # For now, caching the direct output (text or tool call string representation).
                        self.cache.put(
                            sample_id=sample_id,
                            system_prompt=original_system_prompt,
                            user_query=user_query,
                            model_name=self.model_client.model_name,
                            temperature=self.model_client.temperature,
                            response=assistant_response_content,
                            top_p=self.model_client.top_p,
                            top_k=self.model_client.top_k,
                            min_p=self.model_client.min_p,
                            max_tokens=self.model_client.max_tokens,
                            reasoning_effort=self.model_client.reasoning_effort,
                        )
                        logger.info(f"Cached new response for sample {sample_id}")
                except Exception as e:
                    logger.error(
                        f"Failed to generate response for sample {sample_id}: {e}",
                        exc_info=True,
                    )
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
            assistant_response_col_name = self.cfg.dataset.get(
                "column_mapping", {}
            ).get("assistant_response_column", "assistant_response")
            assistant_response_content = sample.get(assistant_response_col_name)
            if assistant_response_content:
                logger.info(
                    f"Using pre-existing response from input sample {sample_id} (column: {assistant_response_col_name})"
                )
            else:
                gen_cfg = self.cfg.generation
                model_name_for_cache = gen_cfg.get("model_name", "unknown_model")
                temperature_for_cache = gen_cfg.get("temperature", 0.0)
                top_p_for_cache = gen_cfg.get("top_p", 0.95)
                top_k_for_cache = gen_cfg.get("top_k", 20)
                min_p_for_cache = gen_cfg.get("min_p", 0.0)
                max_tokens_for_cache = gen_cfg.get("max_tokens", 1024)
                reasoning_effort_for_cache = gen_cfg.get("reasoning_effort", None)
                assistant_response_content = self.cache.get(
                    sample_id=sample_id,
                    system_prompt=original_system_prompt,
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
                    f"Generation disabled. No pre-existing response or cache hit for sample {sample_id}."
                )
                return {
                    "id": sample_id,
                    "error": "No pre-existing or cached response",
                    "evaluation_score": 0.0,
                    "evaluation_reason": "No response available (generation disabled)",
                }

        final_messages_for_eval: List[Message] = []
        if system_prompt_content:
            final_messages_for_eval.append(
                Message(role="system", content=system_prompt_content)
            )
        final_messages_for_eval.append(Message(role="user", content=user_query))
        final_messages_for_eval.append(
            Message(role="assistant", content=assistant_response_content)
        )

        if self.mcp_intermediary_client and self.cfg.agent.type == "mcp_agent":
            rk_session_id: Optional[str] = None
            final_filesystem_state_from_mcp: Optional[Dict[str, Any]] = None
            mcp_agent_eval_result: Optional[Dict[str, Any]] = None
            # discovered_tools_for_sample is now discovered_tools_for_llm_prompt, populated earlier
            # We will add it to the results at the end.

            try:
                # If tool discovery happened before generation, we might have rk_session_id_for_tools.
                # The current design re-initializes a session for the main agent execution block.
                # This is to ensure a clean state for the agent's interaction, separate from discovery.
                # If rk_session_id_for_tools exists from a successful pre-discovery, clean it up now
                # as the main block will create its own.
                if (
                    "rk_session_id_for_tools" in locals()
                    and locals()["rk_session_id_for_tools"]
                    and self.mcp_intermediary_client
                ):
                    logger.info(
                        f"Sample {sample_id}: Cleaning up pre-generation tool discovery session '{locals()['rk_session_id_for_tools']}'."
                    )
                    try:
                        await self.mcp_intermediary_client.cleanup_session(
                            locals()["rk_session_id_for_tools"]
                        )
                    except Exception as e_cleanup_prediscovery:
                        logger.error(
                            f"Sample {sample_id}: Error cleaning up pre-discovery session '{locals()['rk_session_id_for_tools']}': {e_cleanup_prediscovery}",
                            exc_info=True,
                        )

                mcp_backend_ref = self.cfg.agent.get("mcp_backend_ref")
                if not mcp_backend_ref:
                    raise ValueError(
                        "agent.mcp_backend_ref must be configured for mcp_agent."
                    )

                # Assuming one primary backend for now for tool discovery and state capture.
                # This could be extended if multiple backends are involved in the agent's task.
                backend_requests = [
                    {"backend_name_ref": mcp_backend_ref, "num_instances": 1}
                ]
                logger.info(
                    f"Sample {sample_id}: Initializing MCP session for backend '{mcp_backend_ref}'..."
                )
                init_response = await self.mcp_intermediary_client.initialize_session(
                    backend_requests
                )

                if init_response.get("error"):
                    raise RuntimeError(
                        f"MCP session initialization failed: {init_response.get('error_details', init_response['error'])}"
                    )

                rk_session_id = init_response.get("rk_session_id")
                initialized_backends = init_response.get("initialized_backends", [])

                if not rk_session_id or not initialized_backends:
                    raise RuntimeError(
                        f"MCP session initialization returned malformed response (missing rk_session_id or initialized_backends): {init_response}"
                    )

                # Tool discovery is now done before generation.
                # The main session initialization for agent execution happens here.
                # We still need to get the primary_instance_id for state capture from this new session.
                primary_instance_id_for_state_capture: Optional[str] = None
                initialized_backends_main = init_response.get(
                    "initialized_backends", []
                )  # From the main session init
                for backend_info in initialized_backends:
                    if backend_info.get("backend_name_ref") == mcp_backend_ref:
                        instances_info = backend_info.get("instances", [])
                        if instances_info:
                            primary_instance_id_for_state_capture = instances_info[
                                0
                            ].get("instance_id")
                            break

                if not primary_instance_id_for_state_capture:
                    raise RuntimeError(
                        f"Primary MCP instance ID for state capture (backend: {mcp_backend_ref}) not found in main init response: {init_response}"
                    )

                logger.info(
                    f"Sample {sample_id}: Main MCP session initialized. rk_session_id='{rk_session_id}', primary instance_id for state capture='{primary_instance_id_for_state_capture}'. Tools provided to LLM: {len(discovered_tools_for_llm_prompt)}"
                )

                executed_tool_call_results: List[
                    Dict[str, Union[str, Dict[str, Any]]]
                ] = []
                if (
                    generation_result.tool_calls
                ):  # generation_result is from the model_client.generate call
                    for tool_call in generation_result.tool_calls:
                        tool_name_to_execute = tool_call.function.name
                        try:
                            tool_args_to_execute = json.loads(
                                tool_call.function.arguments
                            )
                            if not isinstance(tool_args_to_execute, dict):
                                raise ValueError(
                                    "Tool arguments did not parse to a dictionary."
                                )
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Sample {sample_id}: Failed to parse tool arguments for '{tool_name_to_execute}': {tool_call.function.arguments}. Error: {e}"
                            )
                            executed_tool_call_results.append(
                                {
                                    "tool_name": tool_name_to_execute,
                                    "arguments_str": tool_call.function.arguments,
                                    "error": f"Argument parsing failed: {e}",
                                }
                            )
                            continue  # Skip this tool call

                        logger.info(
                            f"Sample {sample_id}: Executing LLM-requested tool '{tool_name_to_execute}' with args {tool_args_to_execute} on instance '{primary_instance_id_for_state_capture}' of backend '{mcp_backend_ref}'."
                        )
                        try:
                            # Assuming the tool call should target the same primary backend and instance used for state capture.
                            # This might need to be more sophisticated if the agent can target multiple backends/instances.
                            execution_result = await self.mcp_intermediary_client.call_backend_tool(
                                rk_session_id=rk_session_id,
                                instance_id=primary_instance_id_for_state_capture,  # Use the instance ID determined for the main backend
                                backend_name_ref=mcp_backend_ref,
                                tool_name=tool_name_to_execute,
                                tool_args=tool_args_to_execute,
                            )
                            logger.info(
                                f"Sample {sample_id}: Tool '{tool_name_to_execute}' executed. Result: {execution_result}"
                            )
                            executed_tool_call_results.append(
                                {
                                    "tool_name": tool_name_to_execute,
                                    "arguments": tool_args_to_execute,
                                    "result": execution_result,
                                }
                            )
                        except Exception as e_exec:
                            logger.error(
                                f"Sample {sample_id}: Error executing tool '{tool_name_to_execute}': {e_exec}",
                                exc_info=True,
                            )
                            executed_tool_call_results.append(
                                {
                                    "tool_name": tool_name_to_execute,
                                    "arguments": tool_args_to_execute,
                                    "error": f"Execution failed: {str(e_exec)}",
                                }
                            )
                # After potential tool executions, capture the final state
                state_capture_tool = self.cfg.agent.get("state_capture_tool")
                state_capture_args_cfg = self.cfg.agent.get(
                    "state_capture_args", OmegaConf.create({})
                )
                state_capture_args = dict(state_capture_args_cfg)

                if not state_capture_tool:
                    logger.warning(
                        f"Sample {sample_id}: agent.state_capture_tool not configured. Cannot capture final MCP state."
                    )
                else:
                    logger.info(
                        f"Sample {sample_id}: Capturing final state using tool '{state_capture_tool}' with args {state_capture_args} on instance '{primary_instance_id_for_state_capture}'..."
                    )
                    final_filesystem_state_from_mcp = (
                        await self.mcp_intermediary_client.call_backend_tool(
                            rk_session_id=rk_session_id,
                            instance_id=primary_instance_id_for_state_capture,
                            backend_name_ref=mcp_backend_ref,
                            tool_name=state_capture_tool,
                            tool_args=state_capture_args,
                        )
                    )
                    if (
                        final_filesystem_state_from_mcp
                        and final_filesystem_state_from_mcp.get("error")
                    ):
                        logger.error(
                            f"Sample {sample_id}: Failed to capture MCP state: {final_filesystem_state_from_mcp.get('error_details', final_filesystem_state_from_mcp['error'])}"
                        )
                        final_filesystem_state_from_mcp = {
                            "error_capturing_state": True,
                            "details": final_filesystem_state_from_mcp.get("error"),
                        }

                eval_params_cfg = self.cfg.reward.get("params", OmegaConf.create({}))
                eval_params = dict(eval_params_cfg)
                eval_result_obj = self.reward_function(
                    messages=final_messages_for_eval,
                    ground_truth=ground_truth_for_eval,
                    final_filesystem_state=final_filesystem_state_from_mcp,
                    **eval_params,
                )
                logger.info(
                    f"Sample ID: {sample_id} (MCP Agent), Score: {eval_result_obj.score:.2f}, Reason: {eval_result_obj.reason}"
                )
                mcp_agent_eval_result = {
                    "id": sample_id,
                    "user_query": user_query,
                    "system_prompt": system_prompt_content,
                    "assistant_response": assistant_response_content,
                    "ground_truth_for_eval": ground_truth_for_eval,
                    "discovered_tools": discovered_tools_for_llm_prompt,
                    "executed_tool_calls": executed_tool_call_results,  # Add executed tool call results
                    "final_mcp_state_captured": (
                        final_filesystem_state_from_mcp
                        if final_filesystem_state_from_mcp
                        else "Not captured or error"
                    ),
                    "evaluation_score": eval_result_obj.score,
                    "evaluation_reason": eval_result_obj.reason,
                    "evaluation_metrics": (
                        {k: v.model_dump() for k, v in eval_result_obj.metrics.items()}
                        if eval_result_obj.metrics
                        else {}
                    ),
                }
            except Exception as e_mcp:
                logger.error(
                    f"Error during MCP agent processing for sample {sample_id}: {e_mcp}",
                    exc_info=True,
                )
                mcp_agent_eval_result = {
                    "id": sample_id,
                    "user_query": user_query,
                    "ground_truth_for_eval": ground_truth_for_eval,
                    "assistant_response": assistant_response_content,
                    "error": f"MCP agent processing failed: {str(e_mcp)}",
                    "evaluation_score": 0.0,
                    "evaluation_reason": "MCP agent processing error",
                    "discovered_tools": discovered_tools_for_llm_prompt,
                    "executed_tool_calls": executed_tool_call_results,  # Add even on error if some calls were attempted
                }
            finally:
                # Cleanup main execution session
                if rk_session_id and self.mcp_intermediary_client:
                    logger.info(
                        f"Sample {sample_id}: Cleaning up MCP session '{rk_session_id}'..."
                    )
                    try:
                        await self.mcp_intermediary_client.cleanup_session(
                            rk_session_id
                        )
                        logger.info(
                            f"Sample {sample_id}: MCP session '{rk_session_id}' cleaned up successfully."
                        )
                    except Exception as e_cleanup:
                        logger.error(
                            f"Sample {sample_id}: Error cleaning up MCP session '{rk_session_id}': {e_cleanup}",
                            exc_info=True,
                        )
            return mcp_agent_eval_result

        else:
            eval_params_cfg = self.cfg.reward.get("params", OmegaConf.create({}))
            eval_params = dict(eval_params_cfg)
            eval_result_obj = self.reward_function(
                messages=final_messages_for_eval,
                ground_truth=ground_truth_for_eval,
                final_filesystem_state=None,  # No MCP state for non-agent evals
                **eval_params,
            )
            logger.info(
                f"Sample ID: {sample_id}, Score: {eval_result_obj.score:.2f}, Reason: {eval_result_obj.reason}"
            )
            return {
                "id": sample_id,
                "user_query": user_query,
                "system_prompt": system_prompt_content,
                "assistant_response": assistant_response_content,
                "ground_truth_for_eval": ground_truth_for_eval,
                "evaluation_score": eval_result_obj.score,
                "evaluation_reason": eval_result_obj.reason,
                "evaluation_metrics": (
                    {k: v.model_dump() for k, v in eval_result_obj.metrics.items()}
                    if eval_result_obj.metrics
                    else {}
                ),
            }

    async def run(self) -> List[Dict[str, Any]]:
        logger.info("Starting evaluation pipeline run...")

        try:
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

            dataset_source = getattr(
                prompt_dataset_config,
                "path_or_name",
                getattr(prompt_dataset_config, "base_dataset", "dataset"),
            )
            logger.info(f"Loaded {len(prompt_dataset)} samples from {dataset_source}.")
        except InstantiationException as ie:
            final_cause = ie
            while final_cause.__cause__ is not None:
                final_cause = final_cause.__cause__
            if (
                isinstance(final_cause, ValueError)
                and str(final_cause)
                == "Invalid pattern: '**' can only be an entire path component"
            ):
                base_dataset_config_name = prompt_dataset_config.get(
                    "base_dataset", "UnknownBaseDatasetConfig"
                )
                dataset_display_name = base_dataset_config_name
                helpful_message = (
                    f"Failed to load the base dataset specified as '{dataset_display_name}' in your derived dataset configuration. "
                    f"This occurred due to an internal error in the 'datasets' library (via fsspec): '{str(final_cause)}'.\n"
                    "The error message \"Invalid pattern: '**' can only be an entire path component\" often indicates issues with "
                    "how the 'datasets' library is resolving the path to the dataset, potential Hugging Face Hub connectivity/authentication problems, or a corrupted local cache.\n\n"
                    "Please try the following troubleshooting steps:\n"
                    "1. Verify Hugging Face Hub Token: Ensure your token is correctly configured (e.g., run `huggingface-cli login`).\n"
                    "2. Clear Datasets Cache: Try removing the subdirectory related to the actual Hugging Face dataset path/name from `~/.cache/huggingface/datasets/`.\n"
                    "3. Update Libraries: `pip install --upgrade datasets huggingface_hub fsspec`.\n"
                    "4. Test Direct Loading: (See previous detailed instructions for direct loading test script).\n"
                    f"Original InstantiationException details: {ie}"
                )
                logger.error(helpful_message, exc_info=False)
            else:
                logger.error(f"Failed to load prompt dataset: {ie}", exc_info=True)
            return []
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during dataset loading: {e}",
                exc_info=True,
            )
            return []

        all_results: List[Dict[str, Any]] = []
        limit_samples = self.cfg.evaluation_params.get("limit_samples", None)
        samples_to_process_count = len(prompt_dataset)
        if limit_samples is not None and limit_samples > 0:
            samples_to_process_count = min(len(prompt_dataset), limit_samples)

        logger.info(f"Processing {samples_to_process_count} samples.")

        http_session: Optional[aiohttp.ClientSession] = None
        if self.cfg.generation.enabled and self.model_client:
            http_session = aiohttp.ClientSession()

        if self.cfg.get("agent") and self.cfg.agent.get("type") == "mcp_agent":
            self.mcp_intermediary_client = IntermediaryMCPClient(
                intermediary_server_url=self.cfg.agent.intermediary_server_url
            )
            logger.info(
                f"Created IntermediaryMCPClient instance with URL: {self.cfg.agent.intermediary_server_url}"
            )

        async def execute_tasks():
            tasks = []
            nonlocal http_session

            if self.cfg.generation.enabled and self.model_client and not http_session:
                http_session = aiohttp.ClientSession()
                logger.debug(
                    "Created aiohttp.ClientSession for model_client in execute_tasks."
                )

            max_concurrent = self.cfg.generation.api_params.get(
                "max_concurrent_requests", 5
            )
            if not isinstance(max_concurrent, int) or max_concurrent <= 0:
                logger.warning(
                    f"Invalid max_concurrent_requests value ({max_concurrent}), defaulting to 5."
                )
                max_concurrent = 5
            semaphore = asyncio.Semaphore(max_concurrent)

            async def process_with_semaphore_wrapper(
                sample_idx: int, sample_data: Dict[str, Any]
            ):
                prelim_sample_id = sample_data.get("id", f"idx_{sample_idx}")
                async with semaphore:
                    logger.info(
                        f"Concurrency slot acquired for sample '{prelim_sample_id}', attempting to process."
                    )
                    return await self._process_single_sample(
                        sample_data, http_session, original_index=sample_idx
                    )

            for i in range(samples_to_process_count):
                tasks.append(process_with_semaphore_wrapper(i, prompt_dataset[i]))

            batch_size_for_logging = self.cfg.logging_params.get(
                "batch_log_interval", 10
            )
            if (
                not isinstance(batch_size_for_logging, int)
                or batch_size_for_logging <= 0
            ):
                logger.warning(
                    f"Invalid batch_log_interval ({batch_size_for_logging}), defaulting to 10."
                )
                batch_size_for_logging = 10

            for i_outer in range(0, len(tasks), batch_size_for_logging):
                batch_tasks = tasks[i_outer : i_outer + batch_size_for_logging]
                batch_results_values = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )
                for res_idx, res_or_exc in enumerate(batch_results_values):
                    if isinstance(res_or_exc, Exception):
                        logger.error(
                            f"Task for sample index {i_outer + res_idx} failed: {res_or_exc}",
                            exc_info=True,
                        )
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

        try:
            if self.mcp_intermediary_client:
                async with self.mcp_intermediary_client:
                    await execute_tasks()
            else:
                await execute_tasks()
        finally:
            if http_session:  # This is the session for the model_client
                await http_session.close()
                logger.debug(
                    "Closed aiohttp.ClientSession for model_client in main run() finally block."
                )

        output_file_path = self.cfg.output.get("results_file", None)
        if output_file_path:
            if not os.path.isabs(output_file_path) and self.cfg.hydra_output_dir:
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
