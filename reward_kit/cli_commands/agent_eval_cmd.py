"""
CLI command for running agent evaluations.
"""
import os
import json
import logging
import asyncio
import traceback
import uuid
from pathlib import Path

from reward_kit.agent import load_task_from_file, AgentEvaluator
from .common import setup_logging, check_agent_environment # check_environment is not used directly here

# Helper functions specific to agent_eval_command

def validate_task_bundle(task_dir: str) -> tuple[bool, str]:
    """
    Validate that a directory contains the required files for a task bundle.
    Args:
        task_dir: Path to the task bundle directory
    Returns:
        (bool, str) tuple indicating success and error message if failed
    """
    task_path = Path(task_dir)
    if not task_path.exists():
        return False, f"Task directory '{task_dir}' not found"
    if not task_path.is_dir():
        return False, f"'{task_dir}' is not a directory"

    required_files = ["tools.py", "reward.py"] # For current 'Task Bundle' design
    missing_files = [f for f in required_files if not (task_path / f).exists()]
    if missing_files:
        return False, f"Missing required files in task bundle: {', '.join(missing_files)}"

    task_jsonl = task_path / "task.jsonl"
    if not task_jsonl.exists():
        return False, f"No task.jsonl found in '{task_dir}'"

    init_file = task_path / "__init__.py"
    if not init_file.exists():
        return False, f"Missing __init__.py in '{task_dir}'. Task bundle must be a proper Python package."
    return True, ""

def find_task_dataset(args) -> tuple[Path | None, bool]:
    """
    Find and validate the task dataset from arguments.
    Returns: (dataset_path, is_task_dir) tuple or (None, False) if not found/invalid.
    """
    if args.task_dir:
        valid, error_msg = validate_task_bundle(args.task_dir)
        if not valid:
            print(f"Error: {error_msg}")
            return None, False
        task_jsonl_path = Path(args.task_dir) / "task.jsonl"
        return task_jsonl_path, True
    
    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"Error: Dataset file '{args.dataset}' not found")
            return None, False
        return dataset_path, False
        
    print("Error: Either --task-dir or --dataset must be specified for agent-eval.")
    return None, False

def get_toolset_config(args, is_task_dir=False) -> dict | None:
    """Get the toolset configuration based on CLI arguments."""
    config = {}
    if args.registries:
        registries = {}
        for reg_spec in args.registries:
            if "=" not in reg_spec:
                print(f"Error: Registry format should be 'name=path', got '{reg_spec}'")
                return None
            name, path = reg_spec.split("=", 1)
            registries[name] = path
        config["registries"] = registries
    if args.registry_override:
        config["registry_override"] = args.registry_override
    if args.evaluator:
        config["evaluator"] = args.evaluator
    return config

def export_tool_specs(tools_spec: list, export_dir: str) -> str:
    """Export tool specifications to JSON files for manual testing."""
    os.makedirs(export_dir, exist_ok=True)
    spec_file = os.path.join(export_dir, "tools_spec.json")
    with open(spec_file, "w") as f:
        json.dump(tools_spec, f, indent=2)

    for tool in tools_spec:
        tool_name = tool["function"]["name"]
        template_file = os.path.join(export_dir, f"{tool_name}_template.json")
        template = {param_name: "" for param_name in tool["function"]["parameters"]["properties"]}
        with open(template_file, "w") as f:
            json.dump(template, f, indent=2)
    return spec_file

# Main command function
def agent_eval_command(args):
    """Run agent evaluation on a dataset"""
    setup_logging(args.verbose, args.debug)
    logger = logging.getLogger("agent_eval")

    dataset_path, is_task_dir = find_task_dataset(args)
    if not dataset_path:
        return 1

    if args.validate_only: # Should be checked after find_task_dataset successfully validates task_dir
        if args.task_dir : # validate_only makes most sense with task_dir
             print(f"Task bundle in '{args.task_dir}' successfully validated.")
        else:
            print("Dataset path seems valid (file exists). Use --task-dir for full bundle validation.")
        return 0
    
    # Environment checks for actual evaluation runs
    if args.test_mode:
        print("Running in test mode - validating tool setup without requiring API keys.")
        check_agent_environment(test_mode=True) # Prints informational notes
    elif args.no_sim_user and not args.model and not os.environ.get("MODEL_AGENT"):
        print("Warning: No model specified. Since --no-sim-user is active, proceeding to verify tool setup only.")
    elif not check_agent_environment(test_mode=False): # test_mode is False here
        return 1


    toolset_config = get_toolset_config(args, is_task_dir)
    if toolset_config is None: # Should not happen if logic is correct, but defensive
        return 1

    try:
        tasks = load_task_from_file(str(dataset_path))
        if not tasks:
            print(f"Error: No tasks found in dataset file '{dataset_path}'")
            return 1

        if args.task_ids:
            requested_ids = set(args.task_ids.split(","))
            tasks = [t for t in tasks if t.get("id") in requested_ids]
            if not tasks:
                print(f"Error: No tasks found with IDs: {args.task_ids}")
                return 1
        
        if args.max_tasks and args.max_tasks > 0:
            tasks = tasks[:args.max_tasks]

        logger.info(f"Loaded {len(tasks)} tasks from {dataset_path}")
        successes = 0
        failures = 0

        for i, task_data in enumerate(tasks):
            task_id = task_data.get("id", f"task_{uuid.uuid4().hex[:8]}") # Use short UUID
            print(f"\nProcessing task {i+1}/{len(tasks)}: {task_id}")

            toolset_path = task_data.get("toolset")
            if not toolset_path:
                logger.error(f"Task {task_id} has no toolset defined.")
                failures += 1
                continue
            
            if toolset_config.get("registry_override"):
                toolset_path = toolset_config["registry_override"]
                logger.info(f"Overriding toolset to '{toolset_path}' for task {task_id}")

            reward_module_path = ".".join(toolset_path.split(".")[:-1] + ["reward"])
            if toolset_config.get("evaluator"):
                reward_module_path = toolset_config["evaluator"]
                logger.info(f"Using custom evaluator '{reward_module_path}' for task {task_id}")

            seed_sql_val = task_data.get("seed_sql")
            seed_file_path = None
            if seed_sql_val and seed_sql_val.startswith("file:"):
                seed_file_relative = seed_sql_val[5:]
                base_content_path = Path(args.task_dir) if is_task_dir else dataset_path.parent
                seed_file_path = base_content_path / seed_file_relative
                seed_sql_val = None # Clear it as we're using file path

            try:
                # This is the core async runner for a single task evaluation
                async def run_single_task_evaluation():
                    evaluator = AgentEvaluator(
                        task_id=task_id,
                        toolset_path=toolset_path,
                        reward_path=reward_module_path, # This should be just the module path
                        base_dir=args.output_dir,
                        seed_sql=seed_sql_val,
                        seed_file=str(seed_file_path) if seed_file_path else None,
                    )
                    await evaluator.setup()
                    run_id = f"run_{uuid.uuid4().hex[:8]}"
                    run_db_path = await evaluator.create_run(run_id)
                    logger.info(f"Created evaluation run for task {task_id} at {run_db_path}")

                    tools_spec_list = evaluator.tool_registry.get_openai_tools()
                    logger.info(f"Task {task_id} - Available tools ({len(tools_spec_list)}):")
                    for tool_spec_item in tools_spec_list:
                        logger.info(f"  - {tool_spec_item['function']['name']}: {tool_spec_item['function']['description']}")
                    
                    if args.export_tools:
                        export_target_dir = Path(args.export_tools) / task_id
                        spec_file_path = export_tool_specs(tools_spec_list, str(export_target_dir))
                        logger.info(f"Exported tool specifications for task {task_id} to {spec_file_path}")

                    current_messages = list(task_data.get("initial_messages", [])) # Make a mutable copy
                    
                    # Determine model for this run
                    agent_model_str = args.model or os.environ.get("MODEL_AGENT")

                    if args.test_mode or (args.no_sim_user and not agent_model_str):
                        logger.info(f"Task {task_id}: Running in tool validation/mock mode.")
                        if args.mock_response and tools_spec_list:
                            first_tool_name = tools_spec_list[0]["function"]["name"]
                            logger.info(f"Task {task_id}: Simulating execution of tool: {first_tool_name} with mock response.")
                            try:
                                tool_call_result = await evaluator.execute_tool(run_id, first_tool_name, {})
                                logger.info(f"Task {task_id}: Mock tool execution result: {tool_call_result}")
                            except Exception as e_mock:
                                logger.warning(f"Task {task_id}: Mock tool execution failed (potentially expected in test mode): {e_mock}")
                        else:
                            logger.info(f"Task {task_id}: Tools loaded and setup verified.")
                        # Simplified evaluation for test mode if not --no-sim-user
                        if args.test_mode and not args.no_sim_user :
                             logger.info(f"Task {task_id}: Running evaluation with current (potentially mock) data...")
                             # Prepare eval_kwargs from task_data
                             eval_kwargs_from_task = {k: v for k, v in task_data.items() if k not in ["id", "toolset", "initial_messages", "seed_sql", "sim_user_prompt", "n_rollouts"]}
                             evaluation_result = await evaluator.evaluate(run_id=run_id, messages=current_messages, **eval_kwargs_from_task)
                             logger.info(f"Task {task_id}: Test mode evaluation result: {evaluation_result}")
                        return True # Mark as success for test/validation pass

                    if not agent_model_str:
                        logger.error(f"Task {task_id}: No agent model specified. Use --model or set MODEL_AGENT.")
                        return False
                    
                    logger.info(f"Task {task_id}: Running evaluation with model: {agent_model_str}")
                    provider_name = agent_model_str.split("/")[0] if "/" in agent_model_str else "openai" # Default to openai
                    
                    # Actual agent interaction loop (simplified for PoC)
                    # This part needs to be significantly fleshed out for multi-turn, tool use, etc.
                    # For now, simulate one turn of OpenAI call if provider is openai
                    if provider_name == "openai":
                        import openai # Conditional import
                        try:
                            api_key_val = os.environ.get("OPENAI_API_KEY")
                            client = openai.OpenAI(api_key=api_key_val) # Add error handling for client init
                            model_name_str = agent_model_str.split("/")[1] if "/" in agent_model_str else agent_model_str
                            
                            logger.info(f"Task {task_id}: Calling OpenAI model {model_name_str}...")
                            api_response = client.chat.completions.create(
                                model=model_name_str, messages=current_messages, tools=tools_spec_list, tool_choice="auto"
                            )
                            assistant_msg_obj = api_response.choices[0].message
                            current_messages.append({"role": "assistant", "content": assistant_msg_obj.content or ""})

                            if assistant_msg_obj.tool_calls:
                                for tool_call_obj in assistant_msg_obj.tool_calls:
                                    tool_func_name = tool_call_obj.function.name
                                    tool_func_params = json.loads(tool_call_obj.function.arguments)
                                    logger.info(f"Task {task_id}: Agent requests tool call: {tool_func_name} with params: {tool_func_params}")
                                    tool_call_output = await evaluator.execute_tool(run_id, tool_func_name, tool_func_params)
                                    current_messages.append({
                                        "role": "tool", "tool_call_id": tool_call_obj.id,
                                        "name": tool_func_name, "content": json.dumps(tool_call_output) # Ensure content is string
                                    })
                                # Potentially another call to LLM with tool results
                        except Exception as e_openai:
                            logger.error(f"Task {task_id}: OpenAI API call failed: {e_openai}")
                            if args.debug: traceback.print_exc()
                            return False # Mark task as failed if API call fails
                    
                    elif provider_name == "anthropic":
                        # Placeholder for Anthropic - Pylint error was noted for this import
                        logger.warning(f"Task {task_id}: Anthropic provider logic is a placeholder.")
                        # import anthropic # This would be here
                        # ... anthropic specific logic ...
                        # For now, just append a dummy message if anthropic
                        current_messages.append({"role": "assistant", "content": "Anthropic response placeholder."})


                    else:
                        logger.error(f"Task {task_id}: Unsupported model provider: {provider_name}")
                        return False

                    # Final evaluation call
                    eval_kwargs_from_task = {k: v for k, v in task_data.items() if k not in ["id", "toolset", "initial_messages", "seed_sql", "sim_user_prompt", "n_rollouts"]}
                    evaluation_result = await evaluator.evaluate(run_id=run_id, messages=current_messages, **eval_kwargs_from_task)
                    logger.info(f"Task {task_id}: Final evaluation result: {evaluation_result}")
                    # Determine success based on score, e.g. if score == 1.0
                    return evaluation_result.get("score", 0.0) > 0.5 # Example success condition

                # Run the async function for a single task
                task_success = asyncio.run(run_single_task_evaluation())
                if task_success:
                    successes += 1
                else:
                    failures += 1
            
            except Exception as e_task_setup:
                logger.error(f"Error setting up or running evaluator for task {task_id}: {e_task_setup}")
                if args.verbose or args.debug: traceback.print_exc()
                failures += 1

        print(f"\nAgent evaluation summary: {successes} tasks succeeded, {failures} tasks failed.")
        return 0 if failures == 0 else 1

    except Exception as e_main:
        print(f"Critical error during agent evaluation process: {e_main}")
        if args.verbose or args.debug: traceback.print_exc()
        return 1
