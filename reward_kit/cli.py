"""
Command-line interface for reward-kit.
"""

import argparse
import sys
import os
import json
import logging
import asyncio
import traceback
import uuid
from pathlib import Path
import importlib.util

from reward_kit.evaluation import preview_evaluation, create_evaluation

def setup_logging(verbose=False, debug=False):
    """Setup logging configuration"""
    if debug:
        log_level = logging.DEBUG
        format_str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    elif verbose:
        log_level = logging.INFO
        format_str = "%(levelname)s:%(name)s:%(message)s"
    else:
        log_level = logging.WARNING
        format_str = "%(levelname)s:%(message)s"
    
    logging.basicConfig(
        level=log_level,
        format=format_str
    )

def check_environment():
    """Check if required environment variables are set"""
    if not os.environ.get("FIREWORKS_API_KEY"):
        print("Warning: FIREWORKS_API_KEY environment variable is not set.")
        print("This is required for API calls. Set this variable before running the command.")
        print("Example: FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY reward-kit [command]")
        return False
    return True

def check_agent_environment(test_mode=False):
    """Check if required environment variables are set for agent evaluation"""
    missing_vars = []
    if not os.environ.get("MODEL_AGENT"):
        missing_vars.append("MODEL_AGENT")
    
    if test_mode:
        # In test mode, we don't require the environment variables
        if missing_vars:
            print(f"Note: The following environment variables are not set: {', '.join(missing_vars)}")
            print("Since you're running in test mode, these are not required.")
        return True
    
    if missing_vars:
        print(f"Warning: The following environment variables are not set: {', '.join(missing_vars)}")
        print("These are required for agent evaluation. Set these variables before running the command.")
        print("Example: MODEL_AGENT=openai/gpt-4o-mini reward-kit agent-eval [args]")
        print("Alternatively, use --test-mode to validate tool setup without requiring API keys.")
        return False
    return True

def preview_command(args):
    """Preview an evaluator with sample data"""
    
    # Check environment variables
    if not check_environment():
        return 1
    
    # Validate paths
    if args.metrics_folders:
        for folder in args.metrics_folders:
            if "=" not in folder:
                print(f"Error: Metric folder format should be 'name=path', got '{folder}'")
                return 1

    if not args.samples:
        print("Error: Sample file (--samples) is required for preview")
        return 1
    
    if not Path(args.samples).exists():
        print(f"Error: Sample file '{args.samples}' not found")
        return 1
        
    # Run preview
    try:
        preview_result = preview_evaluation(
            metric_folders=args.metrics_folders,
            sample_file=args.samples,
            max_samples=args.max_samples
        )
        
        preview_result.display()
        return 0
    except Exception as e:
        print(f"Error previewing evaluator: {str(e)}")
        return 1

def deploy_command(args):
    """Create and deploy an evaluator"""
    
    # Check environment variables
    if not check_environment():
        return 1
    
    # Validate paths
    if args.metrics_folders:
        for folder in args.metrics_folders:
            if "=" not in folder:
                print(f"Error: Metric folder format should be 'name=path', got '{folder}'")
                return 1
                
    if not args.id:
        print("Error: Evaluator ID (--id) is required for deployment")
        return 1
        
    # Create the evaluator
    try:
        evaluator = create_evaluation(
            evaluator_id=args.id,
            metric_folders=args.metrics_folders,
            display_name=args.display_name or args.id,
            description=args.description or f"Evaluator: {args.id}",
            force=args.force
        )
        
        print(f"Successfully created evaluator: {evaluator['name']}")
        return 0
    except Exception as e:
        print(f"Error creating evaluator: {str(e)}")
        return 1

def validate_task_bundle(task_dir):
    """
    Validate that a directory contains the required files for a task bundle.
    
    Args:
        task_dir: Path to the task bundle directory
        
    Returns:
        (bool, str) tuple indicating success and error message if failed
    """
    task_path = Path(task_dir)
    
    # Check if directory exists
    if not task_path.exists():
        return False, f"Task directory '{task_dir}' not found"
    
    if not task_path.is_dir():
        return False, f"'{task_dir}' is not a directory"
    
    # Check for required files
    required_files = ["tools.py", "reward.py"]
    missing_files = [f for f in required_files if not (task_path / f).exists()]
    
    if missing_files:
        return False, f"Missing required files in task bundle: {', '.join(missing_files)}"
    
    # Check for task.jsonl
    task_jsonl = task_path / "task.jsonl"
    if not task_jsonl.exists():
        return False, f"No task.jsonl found in '{task_dir}'"
    
    # Validate __init__.py exists to make it a proper package
    init_file = task_path / "__init__.py"
    if not init_file.exists():
        return False, f"Missing __init__.py in '{task_dir}'. Task bundle must be a proper Python package."
    
    return True, ""

def find_task_dataset(args):
    """Find and validate the task dataset from arguments.
    
    Args:
        args: CLI arguments
        
    Returns:
        (dataset_path, is_task_dir) tuple or (None, None) if not found
    """
    # If both task-dir and dataset are specified, use task-dir
    if args.task_dir:
        valid, error_msg = validate_task_bundle(args.task_dir)
        if not valid:
            print(f"Error: {error_msg}")
            return None, None
        
        # Find task.jsonl in the task directory
        task_jsonl = os.path.join(args.task_dir, "task.jsonl")
        return task_jsonl, True
    
    # Otherwise, use the dataset argument
    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"Error: Dataset file '{args.dataset}' not found")
            return None, None
        
        return args.dataset, False
    
    # Neither specified
    print("Error: Either --task-dir or --dataset must be specified")
    return None, None

def get_toolset_config(args, is_task_dir=False):
    """
    Get the toolset configuration based on CLI arguments.
    
    Args:
        args: CLI arguments
        is_task_dir: Whether the task is specified via task directory
        
    Returns:
        Dict with configuration or None if invalid
    """
    config = {}
    
    # Handle custom registries
    if args.registries:
        registries = {}
        for reg_spec in args.registries:
            if "=" not in reg_spec:
                print(f"Error: Registry format should be 'name=path', got '{reg_spec}'")
                return None
            
            name, path = reg_spec.split("=", 1)
            registries[name] = path
        
        config["registries"] = registries
    
    # Handle registry override
    if args.registry_override:
        config["registry_override"] = args.registry_override
    
    # Handle custom evaluator
    if args.evaluator:
        config["evaluator"] = args.evaluator
    
    return config

def export_tool_specs(tools_spec, export_dir):
    """
    Export tool specifications to JSON files for manual testing.
    
    Args:
        tools_spec: Tool specifications in OpenAI format
        export_dir: Directory to export to
        
    Returns:
        Path to the exported specification file
    """
    os.makedirs(export_dir, exist_ok=True)
    
    # Generate a spec file
    spec_file = os.path.join(export_dir, "tools_spec.json")
    with open(spec_file, "w") as f:
        json.dump(tools_spec, f, indent=2)
    
    # Generate a template for each tool
    for i, tool in enumerate(tools_spec):
        tool_name = tool["function"]["name"]
        template_file = os.path.join(export_dir, f"{tool_name}_template.json")
        
        # Create template with empty parameters
        template = {}
        for param_name in tool["function"]["parameters"]["properties"]:
            template[param_name] = ""
        
        with open(template_file, "w") as f:
            json.dump(template, f, indent=2)
    
    return spec_file

def agent_eval_command(args):
    """Run agent evaluation on a dataset"""
    # Import here to avoid circular imports
    from reward_kit.agent import load_task_from_file, AgentEvaluator
    
    # Set up logging
    setup_logging(args.verbose, args.debug)
    logger = logging.getLogger("agent_eval")
    
    # Find dataset file
    dataset_path, is_task_dir = find_task_dataset(args)
    if not dataset_path:
        return 1
    
    # If --validate-only, exit after validation
    if args.validate_only:
        print(f"Task bundle successfully validated.")
        return 0
    
    # If test mode or no-sim-user is specified, we don't need to check for API keys
    if args.test_mode:
        print("Running in test mode - validating tool setup without requiring API keys.")
        check_agent_environment(test_mode=True)
    # If --no-sim-user is specified and we're just loading tools, don't require MODEL_AGENT
    elif args.no_sim_user and not args.model:
        # Just print a warning but continue
        print("Warning: No model specified with --model or MODEL_AGENT environment variable.")
        print("Since --no-sim-user is specified, we'll just verify the tool setup without agent evaluation.")
    # Check environment variables
    elif not check_agent_environment(test_mode=False):
        return 1
    
    # Get toolset configuration
    toolset_config = get_toolset_config(args, is_task_dir)
    if toolset_config is None:
        return 1
    
    # Load tasks from dataset
    try:
        tasks = load_task_from_file(dataset_path)
        if not tasks:
            print(f"Error: No tasks found in dataset file '{dataset_path}'")
            return 1
        
        # Filter tasks if task-ids is specified
        if args.task_ids:
            requested_ids = args.task_ids.split(",")
            tasks = [t for t in tasks if t.get("id") in requested_ids]
            
            if not tasks:
                print(f"Error: No tasks found with IDs: {args.task_ids}")
                return 1
        
        # Limit tasks if max-tasks is specified
        if args.max_tasks and args.max_tasks > 0:
            tasks = tasks[:args.max_tasks]
        
        print(f"Loaded {len(tasks)} tasks from {dataset_path}")
        
        # Process each task
        successes = 0
        failures = 0
        
        for i, task in enumerate(tasks):
            print(f"\nProcessing task {i+1}/{len(tasks)}: {task.get('id', f'task_{i}')}")
            
            # Extract task details
            task_id = task.get("id", f"task_{uuid.uuid4()}")
            toolset = task.get("toolset")
            
            if not toolset:
                print(f"Error: Task {task_id} has no toolset defined")
                failures += 1
                continue
            
            # Apply registry override if specified
            if toolset_config.get("registry_override"):
                logger.info(f"Overriding toolset '{toolset}' with '{toolset_config['registry_override']}'")
                toolset = toolset_config["registry_override"]
            
            # Extract reward module path from toolset path
            reward_path = ".".join(toolset.split(".")[:-1] + ["reward"])
            
            # Apply custom evaluator if specified
            if toolset_config.get("evaluator"):
                logger.info(f"Using custom evaluator: {toolset_config['evaluator']}")
                reward_path = toolset_config["evaluator"]
            
            # Check for seed SQL
            seed_sql = task.get("seed_sql")
            seed_file = None
            
            if seed_sql and seed_sql.startswith("file:"):
                # If seed_sql is a file reference, load it
                seed_file_relative = seed_sql[5:]  # Remove "file:" prefix
                if is_task_dir:
                    # If using task-dir, find relative to the task directory
                    seed_file = os.path.join(args.task_dir, seed_file_relative)
                else:
                    # Otherwise, find relative to the dataset file
                    seed_file = os.path.join(os.path.dirname(dataset_path), seed_file_relative)
                seed_sql = None
            
            # Create evaluator
            try:
                # This needs to be async, so we'll use asyncio
                async def run_evaluation():
                    evaluator = AgentEvaluator(
                        task_id=task_id,
                        toolset_path=toolset,
                        reward_path=reward_path,
                        base_dir=args.output_dir,
                        seed_sql=seed_sql,
                        seed_file=seed_file
                    )
                    
                    # Set up the evaluator
                    await evaluator.setup()
                    
                    # Create a run
                    run_id = f"run_{uuid.uuid4().hex[:8]}"
                    run_db_path = await evaluator.create_run(run_id)
                    
                    logger.info(f"Created evaluation run at {run_db_path}")
                    
                    # Get the tools for this task
                    tools_spec = evaluator.tool_registry.get_openai_tools()
                    print(f"Available tools ({len(tools_spec)}):")
                    for tool in tools_spec:
                        print(f"  - {tool['function']['name']}: {tool['function']['description']}")
                    
                    # Export tool specs if requested
                    if args.export_tools:
                        export_dir = os.path.join(args.export_tools, task_id)
                        spec_file = export_tool_specs(tools_spec, export_dir)
                        print(f"Exported tool specifications to {spec_file}")
                    
                    # Get the initial messages
                    messages = task.get("initial_messages", [])
                    
                    # Determine which model to use
                    model = args.model or os.environ.get("MODEL_AGENT")
                    
                    # If test mode is specified, or --no-sim-user is specified without a model,
                    # we can use a mock response or just verify tool setup
                    if args.test_mode or (args.no_sim_user and not model):
                        if args.test_mode:
                            print("Running in test mode without real model API calls.")
                        else:
                            print("Skipping agent evaluation since no model is specified and --no-sim-user is set.")
                        
                        if args.mock_response:
                            print("Using mock agent response for testing...")
                            # Create a mock response that will trigger tool execution
                            mock_message = {
                                "role": "assistant", 
                                "content": "I'll help you with that. Let me use the available tools."
                            }
                            messages.append(mock_message)
                            
                            # Get the first tool to execute as an example
                            if tools_spec:
                                first_tool = tools_spec[0]["function"]["name"]
                                print(f"Simulating execution of tool: {first_tool}")
                                
                                # Execute the tool with empty parameters (this might fail depending on tool requirements)
                                try:
                                    tool_result = await evaluator.execute_tool(run_id, first_tool, {})
                                    print(f"Tool execution result: {tool_result}")
                                except Exception as e:
                                    print(f"Mock tool execution failed (expected in test mode): {str(e)}")
                        else:
                            print("The tools were successfully loaded and setup.")
                        
                        # Run evaluation even in test mode if requested
                        if args.test_mode and not args.no_sim_user:
                            print("Running evaluation with mock data...")
                            end_goal_sql = task.get("end_goal_sql")
                            eval_kwargs = {"end_goal_sql": end_goal_sql} if end_goal_sql else {}
                            
                            # Add any additional parameters from the task
                            for key, value in task.items():
                                if key not in ["id", "toolset", "initial_messages", "seed_sql", "end_goal_sql", "sim_user_prompt", "n_rollouts"]:
                                    eval_kwargs[key] = value
                            
                            try:
                                evaluation = await evaluator.evaluate(
                                    run_id=run_id,
                                    messages=messages,
                                    **eval_kwargs
                                )
                                print(f"Evaluation result: {evaluation}")
                            except Exception as e:
                                print(f"Evaluation failed (expected in test mode): {str(e)}")
                        
                        return True
                    
                    if not model:
                        print("Error: No agent model specified. Use --model or set MODEL_AGENT")
                        return False
                        
                    print(f"Running evaluation with model: {model}")
                    
                    # Check if the model is from OpenAI or Anthropic
                    provider = model.split("/")[0] if "/" in model else "openai"
                    
                    if provider == "openai":
                        # Use OpenAI API
                        import openai
                        try:
                            # Handle different versions of the OpenAI client with proper error handling
                            try:
                                # Get the API key from environment
                                api_key = os.environ.get("OPENAI_API_KEY")
                                
                                # For OpenAI v1.x
                                client = openai.OpenAI(api_key=api_key)
                            except Exception as e:
                                # Log the error for debugging
                                if args.debug:
                                    print(f"OpenAI client creation failed: {str(e)}")
                                
                                try:
                                    # Direct client creation as fallback
                                    from openai import OpenAI
                                    client = OpenAI(api_key=api_key)
                                except Exception as e2:
                                    if args.debug:
                                        print(f"Fallback initialization failed: {str(e2)}")
                                    raise ValueError(f"Could not initialize OpenAI client: {str(e)}")
                        except Exception as e:
                            print(f"Error initializing OpenAI client: {str(e)}")
                            raise
                        
                        # Run the conversation
                        print("Starting agent conversation...")
                        try:
                            # Extract model name from the provider/model format
                            model_name = model.split("/")[1] if "/" in model else model
                            print(f"Using OpenAI model: {model_name}")
                            
                            response = client.chat.completions.create(
                                model=model_name,
                                messages=messages,
                                tools=tools_spec,
                                tool_choice="auto"
                            )
                            
                            # Get the response
                            assistant_message = response.choices[0].message
                            
                            # Handle content that might be None with tool calls
                            assistant_content = assistant_message.content or ""
                            messages.append({"role": "assistant", "content": assistant_content})
                            
                            # Handle tool calls
                            if hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls:
                                for tool_call in assistant_message.tool_calls:
                                    # Execute the tool
                                    tool_name = tool_call.function.name
                                    tool_params = json.loads(tool_call.function.arguments)
                                    
                                    print(f"Executing tool: {tool_name}")
                                    tool_result = await evaluator.execute_tool(run_id, tool_name, tool_params)
                                    
                                    # Convert result to string if it's not already
                                    tool_result_str = json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
                                    
                                    # Add the tool result to the conversation
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "name": tool_name,
                                        "content": tool_result_str
                                    })
                            
                            # Evaluate the result
                            end_goal_sql = task.get("end_goal_sql")
                            eval_kwargs = {"end_goal_sql": end_goal_sql} if end_goal_sql else {}
                            
                            # Add any additional parameters from the task
                            for key, value in task.items():
                                if key not in ["id", "toolset", "initial_messages", "seed_sql", "end_goal_sql", "sim_user_prompt", "n_rollouts"]:
                                    eval_kwargs[key] = value
                            
                            evaluation = await evaluator.evaluate(
                                run_id=run_id,
                                messages=messages,
                                **eval_kwargs
                            )
                            
                            print(f"Evaluation result: {evaluation}")
                            return True
                            
                        except Exception as e:
                            print(f"Error during agent evaluation: {str(e)}")
                            if args.debug:
                                traceback.print_exc()
                            return False
                            
                    elif provider == "anthropic":
                        # Use Anthropic API
                        import anthropic
                        
                        try:
                            # Create client with proper error handling
                            try:
                                # Get the API key from environment
                                api_key = os.environ.get("ANTHROPIC_API_KEY")
                                
                                # For Anthropic latest client
                                if api_key:
                                    client = anthropic.Anthropic(api_key=api_key)
                                else:
                                    client = anthropic.Anthropic() # Use default API key from environment
                                    
                            except Exception as e:
                                if args.debug:
                                    print(f"Error with Anthropic initialization: {str(e)}")
                                raise ValueError(f"Could not initialize Anthropic client: {str(e)}")
                        except Exception as e:
                            print(f"Error initializing Anthropic client: {str(e)}")
                            raise
                        
                        # Convert messages to Anthropic format if needed
                        anthropic_messages = []
                        for msg in messages:
                            if msg["role"] == "user":
                                anthropic_messages.append({"role": "user", "content": msg["content"]})
                            elif msg["role"] == "assistant":
                                anthropic_messages.append({"role": "assistant", "content": msg["content"]})
                            # Skip system messages if Anthropic client doesn't support them
                        
                        print("Starting agent conversation...")
                        try:
                            # Extract model name from the provider/model format
                            model_name = model.split("/")[1] if "/" in model else model
                            print(f"Using Anthropic model: {model_name}")
                            
                            # Updated implementation with better error handling
                            response = client.messages.create(
                                model=model_name,
                                messages=anthropic_messages,
                                tools=tools_spec
                            )
                            
                            # Process response and tool calls similar to OpenAI
                            # TODO: Update this for Anthropic's response format
                            
                            # For now, just add the response message
                            messages.append({
                                "role": "assistant",
                                "content": response.content[0].text
                            })
                            
                            # Evaluate the result
                            end_goal_sql = task.get("end_goal_sql")
                            eval_kwargs = {"end_goal_sql": end_goal_sql} if end_goal_sql else {}
                            
                            # Add any additional parameters from the task
                            for key, value in task.items():
                                if key not in ["id", "toolset", "initial_messages", "seed_sql", "end_goal_sql", "sim_user_prompt", "n_rollouts"]:
                                    eval_kwargs[key] = value
                            
                            evaluation = await evaluator.evaluate(
                                run_id=run_id,
                                messages=messages,
                                **eval_kwargs
                            )
                            
                            print(f"Evaluation result: {evaluation}")
                            return True
                            
                        except Exception as e:
                            print(f"Error during agent evaluation: {str(e)}")
                            if args.debug:
                                traceback.print_exc()
                            return False
                    
                    else:
                        print(f"Unsupported model provider: {provider}")
                        print("Only 'openai' and 'anthropic' are supported")
                        return False
                
                # Run the async function
                result = asyncio.run(run_evaluation())
                
                if result:
                    successes += 1
                else:
                    failures += 1
                    
            except Exception as e:
                print(f"Error setting up evaluator for task {task_id}: {str(e)}")
                if args.verbose or args.debug:
                    traceback.print_exc()
                failures += 1
        
        # Print summary
        print(f"\nEvaluation complete: {successes} tasks succeeded, {failures} tasks failed")
        return 0 if failures == 0 else 1
        
    except Exception as e:
        print(f"Error running agent evaluation: {str(e)}")
        if args.verbose or args.debug:
            traceback.print_exc()
        return 1

def parse_args(args=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="reward-kit: Tools for evaluation and reward modeling"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Preview command
    preview_parser = subparsers.add_parser(
        "preview", 
        help="Preview an evaluator with sample data"
    )
    preview_parser.add_argument(
        "--metrics-folders", "-m",
        nargs="+",
        help="Metric folders in format 'name=path', e.g., 'clarity=./metrics/clarity'"
    )
    preview_parser.add_argument(
        "--samples", "-s",
        required=True,
        help="Path to JSONL file containing sample data"
    )
    preview_parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Maximum number of samples to process (default: 5)"
    )
    
    # Deploy command
    deploy_parser = subparsers.add_parser(
        "deploy", 
        help="Create and deploy an evaluator"
    )
    deploy_parser.add_argument(
        "--id",
        required=True,
        help="ID for the evaluator"
    )
    deploy_parser.add_argument(
        "--metrics-folders", "-m",
        nargs="+",
        required=True,
        help="Metric folders in format 'name=path', e.g., 'clarity=./metrics/clarity'"
    )
    deploy_parser.add_argument(
        "--display-name",
        help="Display name for the evaluator (defaults to ID if not provided)"
    )
    deploy_parser.add_argument(
        "--description",
        help="Description for the evaluator"
    )
    deploy_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force update if evaluator already exists"
    )
    
    # Agent-eval command
    agent_eval_parser = subparsers.add_parser(
        "agent-eval", 
        help="Run agent evaluation on a task dataset"
    )
    
    # Task specification (mutually exclusive)
    task_group = agent_eval_parser.add_argument_group("Task Specification")
    task_group.add_argument(
        "--task-dir",
        help="Path to task bundle directory containing reward.py, tools.py, etc."
    )
    task_group.add_argument(
        "--dataset", "-d",
        help="Path to JSONL file containing task dataset"
    )
    
    # Output and models
    output_group = agent_eval_parser.add_argument_group("Output and Models")
    output_group.add_argument(
        "--output-dir", "-o",
        default="./runs",
        help="Directory to store evaluation runs (default: ./runs)"
    )
    output_group.add_argument(
        "--model",
        help="Override MODEL_AGENT environment variable"
    )
    output_group.add_argument(
        "--sim-model",
        help="Override MODEL_SIM environment variable for simulated user"
    )
    
    # Test and debug options
    debug_group = agent_eval_parser.add_argument_group("Testing and Debugging")
    debug_group.add_argument(
        "--no-sim-user",
        action="store_true",
        help="Disable simulated user (use static initial messages only)"
    )
    debug_group.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode without requiring API keys"
    )
    debug_group.add_argument(
        "--mock-response",
        action="store_true",
        help="Use a mock agent response (works with --test-mode)"
    )
    debug_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug logging"
    )
    debug_group.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate task bundle structure without running evaluation"
    )
    debug_group.add_argument(
        "--export-tools",
        metavar="DIR",
        help="Export tool specifications to directory for manual testing"
    )
    
    # Advanced options
    advanced_group = agent_eval_parser.add_argument_group("Advanced Options")
    advanced_group.add_argument(
        "--task-ids",
        help="Comma-separated list of task IDs to run"
    )
    advanced_group.add_argument(
        "--max-tasks",
        type=int,
        help="Maximum number of tasks to evaluate"
    )
    advanced_group.add_argument(
        "--registries",
        nargs="+",
        help="Custom tool registries in format 'name=path'"
    )
    advanced_group.add_argument(
        "--registry-override",
        help="Override all toolset paths with this registry path"
    )
    advanced_group.add_argument(
        "--evaluator",
        help="Custom evaluator module path (overrides default)"
    )
    
    return parser.parse_args(args)

def main():
    """Main entry point for the CLI"""
    args = parse_args()
    setup_logging(args.verbose, getattr(args, "debug", False))
    
    if args.command == "preview":
        return preview_command(args)
    elif args.command == "deploy":
        return deploy_command(args)
    elif args.command == "agent-eval":
        return agent_eval_command(args)
    else:
        # No command provided, show help
        parser = argparse.ArgumentParser()
        parser.print_help()
        return 0

if __name__ == "__main__":
    sys.exit(main())