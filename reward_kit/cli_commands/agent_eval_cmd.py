"""
CLI command for running agent evaluations using the ForkableResource framework.
"""
import asyncio
try:
    import yaml
except ImportError:
    yaml = None # PyYAML not available

import json # Fallback or for explicit JSON files
from pathlib import Path
import logging # For logger instance
from pydantic import ValidationError

from reward_kit.agent import Orchestrator
from reward_kit.models import TaskDefinitionModel # Import the new Pydantic model
# setup_logging is already called in cli.py's main, but good for standalone use if any
# from .common import setup_logging 

def agent_eval_command(args):
    """
    Run agent evaluation using the Orchestrator and ForkableResource framework.
    """
    logger = logging.getLogger("agent_eval") # Use a specific logger
    logger.info("Starting agent-eval command.")

    if not args.task_def:
        logger.error("Error: --task-def (path to task definition YAML file) is required.")
        return 1

    task_def_path = Path(args.task_def)
    if not task_def_path.exists() or not task_def_path.is_file():
        logger.error(f"Error: Task definition file not found or is not a file: {task_def_path}")
        return 1

    task_definition = None
    try:
        with open(task_def_path, 'r') as f:
            if yaml:
                try:
                    task_definition = yaml.safe_load(f)
                    logger.info(f"Successfully loaded task definition from YAML: {task_def_path}")
                except yaml.YAMLError as e_yaml:
                    logger.error(f"Error parsing YAML file {task_def_path}: {e_yaml}")
                    logger.info("Attempting to parse as JSON as a fallback...")
                    f.seek(0) # Reset file pointer
                    try:
                        task_definition = json.load(f)
                        logger.info(f"Successfully loaded task definition as JSON (fallback): {task_def_path}")
                    except json.JSONDecodeError as e_json:
                        logger.error(f"Error parsing file as JSON (fallback attempt for {task_def_path}): {e_json}")
                        return 1
            else: # PyYAML not available, try to parse as JSON directly
                logger.warning("PyYAML not installed. Attempting to parse task definition as JSON.")
                try:
                    task_definition = json.load(f)
                    logger.info(f"Successfully loaded task definition from JSON (PyYAML not found): {task_def_path}")
                except json.JSONDecodeError as e_json:
                    logger.error(f"Error: Could not parse task definition file {task_def_path} as JSON. PyYAML is recommended for YAML files.")
                    return 1
                
    except Exception as e:
        logger.error(f"Error reading task definition file {task_def_path}: {e}")
        return 1

    if task_definition is None: # Should have been caught by earlier returns, but defensive
        logger.error("Task definition could not be loaded.")
        return 1

    # Validate the loaded dictionary against the Pydantic model
    try:
        validated_task_def = TaskDefinitionModel.model_validate(task_definition)
        logger.info(f"Task definition validated successfully: {validated_task_def.name}")
    except ValidationError as e_val:
        logger.error(f"Invalid task definition file structure in {task_def_path}:")
        for error in e_val.errors():
            logger.error(f"  - {'.'.join(map(str,error['loc'])) if error['loc'] else 'root'}: {error['msg']}")
        return 1
    except Exception as e_other_val: # Catch any other validation related errors
        logger.error(f"Unexpected error during task definition validation: {e_other_val}")
        return 1


    # Instantiate Orchestrator with the validated Pydantic model instance
    try:
        orchestrator = Orchestrator(task_definition=validated_task_def)
    except Exception as e:
        logger.error(f"Error instantiating Orchestrator: {e}")
        return 1
        
    # Run the Orchestrator's PoC lifecycle
    try:
        # asyncio.run() is the entry point for async code from sync CLI
        async def main_flow():
            # setup_base_resource is called within execute_task_poc if needed,
            # or can be called separately if explicit setup control is desired before poc.
            # For simplicity here, let execute_task_poc handle its own setup.
            # await orchestrator.setup_base_resource() # This is optional if execute_task_poc handles it
            result = await orchestrator.execute_task_poc() 
            # Handle or log the result if necessary
            if result is None:
                logger.warning("Orchestrator's execute_task_poc returned None, indicating a possible issue during execution.")
            else:
                logger.info(f"Orchestrator PoC execution result: {result}")
        
        asyncio.run(main_flow())
        logger.info("agent-eval command finished successfully.")
        return 0
    except Exception as e:
        logger.error(f"Error during agent-eval execution: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1
