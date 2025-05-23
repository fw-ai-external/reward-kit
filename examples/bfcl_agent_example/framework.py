"""
BFCL Framework - Core components for BFCL agent evaluation

This module provides the foundational classes and utilities that developers can
use to create and run BFCL agent evaluations without duplicating code.
"""

import asyncio
import copy
import importlib
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class BFCLResource:
    """
    Base resource manager for BFCL environments.

    This class handles:
    - Loading environment classes
    - Managing environment state
    - Serializing/deserializing state
    - Executing tool calls
    - Generating tool specifications

    Developers should not need to modify this class in most cases.
    """

    def __init__(self, package_name: Optional[str] = None):  # Changed type hint
        """
        Initialize a BFCL resource manager.

        Args:
            package_name: Optional package name for imports (defaults to current package)
        """
        self._env_instances: Dict[str, Any] = {}
        self._initial_config: Dict[str, Any] = {}
        self._package_name = package_name or ".".join(__name__.split(".")[:-1])
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def setup(self, config: Dict[str, Any]) -> None:
        """
        Initialize the resource with a given configuration.

        Args:
            config: Configuration dictionary with involved_classes and initial_config
        """
        self._initial_config = copy.deepcopy(config)
        involved_classes = config.get("involved_classes", [])
        initial_config_data = config.get("initial_config", {})

        # Determine class mappings
        class_mappings = self._get_class_mappings()
        stateless_classes = self._get_stateless_classes()

        for class_name in involved_classes:
            if class_name not in self._env_instances:
                if class_name not in class_mappings:
                    self.logger.error(f"Unknown class: {class_name}")
                    continue

                module_name = class_mappings[class_name]
                try:
                    # Import from package if provided, otherwise try absolute import
                    if self._package_name:
                        module = importlib.import_module(
                            f"{self._package_name}.{module_name}"
                        )
                    else:
                        module = importlib.import_module(module_name)

                    class_ = getattr(module, class_name)
                    instance = class_()

                    # Initialize with config if not stateless
                    if class_name not in stateless_classes and initial_config_data.get(
                        class_name
                    ):
                        instance._load_scenario(
                            copy.deepcopy(initial_config_data[class_name])
                        )

                    self._env_instances[class_name] = instance
                    self.logger.info(f"Loaded environment class: {class_name}")
                except Exception as e:
                    self.logger.error(f"Error loading {class_name}: {e}")

    def _get_class_mappings(self) -> Dict[str, str]:
        """
        Get mappings from class names to module paths.

        Override this method in subclasses to provide custom mappings.

        Returns:
            Dictionary mapping class names to module paths
        """
        return {
            "GorillaFileSystem": "envs.gorilla_file_system",
            "TwitterAPI": "envs.posting_api",
            "MathAPI": "envs.math_api",
            "MessageAPI": "envs.message_api",
            "TicketAPI": "envs.ticket_api",
            "TradingBot": "envs.trading_bot",
            "TravelAPI": "envs.travel_booking",
            "VehicleControlAPI": "envs.vehicle_control",
            # Add other environment classes here
        }

    def _get_stateless_classes(self) -> List[str]:
        """
        Get names of stateless environment classes.

        Override this method in subclasses to specify stateless classes.

        Returns:
            List of class names that are stateless
        """
        return []

    def fork(self) -> "BFCLResource":
        """
        Create an independent copy of this resource with the same state.

        Returns:
            A new BFCLResource instance with copied state
        """
        new_resource = self.__class__(self._package_name)
        new_resource._env_instances = copy.deepcopy(self._env_instances)
        new_resource._initial_config = copy.deepcopy(self._initial_config)
        return new_resource

    def reset(self) -> None:
        """Reset to initial state by reloading from initial config."""
        self._env_instances.clear()
        self.setup(self._initial_config)

    def get_tools_spec(self) -> List[Dict[str, Any]]:
        """
        Get OpenAPI-compatible tool specifications for all environment methods.

        Returns:
            List of tool specifications
        """
        tool_specs = []
        for instance in self._env_instances.values():
            # Inspect methods of the instance
            for name, method in inspect.getmembers(
                instance, predicate=inspect.ismethod
            ):
                if name.startswith("_"):  # Skip private methods
                    continue

                # Infer schema from method signature
                try:
                    schema = self._infer_schema_from_method(method)
                    tool_specs.append(schema)
                except Exception as e:
                    self.logger.warning(f"Could not infer schema for {name}: {e}")

        return tool_specs

    def step(self, action_name: str, action_params: Dict[str, Any]) -> Any:
        """
        Execute a tool action with the given parameters.

        Args:
            action_name: Name of the tool/method to call
            action_params: Dictionary of parameters for the tool

        Returns:
            Result from the tool execution
        """
        # Find the correct environment instance and call the method
        for instance in self._env_instances.values():
            if hasattr(instance, action_name):
                try:
                    # Convert tuple to list if needed
                    for key, value in action_params.items():
                        if isinstance(value, tuple):
                            action_params[key] = list(value)

                    # Execute the method
                    result = getattr(instance, action_name)(**action_params)
                    return result
                except Exception as e:
                    self.logger.error(f"Error executing tool {action_name}: {e}")
                    return {"error": f"Error executing tool {action_name}: {e}"}

        return {"error": f"Tool {action_name} not found in available resources"}

    def get_comparable_state(self) -> Dict[str, Any]:
        """
        Get a serializable representation of the current state.

        Returns:
            Dictionary containing serialized state
        """
        state = {}
        for class_name, instance in self._env_instances.items():
            state[class_name] = self._serialize_instance(instance)
        return state

    def _serialize_instance(self, instance: Any) -> Dict[str, Any]:
        """
        Serialize an instance's state for comparison.

        Override this method in subclasses for custom serialization.

        Args:
            instance: Object instance to serialize

        Returns:
            Dictionary containing serialized state
        """
        result = {}

        for attr_name, value in vars(instance).items():
            if attr_name.startswith("_"):
                continue  # Skip private attributes

            try:
                # Try to serialize with JSON
                json.dumps(value)
                result[attr_name] = value
            except (TypeError, OverflowError):
                # Fall back to string representation
                result[attr_name] = str(value)

        return result

    def _infer_schema_from_method(self, method: Any) -> Dict[str, Any]:
        """
        Infer OpenAPI schema from a method signature.

        Args:
            method: Method to analyze

        Returns:
            Dictionary containing OpenAPI schema
        """
        name = method.__name__
        docstring = method.__doc__ or ""

        parameters_value: Dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        schema: Dict[str, Any] = {
            "name": name,
            "description": docstring,
            "parameters": parameters_value,
        }

        # Analyze method signature
        sig = inspect.signature(method)
        type_mapping: Dict[type, str] = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            # Add more type mappings as needed
        }

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # Default to string type
            param_schema: Dict[str, Any] = {"type": "string"}  # type: ignore [no-redef]

            # Try to infer type from annotation
            if param.annotation != inspect.Parameter.empty:
                param_type = param.annotation

                # Handle Union/Optional types
                if hasattr(param_type, "__origin__") and param_type.__origin__ is Union:
                    non_none_types = [
                        t for t in param_type.__args__ if t is not type(None)
                    ]
                    if non_none_types:
                        param_type = non_none_types[0]

                if param_type in type_mapping:
                    param_schema["type"] = type_mapping[param_type]
                elif hasattr(param_type, "__origin__"):
                    # Handle generics like List[str]
                    if param_type.__origin__ in (list, List):
                        param_schema = {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                        # Try to infer item type
                        if hasattr(param_type, "__args__") and param_type.__args__:
                            item_type = param_type.__args__[0]
                            if item_type in type_mapping:
                                param_schema["items"]["type"] = type_mapping[item_type]

            schema["parameters"]["properties"][param_name] = param_schema

            # Add to required list if no default value
            if param.default is inspect.Parameter.empty:
                schema["parameters"]["required"].append(param_name)

        return schema


class BFCLOrchestrator:
    """
    Orchestrator for executing BFCL agent evaluations.

    This class handles:
    - Loading task configurations
    - Managing the agent-environment interaction
    - Collecting and evaluating results

    Developers should not need to modify this class in most cases.
    """

    def __init__(
        self,
        task_config: Union[Dict[str, Any], str, Path],
        resource_class: Optional[type] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            task_config: Task configuration (dict or path to YAML file)
            resource_class: Optional custom resource class (defaults to BFCLResource)
        """
        # Load task config if string or Path
        if isinstance(task_config, (str, Path)):
            with open(task_config, "r") as f:
                self.task_config = yaml.safe_load(f)
        else:
            self.task_config = task_config

        self.name = self.task_config.get("name", "unnamed_task")
        self.logger = logging.getLogger(f"BFCLOrchestrator.{self.name}")

        # Create resource instance
        resource_class_val = (
            resource_class or BFCLResource
        )  # Ensure type for self.resource
        self.resource: BFCLResource = resource_class_val()

        # Initialize state
        self.conversation_history: List[Dict[str, Any]] = []
        self.tool_calls_history: List[Dict[str, Any]] = []
        self.current_states: List[Dict[str, Any]] = []
        self.model: str = os.environ.get("MODEL_AGENT", "openai/gpt-4")

    async def setup(self) -> bool:
        """
        Set up the resource and prepare for execution.

        Returns:
            True if setup succeeded, False otherwise
        """
        try:
            self.resource.setup(self.task_config.get("base_resource_config", {}))
            self.available_tools = self.resource.get_tools_spec()
            self.logger.info(f"Setup complete for task: {self.name}")
            self.logger.info(f"Found {len(self.available_tools)} available tools")
            return True
        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            return False

    async def run_evaluation(self) -> Dict[str, Any]:
        """
        Run the evaluation task and return results.

        Returns:
            Dictionary with evaluation results
        """
        self.logger.info(f"Starting evaluation for task: {self.name}")

        try:
            # Setup resource
            if not await self.setup():
                return {"error": "Failed to set up resource"}

            # Get messages from task config
            messages = self.task_config.get("messages", [])
            if not messages:
                return {"error": "No messages found in task configuration"}

            # Process turns
            for i, message in enumerate(messages):
                self.logger.info(f"Processing turn {i+1}/{len(messages)}")

                # Process user message
                content = self._parse_message_content(message.get("content", ""))
                if not content:
                    continue

                user_message = {"role": "user", "content": content}
                self.conversation_history.append(user_message)

                # Process agent response and tool calls
                await self._process_agent_turn()

                # Save state after turn
                current_state = self.resource.get_comparable_state()
                self.current_states.append(current_state)

            # Evaluate results
            return self._evaluate_results()

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}", exc_info=True)
            return {"error": str(e)}

    def _parse_message_content(self, content: Any) -> str:
        """
        Parse message content handling various formats.

        Args:
            content: Raw message content

        Returns:
            Parsed content as string
        """
        if not content:
            return ""

        # Handle BFCL's JSON string format
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list) and parsed:
                    # Use first message content if it's a list
                    if isinstance(parsed[0], dict) and "content" in parsed[0]:
                        return parsed[0]["content"]
            except (json.JSONDecodeError, TypeError):
                # Use content as is if not valid JSON
                pass

            return content

        # Handle direct dict or other formats
        if isinstance(content, dict) and "content" in content:
            return content["content"]

        # Fall back to string representation
        return str(content)

    async def _process_agent_turn(self) -> None:
        """Process a single turn of agent interaction."""
        # Parse model string
        provider, model_name = self._parse_model_string(self.model)

        if provider == "openai":
            # Process with OpenAI
            try:
                response = await self._run_openai_agent(model_name)

                if "error" in response:
                    self.logger.error(f"OpenAI API error: {response['error']}")
                    self.conversation_history.append(
                        {"role": "assistant", "content": f"Error: {response['error']}"}
                    )
                    return

                # Get assistant message
                assistant_message = response.choices[0].message
                self.conversation_history.append(dict(assistant_message))

                # Process tool calls
                if (
                    hasattr(assistant_message, "tool_calls")
                    and assistant_message.tool_calls
                ):
                    await self._process_tool_calls(assistant_message.tool_calls)
            except Exception as e:
                self.logger.error(f"Error processing agent turn: {e}")
                self.conversation_history.append(
                    {"role": "assistant", "content": f"Error: {e}"}
                )
        else:
            self.logger.error(f"Unsupported provider: {provider}")
            self.conversation_history.append(
                {
                    "role": "assistant",
                    "content": f"Error: Unsupported provider {provider}",
                }
            )

    async def _run_openai_agent(self, model_name: str) -> Any:
        """
        Run OpenAI agent with current conversation history.

        Args:
            model_name: OpenAI model name to use

        Returns:
            OpenAI API response
        """
        try:
            import openai

            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

            # Prepare tools in OpenAI format
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                }
                for tool in self.available_tools
            ]

            self.logger.debug(
                f"Calling OpenAI: model={model_name}, messages={self.conversation_history}"
            )

            # Call OpenAI API
            response = client.chat.completions.create(
                model=model_name,
                messages=self.conversation_history,
                tools=tools,
                tool_choice="auto",
                temperature=0,
            )

            return response
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return {"error": str(e)}

    async def _process_tool_calls(self, tool_calls: List[Any]) -> None:
        """
        Process tool calls from agent response.

        Args:
            tool_calls: List of tool calls from OpenAI response
        """
        for tool_call in tool_calls:
            try:
                # Extract function name and arguments
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                self.logger.info(
                    f"Processing tool call: {function_name}({function_args})"
                )

                # Execute tool call
                result = self.resource.step(function_name, function_args)

                # Record tool call
                self.tool_calls_history.append(
                    {"name": function_name, "args": function_args, "result": result}
                )

                # Add tool result to conversation
                tool_result_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(result),
                }

                self.conversation_history.append(tool_result_message)

                self.logger.info(f"Tool call result: {result}")
            except Exception as e:
                self.logger.error(f"Error processing tool call: {e}")

                # Add error to conversation
                tool_id = getattr(tool_call, "id", "unknown")
                tool_name = (
                    tool_call.function.name
                    if hasattr(tool_call, "function")
                    and hasattr(tool_call.function, "name")
                    else "unknown"
                )

                error_message = {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": json.dumps({"error": str(e)}),
                }

                self.conversation_history.append(error_message)

    def _evaluate_results(self) -> Dict[str, Any]:
        """
        Evaluate task results against ground truth.

        Override this method in subclasses for custom evaluation.

        Returns:
            Dictionary with evaluation results
        """
        # Get evaluation criteria
        evaluation_criteria = self.task_config.get("evaluation_criteria", {})

        # Check if ground truth data exists
        ground_truth_function_calls = evaluation_criteria.get(
            "ground_truth_function_calls", []
        )
        if not ground_truth_function_calls:
            return {"score": 0, "reason": "No ground truth function calls found"}

        # Calculate function call score
        function_call_score = self._calculate_function_call_score(
            ground_truth_function_calls
        )

        # Get ground truth state if available
        ground_truth_state = evaluation_criteria.get(
            "ground_truth_comparable_state", {}
        )
        state_match = False

        if ground_truth_state and self.current_states:
            # Compare final state with ground truth
            final_state = self.current_states[-1]
            state_match = self._compare_states(final_state, ground_truth_state)

        # Calculate overall score
        state_score = 1.0 if state_match else 0.0
        overall_score = (function_call_score + state_score) / 2.0

        # Prepare result
        return {
            "score": overall_score,
            "function_call_score": function_call_score,
            "state_match": state_match,
            "reason": self._generate_evaluation_reason(
                function_call_score, state_match
            ),
        }

    def _calculate_function_call_score(
        self, ground_truth_calls: List[List[str]]
    ) -> float:
        """
        Calculate how well agent's tool calls match ground truth.

        Args:
            ground_truth_calls: List of lists of ground truth function calls

        Returns:
            Score between 0.0 and 1.0
        """
        if not ground_truth_calls or not self.tool_calls_history:
            return 0.0

        # Flatten ground truth calls
        flat_gt_calls = []
        for turn_calls in ground_truth_calls:
            flat_gt_calls.extend(turn_calls)

        # Convert agent calls to comparable format
        agent_calls = [
            f"{call['name']}({self._format_args(call['args'])})"
            for call in self.tool_calls_history
        ]

        # Count matches (simplified)
        matches = 0
        for gt_call in flat_gt_calls:
            if any(gt_call == agent_call for agent_call in agent_calls):
                matches += 1

        # Calculate score
        return matches / len(flat_gt_calls)

    def _format_args(self, args: Dict[str, Any]) -> str:
        """
        Format function arguments for comparison.

        Args:
            args: Dictionary of function arguments

        Returns:
            Formatted arguments string
        """
        parts = []
        for key, value in sorted(args.items()):
            if isinstance(value, str):
                parts.append(f"{key}='{value}'")
            else:
                parts.append(f"{key}={value}")
        return ", ".join(parts)

    def _compare_states(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> bool:
        """
        Compare two states for equality.

        Override this method in subclasses for custom state comparison.

        Args:
            state1: First state
            state2: Second state

        Returns:
            True if states match, False otherwise
        """
        return state1 == state2

    def _generate_evaluation_reason(
        self, function_call_score: float, state_match: bool
    ) -> str:
        """
        Generate human-readable reason for evaluation scores.

        Override this method in subclasses for custom reasons.

        Args:
            function_call_score: Score for function call matches
            state_match: Whether final state matches ground truth

        Returns:
            Human-readable explanation
        """
        reasons = []

        if function_call_score == 1.0:
            reasons.append("All required function calls were correctly executed.")
        elif function_call_score >= 0.75:
            reasons.append("Most required function calls were correctly executed.")
        elif function_call_score >= 0.5:
            reasons.append("Some required function calls were correctly executed.")
        elif function_call_score > 0:
            reasons.append("Few required function calls were correctly executed.")
        else:
            reasons.append("No required function calls were correctly executed.")

        if state_match:
            reasons.append("Final state matches the expected state.")
        else:
            reasons.append("Final state does not match the expected state.")

        return " ".join(reasons)

    def _parse_model_string(self, model_string: str) -> Tuple[str, str]:
        """
        Parse model string in format 'provider/model_name'.

        Args:
            model_string: Model string to parse

        Returns:
            Tuple of (provider, model_name)
        """
        if "/" in model_string:
            provider, model_name = model_string.split("/", 1)
            return provider.lower(), model_name
        else:
            # Default to OpenAI if no provider specified
            return "openai", model_string


async def run_bfcl_task(
    task_path: Union[str, Path],
    model: Optional[str] = None,
    resource_class: Optional[type] = None,
    orchestrator_class: Optional[type] = None,
) -> Dict[str, Any]:
    """
    Run a BFCL task from a YAML file.

    This is the main entry point for running BFCL evaluations.

    Args:
        task_path: Path to task YAML file
        model: Optional model override (format: provider/model_name)
        resource_class: Optional custom resource class
        orchestrator_class: Optional custom orchestrator class

    Returns:
        Dictionary with evaluation results
    """
    logger = logging.getLogger("bfcl_runner")

    # Convert to Path if string
    if isinstance(task_path, str):
        task_path = Path(task_path)

    # Check if task file exists
    if not task_path.exists():
        logger.error(f"Task file not found: {task_path}")
        return {"error": f"Task file not found: {task_path}"}

    # Set model if provided
    original_model: Optional[str] = None
    if model:
        logger.info(f"Using model: {model}")
        original_model = os.environ.get("MODEL_AGENT")
        os.environ["MODEL_AGENT"] = model

    # Check for OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        logger.warning("OPENAI_API_KEY not set, some providers may not work")

    # Default orchestrator class
    if orchestrator_class is None:
        orchestrator_class = BFCLOrchestrator

    # Create orchestrator and run evaluation
    try:
        orchestrator = orchestrator_class(task_path, resource_class)
        results = await orchestrator.run_evaluation()

        # Restore original model if changed
        if model and original_model:
            os.environ["MODEL_AGENT"] = original_model

        return results
    except Exception as e:
        logger.error(f"Error running task: {e}", exc_info=True)

        # Restore original model if changed
        if model and original_model:
            os.environ["MODEL_AGENT"] = original_model

        return {"error": str(e)}
