"""
Orchestrator for the Agent Evaluation Framework V2.
Manages the lifecycle of a task using ForkableResources.
"""

import asyncio
import importlib
import inspect
import json  # Add json import
import logging
import os  # Add os import
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type
from unittest.mock import AsyncMock  # Add this import

# Attempt to import OpenAI client
try:
    from openai import AsyncOpenAI, OpenAI
    from openai.types.chat import ChatCompletionMessage, ChatCompletionToolParam
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
    )

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Define dummy types if openai is not installed, to avoid runtime errors on load
    from typing import Any, Dict, List, Optional, Union

    # Use simple class definitions for runtime and type checking
    class OpenAI:
        def __init__(self, **kwargs: Any) -> None:
            pass

    class AsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            pass

    class ChatCompletionMessage:
        content: str = ""
        role: str = "assistant"

    class ChatCompletionToolParam:
        pass

    class ChatCompletionMessageToolCall:
        pass


# Max steps for the inner loop within a single user turn
MAX_STEPS_PER_USER_TURN = 10

from ..models import Message, TaskDefinitionModel  # Import Message model
from .resource_abc import ForkableResource

# Import specific resource types for type checking if needed, or handle dynamically
from .resources import (  # Import BFCLSimAPIResource
    BFCLSimAPIResource,
    DockerResource,
    FileSystemResource,
    PythonStateResource,
    SQLResource,
)


class Orchestrator:
    def __init__(self, task_definition: TaskDefinitionModel):
        self.task_definition = task_definition
        self.base_resource: Optional[ForkableResource] = None
        self.tools_module: Optional[Any] = None
        self.reward_function: Optional[Callable[..., Any]] = None
        self.logger = logging.getLogger(f"Orchestrator.{self.task_definition.name}")
        self.logger.setLevel(logging.DEBUG)  # Ensure debug logs are processed
        self.logger.info(
            f"Orchestrator initialized for task: {self.task_definition.name}"
        )
        self._openai_client: Optional[AsyncOpenAI] = None

    def _initialize_openai_client(self):
        """Initializes the AsyncOpenAI client if available and not already initialized."""
        if not OPENAI_AVAILABLE:
            self.logger.warning(
                "OpenAI library not available. Cannot use OpenAI models."
            )
            return
        if self._openai_client is None:
            # Consider adding error handling for missing API key
            try:
                self._openai_client = AsyncOpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY")
                )
                self.logger.info("AsyncOpenAI client initialized.")
            except Exception as e:
                self.logger.error(f"Failed to initialize AsyncOpenAI client: {e}")
                self._openai_client = None  # Ensure it's None if init fails

    def _load_module_and_function(self, full_path: str) -> Optional[Callable[..., Any]]:
        try:
            module_path, function_name = full_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            func = getattr(module, function_name)
            if not callable(func):
                self.logger.error(
                    f"Loaded attribute '{function_name}' from '{module_path}' is not callable."
                )
                return None
            self.logger.info(
                f"Successfully loaded function '{function_name}' from module '{module_path}'."
            )
            return func
        except (ImportError, AttributeError, ValueError) as e:
            self.logger.error(f"Failed to load function from '{full_path}': {e}")
            return None

    async def _load_task_components(self) -> bool:
        if self.task_definition.tools_module_path:
            try:
                self.tools_module = importlib.import_module(
                    self.task_definition.tools_module_path
                )
                self.logger.info(
                    f"Successfully loaded tools module: {self.task_definition.tools_module_path}"
                )
            except ImportError as e:
                self.logger.error(
                    f"Failed to import tools module '{self.task_definition.tools_module_path}': {e}"
                )
                return False
        else:
            self.logger.info(
                "No 'tools_module_path' specified. Tools may only come from resource.get_tools_spec()."
            )

        if self.task_definition.reward_function_path:
            self.reward_function = self._load_module_and_function(
                self.task_definition.reward_function_path
            )
            if not self.reward_function:
                return False
        else:
            self.logger.error("Reward function path is mandatory but missing.")
            return False
        return True

    def _get_resource_class(self, resource_type_name: str) -> Type[ForkableResource]:
        # This method will now need to look into reward_kit.agent_v2.resources
        # For example: from .resources import SQLResource, PythonStateResource etc.
        # And then map resource_type_name string to the class.
        # For now, a placeholder that would need specific imports or a registry.

        # Option 1: Direct mapping (requires importing all known resource types here)
        # from .resources import PythonStateResource, SQLResource, FileSystemResource, DockerResource # noqa

        mapping = {
            "PythonStateResource": PythonStateResource,
            "SQLResource": SQLResource,
            "FileSystemResource": FileSystemResource,
            "DockerResource": DockerResource,
            "BFCLSimAPIResource": BFCLSimAPIResource,  # Add BFCLSimAPIResource to mapping
        }
        resource_class = mapping.get(resource_type_name)

        if resource_class is None:
            raise ValueError(
                f"Resource class '{resource_type_name}' not found or not mapped in Orchestrator._get_resource_class."
            )
        # No need to check issubclass here if mapping is correct and types are imported.
        return resource_class

    async def setup_base_resource(self) -> None:
        resource_type = self.task_definition.resource_type
        base_config = self.task_definition.base_resource_config

        self.logger.info(
            f"Attempting to set up base resource of type '{resource_type}'..."
        )
        try:
            ResourceClass = self._get_resource_class(resource_type)
            self.base_resource = ResourceClass()
            await self.base_resource.setup(base_config)
            self.logger.info(f"Base resource '{resource_type}' setup complete.")
        except ValueError as e_val:
            self.logger.error(
                f"Could not get resource class '{resource_type}'. {e_val}"
            )
            self.base_resource = None
        except Exception as e_setup:
            self.logger.error(
                f"Failed to setup base resource '{resource_type}'. {e_setup}",
                exc_info=True,
            )
            self.base_resource = None

    async def _get_available_tools(
        self, episode_resource: ForkableResource
    ) -> Dict[str, Callable[..., Any]]:
        available_tools: Dict[str, Callable[..., Any]] = {}
        if episode_resource:
            resource_tool_specs = await episode_resource.get_tools_spec()
            self.logger.debug(
                f"Raw tool specs from resource.get_tools_spec(): {resource_tool_specs}"
            )  # <-- ADDED DEBUG LOG
            for tool_spec in resource_tool_specs:
                # Corrected logic based on BFCLSimAPIResource._infer_schema_from_method output
                tool_name = tool_spec.get("name")  # Get name directly from spec root
                if tool_name:
                    # Create an async adapter function that calls episode_resource.step
                    async def resource_tool_adapter(
                        params: Dict[str, Any],
                        bound_tool_name=tool_name,
                        bound_resource=episode_resource,
                    ):
                        # Ensure params are passed correctly to step
                        return await bound_resource.step(
                            action_name=bound_tool_name, action_params=params
                        )

                    available_tools[tool_name] = resource_tool_adapter
                    self.logger.debug(f"Added tool '{tool_name}' from resource spec.")
                else:
                    self.logger.warning(
                        f"Skipping resource tool spec due to missing 'name': {tool_spec}"
                    )

        # Keep the rest of the function for module tools
        if self.tools_module:
            self.logger.debug(
                f"Inspecting tools_module: {self.tools_module} (type: {type(self.tools_module)})"
            )

            members_to_inspect = []
            if inspect.ismodule(self.tools_module):
                self.logger.debug(
                    f"tools_module is a module. Using inspect.getmembers."
                )
                members_to_inspect = inspect.getmembers(self.tools_module)
            elif hasattr(
                self.tools_module, "__dict__"
            ):  # For class instances like DummyToolsModule
                self.logger.debug(
                    f"tools_module is an object with __dict__. Iterating __dict__.items()."
                )
                members_to_inspect = self.tools_module.__dict__.items()
            else:  # Fallback to getmembers if no __dict__ or not a module
                self.logger.debug(
                    f"tools_module is not a module and has no __dict__. Falling back to inspect.getmembers."
                )
                members_to_inspect = inspect.getmembers(self.tools_module)

            for name, member in members_to_inspect:
                self.logger.debug(
                    f"Found member in tools_module: '{name}', type: {type(member)}, callable: {callable(member)}"
                )
                if name.startswith("_") or not callable(member):
                    self.logger.debug(
                        f"Skipping member '{name}' (startswith_ or not callable)."
                    )
                    continue

                is_tool_async_nature = False
                signature_target = member  # Default to member itself for signature

                # Log type and isinstance check
                # Try isinstance first, then by name as a fallback for potential mock type identity issues
                is_async_mock_instance = isinstance(member, AsyncMock)
                if not is_async_mock_instance and type(member).__name__ == "AsyncMock":
                    self.logger.warning(
                        f"Member '{name}' identified as AsyncMock by name, not isinstance. Type: {type(member)}, Expected AsyncMock type: {type(AsyncMock)}"
                    )
                    is_async_mock_instance = True  # Treat it as an AsyncMock

                self.logger.debug(
                    f"Member '{name}': is_async_mock_instance = {is_async_mock_instance} (after name check), type is {type(member)}"
                )

                if is_async_mock_instance:
                    is_tool_async_nature = True
                    # Try to get the wrapped function for accurate signature inspection
                    if hasattr(member, "func") and member.func is not None:
                        self.logger.debug(
                            f"Member '{name}' is AsyncMock, using wrapped func (from .func) for signature: {member.func}"
                        )
                        signature_target = member.func
                    elif (
                        hasattr(member, "_mock_wraps")
                        and member._mock_wraps is not None
                    ):  # Check _mock_wraps (internal)
                        self.logger.debug(
                            f"Member '{name}' is AsyncMock, using wrapped func (from ._mock_wraps) for signature: {member._mock_wraps}"
                        )
                        signature_target = member._mock_wraps
                    else:
                        self.logger.debug(
                            f"Member '{name}' is AsyncMock, but no 'func' or '_mock_wraps' attribute found. Signature target will be the AsyncMock itself."
                        )
                elif asyncio.iscoroutinefunction(member):
                    self.logger.debug(f"Member '{name}' is coroutine function.")
                    is_tool_async_nature = True
                else:
                    self.logger.debug(
                        f"Member '{name}' is not identified as async tool (not AsyncMock by isinstance/name, not coroutine function)."
                    )

                if is_tool_async_nature:
                    try:
                        sig = inspect.signature(signature_target)
                        resource_param_name = next(
                            (
                                pname
                                for pname in ["resource", "db_resource"]
                                if pname in sig.parameters
                            ),
                            None,
                        )

                        if resource_param_name:

                            async def module_tool_adapter(
                                params: Dict[str, Any],
                                bound_tool_func=member,
                                bound_resource=episode_resource,
                                res_param_name=resource_param_name,
                            ):
                                tool_kwargs = {res_param_name: bound_resource, **params}
                                return await bound_tool_func(**tool_kwargs)

                            available_tools[name] = module_tool_adapter
                            self.logger.debug(
                                f"Added tool '{name}' from tools_module (type: {type(member)}, signature target: {type(signature_target)})."
                            )
                        else:
                            self.logger.debug(
                                f"Skipping module tool '{name}' (type: {type(member)}): no 'resource' or 'db_resource' parameter in signature '{sig}'."
                            )
                    except ValueError as e_sig:
                        self.logger.debug(
                            f"Skipping module tool '{name}' (type: {type(member)}): could not get signature from {type(signature_target)}. Error: {e_sig}"
                        )
        self.logger.info(f"Combined available tools: {list(available_tools.keys())}")
        return available_tools

    async def execute_task_poc(self) -> Optional[Dict[str, Any]]:
        if not await self._load_task_components():
            self.logger.error("Failed to load task components.")
            return None
        if not self.base_resource:
            await self.setup_base_resource()
        if not self.base_resource:
            self.logger.error("Base resource setup failed or not performed.")
            return None
        if not self.reward_function:
            self.logger.error("Reward function not loaded.")
            return None  # Should be caught by _load_task_components

        self.logger.info(
            f"Starting execution for task '{self.task_definition.name}'..."
        )
        episode_resource: Optional[ForkableResource] = None
        evaluation_result: Optional[Dict[str, Any]] = None

        all_user_turns_successful_function_calls: List[List[Dict[str, Any]]] = (
            []
        )  # Track successful calls for reward fn, list of lists (per user turn)
        conversation_messages: List[Dict[str, Any]] = (
            []
        )  # Use dicts for API compatibility

        # --- Agent Model Setup ---
        agent_model_name = os.environ.get("MODEL_AGENT")
        if not agent_model_name:
            self.logger.error("MODEL_AGENT environment variable not set.")
            return None
        if agent_model_name.startswith("openai/"):
            self._initialize_openai_client()
            if not self._openai_client:
                self.logger.error("OpenAI client failed to initialize. Cannot proceed.")
                return None
            agent_model_name = agent_model_name.split("openai/", 1)[
                1
            ]  # Get actual model name
            self.logger.info(f"Using OpenAI model: {agent_model_name}")
        else:
            # Placeholder for other model providers if needed in the future
            self.logger.error(
                f"Unsupported model provider for MODEL_AGENT: {agent_model_name}"
            )
            return None

        try:
            # --- Task Setup ---
            if not await self._load_task_components():
                self.logger.error("Failed to load task components.")
                return None
            if not self.base_resource:
                await self.setup_base_resource()
            if not self.base_resource:
                self.logger.error("Base resource setup failed or not performed.")
                return None
            if not self.reward_function:
                self.logger.error("Reward function not loaded.")
                return None

            self.logger.info("Forking base resource for episode...")
            episode_resource = await self.base_resource.fork()
            self.logger.info(
                f"Episode resource forked: {type(episode_resource).__name__}"
            )

            # --- Initial Conversation State ---
            # The conversation_messages list will be built turn by turn.
            # We need a copy of the user turns from the task definition.
            user_turns_from_task: List[Dict[str, Any]] = []
            if self.task_definition.messages:
                for msg_data in self.task_definition.messages:
                    if isinstance(msg_data, dict) and msg_data.get("role") == "user":
                        # Ensure it's a dict and has a role, content can be complex
                        user_turns_from_task.append(msg_data)
                    elif isinstance(msg_data, Message) and msg_data.role == "user":
                        user_turns_from_task.append(
                            msg_data.model_dump(exclude_none=True)
                        )
                    else:
                        self.logger.warning(
                            f"Skipping non-user message or invalid message type in task definition's messages: {msg_data}"
                        )

            if not user_turns_from_task:
                self.logger.error(
                    "No user turns found in task definition's messages. Cannot proceed."
                )
                return None

            # --- Interaction Loop ---
            # Loop through the user turns defined in the task or up to poc_max_turns
            num_defined_user_turns = len(user_turns_from_task)
            max_interaction_turns = min(
                self.task_definition.poc_max_turns, num_defined_user_turns
            )

            current_user_turn_index = 0

            for turn_num in range(
                1, max_interaction_turns + 1
            ):  # Outer loop for user turns
                self.logger.info(
                    f"--- User Turn {turn_num}/{max_interaction_turns} (Overall Index {current_user_turn_index + 1}/{num_defined_user_turns}) ---"
                )

                current_user_turn_accumulated_successful_calls: List[Dict[str, Any]] = (
                    []
                )

                # Add the current user turn's message(s) to the conversation history
                if current_user_turn_index < num_defined_user_turns:
                    current_user_turn_message = user_turns_from_task[
                        current_user_turn_index
                    ]
                    # The user message content might be a string or a list of content blocks (e.g. for multi-modal)
                    # For BFCL, it's a string that might represent a JSON list of user messages for that turn.
                    # We need to parse it if it's a JSON string representing a list of messages.
                    try:
                        # Attempt to parse content if it's a string that looks like a JSON list
                        if isinstance(current_user_turn_message.get("content"), str):
                            parsed_content = json.loads(
                                current_user_turn_message["content"]
                            )
                            if isinstance(parsed_content, list):
                                for sub_msg_dict in parsed_content:
                                    if (
                                        isinstance(sub_msg_dict, dict)
                                        and "role" in sub_msg_dict
                                        and "content" in sub_msg_dict
                                    ):
                                        conversation_messages.append(sub_msg_dict)
                                    else:
                                        self.logger.warning(
                                            f"Skipping sub-message in user turn due to invalid format: {sub_msg_dict}"
                                        )
                                        conversation_messages.append(
                                            current_user_turn_message
                                        )  # Fallback to original if parsing fails partially
                                        break  # Stop processing sub-messages for this turn
                                else:  # If loop completed without break
                                    pass  # Successfully processed all sub-messages
                            else:  # Content is a JSON string but not a list
                                conversation_messages.append(current_user_turn_message)
                        else:  # Content is not a string or already a complex object
                            conversation_messages.append(current_user_turn_message)
                    except (
                        json.JSONDecodeError
                    ):  # Content is a string but not valid JSON
                        conversation_messages.append(current_user_turn_message)

                    current_user_turn_index += 1
                else:
                    self.logger.info(
                        "No more user turns defined by task. Ending interaction."
                    )
                    break  # Break outer loop if no more user messages from task def

                # 1. Get available tools for this user turn (can be dynamic based on resource state)
                # For BFCL, tools are generally static for the episode, but good practice to refresh.
                resource_tool_specs = await episode_resource.get_tools_spec()
                available_tools_adapters = await self._get_available_tools(
                    episode_resource
                )  # Get adapters for execution

                # Format tools for OpenAI API (should be done once per user turn, or if tools change)
                openai_tools: List[ChatCompletionToolParam] = []
                if OPENAI_AVAILABLE:
                    for spec in resource_tool_specs:
                        # Ensure spec has the structure {"type": "function", "function": {...}}
                        if "name" in spec and "parameters" in spec:
                            openai_tools.append(
                                ChatCompletionToolParam(
                                    type="function",
                                    function={
                                        "name": spec["name"],
                                        "description": spec.get("description", ""),
                                        "parameters": spec[
                                            "parameters"
                                        ],  # Assuming this matches OpenAI schema
                                    },
                                )
                            )
                        else:
                            self.logger.warning(
                                f"Skipping tool spec due to missing name/parameters: {spec}"
                            )
                else:
                    self.logger.warning(
                        "OpenAI not available, cannot format tools for API."
                    )

                if (
                    not available_tools_adapters and not openai_tools
                ):  # If no tools can be formed or executed
                    self.logger.info(
                        "No tools available from resource or module for this turn. Agent cannot make tool calls."
                    )
                    # Agent might still respond textually. Let the loop proceed for one LLM call.

                # Inner loop for multi-step tool use within this single user turn
                current_inner_step = 0
                while current_inner_step < MAX_STEPS_PER_USER_TURN:
                    current_inner_step += 1
                    self.logger.info(
                        f"--- User Turn {turn_num}, Inner Step {current_inner_step}/{MAX_STEPS_PER_USER_TURN} ---"
                    )

                    # 2. Call the LLM (OpenAI)
                    try:
                        self.logger.debug(
                            f"Calling OpenAI: model={agent_model_name}, messages_FULL_HISTORY={json.dumps(conversation_messages, indent=2)}, tools={openai_tools}"
                        )  # Log full message history
                        if not self._openai_client:
                            raise Exception("OpenAI client not initialized")

                        response = await self._openai_client.chat.completions.create(
                            model=agent_model_name,
                            messages=conversation_messages,  # type: ignore
                            tools=openai_tools if openai_tools else None,
                            tool_choice="auto" if openai_tools else None,
                        )
                        response_message = response.choices[0].message
                        self.logger.debug(
                            f"OpenAI response message: {response_message}"
                        )

                    except Exception as e_openai:
                        self.logger.error(
                            f"Error calling OpenAI API: {e_openai}", exc_info=True
                        )
                        # Break inner loop on API error, then outer loop will decide to continue or break.
                        # For now, let's break the outer loop as well to prevent cascading errors.
                        # TODO: Consider more nuanced error handling for outer loop.
                        evaluation_result = {"error": f"OpenAI API error: {e_openai}"}
                        # Clean up and return
                        if episode_resource:
                            await episode_resource.close()
                        if self.base_resource:
                            await self.base_resource.close()
                            self.base_resource = None
                        return evaluation_result

                    # 3. Process LLM Response
                    # Append assistant's response (content and tool calls) to history
                    conversation_messages.append(
                        response_message.model_dump(exclude_none=True)
                    )

                    tool_calls = response_message.tool_calls
                    if tool_calls:
                        self.logger.info(
                            f"Assistant requested {len(tool_calls)} tool calls in this step."
                        )
                        current_llm_response_successful_calls: List[Dict[str, Any]] = []
                        for tool_call in tool_calls:
                            function_name = tool_call.function.name
                            function_args_str = tool_call.function.arguments
                            self.logger.info(
                                f"Attempting tool call: {function_name}({function_args_str})"
                            )

                            tool_adapter = available_tools_adapters.get(function_name)
                            if tool_adapter:
                                try:
                                    function_args = json.loads(function_args_str)
                                    function_response = await tool_adapter(
                                        function_args
                                    )
                                    self.logger.info(
                                        f"Tool '{function_name}' result: {str(function_response)[:200]}..."
                                    )
                                    conversation_messages.append(
                                        {
                                            "tool_call_id": tool_call.id,
                                            "role": "tool",
                                            "name": function_name,
                                            "content": json.dumps(function_response),
                                        }
                                    )
                                    current_llm_response_successful_calls.append(
                                        {
                                            "name": function_name,
                                            "args": function_args,
                                        }
                                    )
                                except json.JSONDecodeError:
                                    self.logger.error(
                                        f"Failed to parse arguments for tool '{function_name}': {function_args_str}"
                                    )
                                    conversation_messages.append(
                                        {
                                            "tool_call_id": tool_call.id,
                                            "role": "tool",
                                            "name": function_name,
                                            "content": json.dumps(
                                                {"error": "Invalid JSON arguments"}
                                            ),
                                        }
                                    )
                                except Exception as e_tool_exec:
                                    self.logger.error(
                                        f"Error executing tool '{function_name}': {e_tool_exec}",
                                        exc_info=True,
                                    )
                                    conversation_messages.append(
                                        {
                                            "tool_call_id": tool_call.id,
                                            "role": "tool",
                                            "name": function_name,
                                            "content": json.dumps(
                                                {
                                                    "error": f"Execution failed: {e_tool_exec}"
                                                }
                                            ),
                                        }
                                    )
                            else:
                                self.logger.error(
                                    f"Tool '{function_name}' requested by model but not found in available tools."
                                )
                                conversation_messages.append(
                                    {
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": json.dumps(
                                            {"error": "Tool not found"}
                                        ),
                                    }
                                )

                        if current_llm_response_successful_calls:
                            current_user_turn_accumulated_successful_calls.extend(
                                current_llm_response_successful_calls
                            )

                        # If tool calls were made, continue the inner loop for the LLM to react to tool results.
                        if (
                            not openai_tools and not available_tools_adapters
                        ):  # No tools were ever available
                            self.logger.info(
                                "No tools were available, but LLM hallucinated tool calls. Breaking inner loop."
                            )
                            break  # Break inner loop
                    else:
                        # No tool calls from LLM in this step, means assistant provided a final textual response for this user turn.
                        self.logger.info(
                            "Assistant did not request tool calls in this step. Ending inner loop for this user turn."
                        )
                        break  # Break the inner while loop
                else:  # Inner while loop finished due to max_steps_per_user_turn
                    self.logger.warning(
                        f"Reached max steps ({MAX_STEPS_PER_USER_TURN}) for user turn {turn_num}. Ending inner loop."
                    )
                # End of inner while loop for multi-step tool use

                if current_user_turn_accumulated_successful_calls:
                    all_user_turns_successful_function_calls.append(
                        current_user_turn_accumulated_successful_calls
                    )
            # End of outer for loop for user turns

            # --- Evaluation ---
            self.logger.info("Evaluating task outcome...")
            task_achieved = False  # Reset task_achieved, as PoC logic is gone
            eval_criteria = self.task_definition.evaluation_criteria

            # Log evaluation_criteria and its relevant fields before calling reward function
            self.logger.debug(f"Evaluation criteria object: {eval_criteria}")
            if eval_criteria:
                self.logger.debug(
                    f"Evaluation criteria ground_truth_function_calls: {getattr(eval_criteria, 'ground_truth_function_calls', 'AttributeError or None')}"
                )
                self.logger.debug(
                    f"Evaluation criteria ground_truth_comparable_state: {getattr(eval_criteria, 'ground_truth_comparable_state', 'AttributeError or None')}"
                )

            # Check if episode_resource is SQLResource for final_state_query
            # from .resources import SQLResource # Would be needed here for isinstance
            if (
                eval_criteria and eval_criteria.final_state_query
            ):  # and isinstance(episode_resource, SQLResource):
                if hasattr(episode_resource, "step"):  # Generic check
                    query_res_step = await episode_resource.step(
                        "fetch_val_sql", {"query": eval_criteria.final_state_query}
                    )
                    if query_res_step.get("status") == "success":
                        outcome = query_res_step.get("result")
                        if eval_criteria.expected_query_result_transform:
                            try:
                                transform_func = eval(
                                    eval_criteria.expected_query_result_transform
                                )
                                task_achieved = bool(transform_func(outcome))
                            except Exception as e_tf:
                                self.logger.error(f"Error applying transform: {e_tf}")
                        else:
                            task_achieved = bool(outcome)
                        self.logger.info(
                            f"Final state query outcome: {outcome}, Task achieved: {task_achieved}"
                        )
                    else:
                        self.logger.error(
                            f"Failed to execute final_state_query: {query_res_step.get('message')}"
                        )

            # TODO: Re-evaluate how task_achieved should be determined without PoC logic
            # Maybe based on final observation, specific tool calls, or reward function logic itself?

            # Log evaluation_criteria and its relevant fields before calling reward function
            self.logger.debug(f"Evaluation criteria object: {eval_criteria}")
            if eval_criteria:
                self.logger.debug(
                    f"Evaluation criteria ground_truth_function_calls: {getattr(eval_criteria, 'ground_truth_function_calls', 'AttributeError or None')}"
                )
                self.logger.debug(
                    f"Evaluation criteria ground_truth_comparable_state: {getattr(eval_criteria, 'ground_truth_comparable_state', 'AttributeError or None')}"
                )

            # Prepare ground_truth dictionary for the reward function
            ground_truth_for_reward = None
            if eval_criteria:
                ground_truth_for_reward = {
                    "ground_truth_function_calls": getattr(
                        eval_criteria, "ground_truth_function_calls", None
                    ),
                    "ground_truth_comparable_state": getattr(
                        eval_criteria, "ground_truth_comparable_state", None
                    ),
                }

            # Prepare state dictionary for reward function
            state_for_reward = {
                "resource": episode_resource,
                "successful_func_calls": all_user_turns_successful_function_calls,
                # Add other relevant state info if needed
            }

            # Prepare eval_args dictionary
            eval_args = {
                "messages": conversation_messages,  # Pass final conversation history (as dicts)
                "state": state_for_reward,
                "task_achieved": task_achieved,  # Still needs proper determination
                # "tool_usage_counts": tool_usage_counts, # Replaced by successful_func_calls in state
                "task_definition_name": self.task_definition.name,
            }
            # Unpack the ground truth dictionary into eval_args if it exists
            if ground_truth_for_reward:
                eval_args.update(ground_truth_for_reward)

            # Call the reward function
            evaluation_result = self.reward_function(**eval_args)
            self.logger.info(f"Reward function result: {evaluation_result}")

        except Exception as e_lifecycle:
            self.logger.error(
                f"Exception during task lifecycle: {e_lifecycle}", exc_info=True
            )
            evaluation_result = {"error": str(e_lifecycle)}  # Return error info
        finally:
            if episode_resource:
                await episode_resource.close()
                self.logger.info("Episode resource closed.")
            if self.base_resource:
                await self.base_resource.close()
                self.base_resource = None
                self.logger.info("Base resource closed.")
        self.logger.info(f"Execution for task '{self.task_definition.name}' finished.")
        return evaluation_result
