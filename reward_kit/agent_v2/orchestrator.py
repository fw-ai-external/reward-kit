"""
Orchestrator for the Agent Evaluation Framework V2.
Manages the lifecycle of a task using ForkableResources.
"""
import logging
import importlib
import asyncio
import inspect
from unittest.mock import AsyncMock # Add this import
from typing import Type, Optional, Callable, Dict, List, Any

from ..models import TaskDefinitionModel # Assuming models.py is one level up
from .resource_abc import ForkableResource
# Import specific resource types for type checking if needed, or handle dynamically
# from .resources import SQLResource # Example, if specific checks are needed

class Orchestrator:
    def __init__(self, task_definition: TaskDefinitionModel): 
        self.task_definition = task_definition
        self.base_resource: Optional[ForkableResource] = None
        self.tools_module: Optional[Any] = None
        self.reward_function: Optional[Callable[..., Any]] = None
        self.logger = logging.getLogger(f"Orchestrator.{self.task_definition.name}")
        self.logger.setLevel(logging.DEBUG) # Ensure debug logs are processed
        self.logger.info(f"Orchestrator initialized for task: {self.task_definition.name}")

    def _load_module_and_function(self, full_path: str) -> Optional[Callable[..., Any]]:
        try:
            module_path, function_name = full_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            func = getattr(module, function_name)
            if not callable(func):
                self.logger.error(f"Loaded attribute '{function_name}' from '{module_path}' is not callable.")
                return None
            self.logger.info(f"Successfully loaded function '{function_name}' from module '{module_path}'.")
            return func
        except (ImportError, AttributeError, ValueError) as e:
            self.logger.error(f"Failed to load function from '{full_path}': {e}")
            return None

    async def _load_task_components(self) -> bool:
        if self.task_definition.tools_module_path:
            try:
                self.tools_module = importlib.import_module(self.task_definition.tools_module_path)
                self.logger.info(f"Successfully loaded tools module: {self.task_definition.tools_module_path}")
            except ImportError as e:
                self.logger.error(f"Failed to import tools module '{self.task_definition.tools_module_path}': {e}")
                return False
        else:
            self.logger.info("No 'tools_module_path' specified. Tools may only come from resource.get_tools_spec().")

        if self.task_definition.reward_function_path:
            self.reward_function = self._load_module_and_function(self.task_definition.reward_function_path)
            if not self.reward_function: return False
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
        from .resources import PythonStateResource, SQLResource, FileSystemResource, DockerResource # noqa
        
        mapping = {
            "PythonStateResource": PythonStateResource,
            "SQLResource": SQLResource,
            "FileSystemResource": FileSystemResource,
            "DockerResource": DockerResource,
        }
        resource_class = mapping.get(resource_type_name)

        if resource_class is None: 
            raise ValueError(f"Resource class '{resource_type_name}' not found or not mapped in Orchestrator._get_resource_class.")
        # No need to check issubclass here if mapping is correct and types are imported.
        return resource_class

    async def setup_base_resource(self) -> None:
        resource_type = self.task_definition.resource_type
        base_config = self.task_definition.base_resource_config
        
        self.logger.info(f"Attempting to set up base resource of type '{resource_type}'...")
        try:
            ResourceClass = self._get_resource_class(resource_type)
            self.base_resource = ResourceClass()
            await self.base_resource.setup(base_config)
            self.logger.info(f"Base resource '{resource_type}' setup complete.")
        except ValueError as e_val: 
            self.logger.error(f"Could not get resource class '{resource_type}'. {e_val}"); self.base_resource = None
        except Exception as e_setup: 
            self.logger.error(f"Failed to setup base resource '{resource_type}'. {e_setup}", exc_info=True); self.base_resource = None

    async def _get_available_tools(self, episode_resource: ForkableResource) -> Dict[str, Callable[..., Any]]:
        available_tools: Dict[str, Callable[..., Any]] = {}
        if episode_resource:
            resource_tool_specs = await episode_resource.get_tools_spec()
            for tool_spec in resource_tool_specs:
                tool_name = tool_spec.get("function", {}).get("name")
                if tool_name:
                    async def resource_tool_adapter(params: Dict[str, Any], bound_tool_name=tool_name, bound_resource=episode_resource):
                        return await bound_resource.step(action_name=bound_tool_name, action_params=params)
                    available_tools[tool_name] = resource_tool_adapter
        
        if self.tools_module:
            self.logger.debug(f"Inspecting tools_module: {self.tools_module} (type: {type(self.tools_module)})")
            
            members_to_inspect = []
            if inspect.ismodule(self.tools_module):
                self.logger.debug(f"tools_module is a module. Using inspect.getmembers.")
                members_to_inspect = inspect.getmembers(self.tools_module)
            elif hasattr(self.tools_module, '__dict__'): # For class instances like DummyToolsModule
                self.logger.debug(f"tools_module is an object with __dict__. Iterating __dict__.items().")
                members_to_inspect = self.tools_module.__dict__.items()
            else: # Fallback to getmembers if no __dict__ or not a module
                self.logger.debug(f"tools_module is not a module and has no __dict__. Falling back to inspect.getmembers.")
                members_to_inspect = inspect.getmembers(self.tools_module)

            for name, member in members_to_inspect:
                self.logger.debug(f"Found member in tools_module: '{name}', type: {type(member)}, callable: {callable(member)}")
                if name.startswith("_") or not callable(member):
                    self.logger.debug(f"Skipping member '{name}' (startswith_ or not callable).")
                    continue

                is_tool_async_nature = False
                signature_target = member # Default to member itself for signature

                # Log type and isinstance check
                # Try isinstance first, then by name as a fallback for potential mock type identity issues
                is_async_mock_instance = isinstance(member, AsyncMock)
                if not is_async_mock_instance and type(member).__name__ == 'AsyncMock':
                    self.logger.warning(f"Member '{name}' identified as AsyncMock by name, not isinstance. Type: {type(member)}, Expected AsyncMock type: {type(AsyncMock)}")
                    is_async_mock_instance = True # Treat it as an AsyncMock
                
                self.logger.debug(f"Member '{name}': is_async_mock_instance = {is_async_mock_instance} (after name check), type is {type(member)}")

                if is_async_mock_instance:
                    is_tool_async_nature = True 
                    # Try to get the wrapped function for accurate signature inspection
                    if hasattr(member, 'func') and member.func is not None:
                        self.logger.debug(f"Member '{name}' is AsyncMock, using wrapped func (from .func) for signature: {member.func}")
                        signature_target = member.func
                    elif hasattr(member, '_mock_wraps') and member._mock_wraps is not None: # Check _mock_wraps (internal)
                        self.logger.debug(f"Member '{name}' is AsyncMock, using wrapped func (from ._mock_wraps) for signature: {member._mock_wraps}")
                        signature_target = member._mock_wraps
                    else:
                        self.logger.debug(f"Member '{name}' is AsyncMock, but no 'func' or '_mock_wraps' attribute found. Signature target will be the AsyncMock itself.")
                elif asyncio.iscoroutinefunction(member):
                    self.logger.debug(f"Member '{name}' is coroutine function.")
                    is_tool_async_nature = True
                else:
                    self.logger.debug(f"Member '{name}' is not identified as async tool (not AsyncMock by isinstance/name, not coroutine function).")

                if is_tool_async_nature:
                    try:
                        sig = inspect.signature(signature_target)
                        resource_param_name = next((pname for pname in ["resource", "db_resource"] if pname in sig.parameters), None)
                        
                        if resource_param_name:
                            async def module_tool_adapter(params: Dict[str, Any], bound_tool_func=member, bound_resource=episode_resource, res_param_name=resource_param_name):
                                tool_kwargs = {res_param_name: bound_resource, **params}
                                return await bound_tool_func(**tool_kwargs)
                            available_tools[name] = module_tool_adapter
                            self.logger.debug(f"Added tool '{name}' from tools_module (type: {type(member)}, signature target: {type(signature_target)}).")
                        else:
                            self.logger.debug(f"Skipping module tool '{name}' (type: {type(member)}): no 'resource' or 'db_resource' parameter in signature '{sig}'.")
                    except ValueError as e_sig: 
                        self.logger.debug(f"Skipping module tool '{name}' (type: {type(member)}): could not get signature from {type(signature_target)}. Error: {e_sig}")
        self.logger.info(f"Combined available tools: {list(available_tools.keys())}")
        return available_tools

    async def execute_task_poc(self) -> Optional[Dict[str, Any]]:
        if not await self._load_task_components(): self.logger.error("Failed to load task components."); return None
        if not self.base_resource: await self.setup_base_resource()
        if not self.base_resource: self.logger.error("Base resource setup failed or not performed."); return None
        if not self.reward_function: self.logger.error("Reward function not loaded."); return None # Should be caught by _load_task_components

        self.logger.info(f"Starting PoC execution for task '{self.task_definition.name}'...")
        episode_resource: Optional[ForkableResource] = None; evaluation_result: Optional[Dict[str, Any]] = None
        tool_usage_counts: Dict[str, Any] = {}

        try:
            self.logger.info("Forking base resource for episode..."); episode_resource = await self.base_resource.fork()
            self.logger.info(f"Episode resource forked: {type(episode_resource).__name__}")
            
            max_turns = self.task_definition.poc_max_turns; current_turn = 0
            observation = await episode_resource.get_observation()
            self.logger.info(f"Turn {current_turn}: Initial Observation: {str(observation)[:200]}...")
            all_tools = await self._get_available_tools(episode_resource)

            for turn_num in range(1, max_turns + 1):
                self.logger.info(f"--- Turn {turn_num}/{max_turns} ---")
                if not all_tools: self.logger.info("No tools available. Ending interaction."); break
                
                tool_to_call_name: Optional[str] = None; tool_params: Dict[str, Any] = {}
                # Specific PoC logic for flight task (requires SQLResource to be imported for isinstance check)
                # from .resources import SQLResource # Import locally if needed for isinstance
                # For now, this specific logic might not run if SQLResource is not yet in .resources
                if self.task_definition.name.startswith("Flight Booking Task"): # and isinstance(episode_resource, SQLResource)
                    if turn_num == 1 and "search_flights" in all_tools:
                        tool_to_call_name = "search_flights"; tool_params = {"origin": "SFO", "dest": "JFK", "date": "2023-09-16"}
                    elif turn_num == 2 and "create_booking" in all_tools and tool_usage_counts.get("search_flights_success_data"):
                        flight_id = tool_usage_counts.get("search_flights_success_data", {}).get("id")
                        if flight_id: tool_to_call_name = "create_booking"; tool_params = {"flight_id": flight_id, "passenger": "Alice"}
                    elif turn_num == 3 and "pay_booking" in all_tools and tool_usage_counts.get("create_booking_success_data"):
                        booking_id = tool_usage_counts.get("create_booking_success_data", {}).get("booking_id")
                        if booking_id: tool_to_call_name = "pay_booking"; tool_params = {"booking_id": booking_id, "payment_method": "credit_card_poc"}
                
                if not tool_to_call_name and all_tools: 
                    tool_to_call_name = list(all_tools.keys())[0]; tool_params = {"generic_param": "value"} 
                    self.logger.info(f"PoC Agent: Generic fallback, picked tool '{tool_to_call_name}'")

                if tool_to_call_name and tool_to_call_name in all_tools:
                    self.logger.info(f"PoC Agent: Calling tool '{tool_to_call_name}' with params: {tool_params}")
                    tool_func = all_tools[tool_to_call_name]
                    try:
                        tool_res = await tool_func(tool_params)
                        self.logger.info(f"Tool '{tool_to_call_name}' result: {str(tool_res)[:200]}...")
                        tool_usage_counts[tool_to_call_name] = tool_usage_counts.get(tool_to_call_name, 0) + 1
                        if tool_to_call_name == "search_flights" and tool_res and isinstance(tool_res, list) and len(tool_res) > 0:
                            tool_usage_counts["search_flights_success_data"] = tool_res[0] 
                        if tool_to_call_name == "create_booking" and isinstance(tool_res, dict) and not tool_res.get("error"):
                            tool_usage_counts["create_booking_success_data"] = tool_res
                        if tool_to_call_name == "pay_booking" and isinstance(tool_res, dict) and tool_res.get("success"):
                            self.logger.info("PoC Agent: Payment successful, ending interaction."); break
                    except Exception as e_tool: self.logger.error(f"Error calling tool '{tool_to_call_name}': {e_tool}")
                else: self.logger.info("PoC Agent: No tool to call this turn. Ending interaction."); break
                
                observation = await episode_resource.get_observation()
                self.logger.info(f"Turn {turn_num} completed. New Observation: {str(observation)[:200]}...")
            
            self.logger.info("Evaluating task outcome...")
            task_achieved = False
            eval_criteria = self.task_definition.evaluation_criteria
            # Check if episode_resource is SQLResource for final_state_query
            # from .resources import SQLResource # Would be needed here for isinstance
            if eval_criteria and eval_criteria.final_state_query: # and isinstance(episode_resource, SQLResource):
                if hasattr(episode_resource, 'step'): # Generic check
                    query_res_step = await episode_resource.step("fetch_val_sql", {"query": eval_criteria.final_state_query})
                    if query_res_step.get("status") == "success":
                        outcome = query_res_step.get("result")
                        if eval_criteria.expected_query_result_transform:
                            try: transform_func = eval(eval_criteria.expected_query_result_transform); task_achieved = bool(transform_func(outcome))
                            except Exception as e_tf: self.logger.error(f"Error applying transform: {e_tf}")
                        else: task_achieved = bool(outcome)
                        self.logger.info(f"Final state query outcome: {outcome}, Task achieved: {task_achieved}")
                    else: self.logger.error(f"Failed to execute final_state_query: {query_res_step.get('message')}")
            
            # if isinstance(episode_resource, SQLResource): # Update tool counts from DB
            if hasattr(episode_resource, 'step'): # Generic check for step method
                counts_res = await episode_resource.step("fetch_all_sql", {"query": "SELECT tool_name, COUNT(*) as count FROM tool_calls GROUP BY tool_name"})
                if counts_res.get("status") == "success" and counts_res.get("result"):
                    for r_item in counts_res["result"]: tool_usage_counts[r_item["tool_name"]] = r_item["count"]
                self.logger.info(f"Final tool usage counts from DB: {tool_usage_counts}")

            eval_args = {"task_achieved": task_achieved, "tool_usage_counts": tool_usage_counts, "task_definition_name": self.task_definition.name}
            evaluation_result = self.reward_function(**eval_args)
            self.logger.info(f"Reward function result: {evaluation_result}")

        except Exception as e_lifecycle: self.logger.error(f"Exception during task lifecycle: {e_lifecycle}", exc_info=True)
        finally:
            if episode_resource: await episode_resource.close(); self.logger.info("Episode resource closed.")
            if self.base_resource: await self.base_resource.close(); self.base_resource = None; self.logger.info("Base resource closed.")
        self.logger.info(f"PoC execution for task '{self.task_definition.name}' finished.")
        return evaluation_result
