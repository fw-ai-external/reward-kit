import asyncio
import logging
import uuid
from pathlib import Path 
from typing import Any, Dict, List, Optional, Set, Tuple, AsyncIterator # Keep AsyncIterator for now, might not be needed

import docker
import docker.errors
import docker.models.containers
import httpx

from reward_kit.mcp_agent.config import AppConfig, BackendServerConfig
import mcp.types as types 
from mcp.client.session import ClientSession, DEFAULT_CLIENT_INFO, SessionMessage
from mcp.client.stdio import StdioServerParameters, stdio_client
from anyio.abc import ObjectReceiveStream, ObjectSendStream # For type hints if needed directly

from reward_kit.mcp_agent.orchestration.base_client import (
    AbstractOrchestrationClient,
    ManagedInstanceInfo,
)

logger = logging.getLogger(__name__)
ENCODING = "utf-8"

class LocalDockerOrchestrationClient(AbstractOrchestrationClient):
    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self.docker_client: Optional[docker.DockerClient] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self._used_host_ports: Set[int] = set()
        self._temporary_images: Set[str] = set()
        
        # For managing dedicated tasks for stdio instances
        self._stdio_instance_tasks: Dict[str, asyncio.Task] = {}
        # To store ClientSession objects created by the dedicated tasks
        self._stdio_client_sessions: Dict[str, ClientSession] = {}
        # To store shutdown events for each stdio instance task
        self._stdio_shutdown_events: Dict[str, asyncio.Event] = {}

    async def startup(self) -> None:
        try:
            self.docker_client = docker.from_env() 
            if not self.docker_client.ping(): # type: ignore
                raise ConnectionError("Failed to connect to Docker daemon using docker.from_env().")
            logger.info("Successfully connected to Docker daemon.")
        except docker.errors.DockerException as e:
            logger.warning(f"docker.from_env() failed: {e}. Trying explicit base_url.")
            try:
                self.docker_client = docker.DockerClient(base_url="unix://var/run/docker.sock")
                if not self.docker_client.ping(): # type: ignore
                    raise ConnectionError("Failed to connect to Docker daemon with explicit base_url.")
                logger.info("Successfully connected to Docker daemon with explicit base_url.")
            except docker.errors.DockerException as e_explicit:
                 raise ConnectionError(f"Docker client initialization failed: {e_explicit}") from e_explicit

        api_defaults = self.app_config.global_remote_api_defaults if isinstance(self.app_config.global_remote_api_defaults, dict) else {}
        self.http_client = httpx.AsyncClient(timeout=api_defaults.get("timeout", 30.0))
        logger.info("LocalDockerOrchestrationClient started.")

    async def _manage_stdio_instance_lifecycle(
        self, 
        instance_uuid: str, 
        container_name: str,
        server_params: StdioServerParameters,
        initialization_complete_event: asyncio.Event,
        shutdown_event: asyncio.Event
    ):
        """Dedicated task to manage a single stdio_client's lifecycle."""
        client_session_stdio: Optional[ClientSession] = None
        try:
            logger.info(f"[{container_name}] Lifecycle task started.")
            async with stdio_client(server_params) as (read_stream, write_stream):
                logger.info(f"[{container_name}] Stdio transport established via stdio_client.")
                
                client_session_stdio = ClientSession(
                    read_stream=read_stream, write_stream=write_stream, 
                    client_info=DEFAULT_CLIENT_INFO
                )
                
                # Start the client session without using the context manager to avoid cancel scope conflicts
                await client_session_stdio.__aenter__()
                try:
                    logger.info(f"[{container_name}] Attempting to initialize ClientSession...")
                    await asyncio.wait_for(client_session_stdio.initialize(), timeout=15.0)
                    logger.info(f"[{container_name}] ClientSession initialized successfully.")
                    
                    # Log the tools reported by the backend server by calling list_tools
                    try:
                        # list_tools() should be called on the session, it returns ListToolsResult
                        # ListToolsResult has a 'tools' attribute which is List[Tool]
                        # Tool has a 'name' attribute.
                        list_tools_response = await asyncio.wait_for(client_session_stdio.list_tools(), timeout=5.0)
                        if hasattr(list_tools_response, 'tools') and list_tools_response.tools is not None:
                            reported_tools = [tool.name for tool in list_tools_response.tools]
                            logger.info(f"[{container_name}] Backend server reported tools: {reported_tools}")
                        else:
                            # This case implies list_tools_response is not as expected or tools list is empty/None
                            logger.warning(f"[{container_name}] Backend server list_tools response did not contain 'tools' attribute or it was None. Response: {list_tools_response}")
                    except AttributeError as e_attr:
                        # This handles cases where list_tools_response itself might be an error object without 'tools'
                        logger.warning(f"[{container_name}] AttributeError accessing tools from list_tools response: {e_attr}. Response: {list_tools_response}")
                    except Exception as e_list_tools:
                        logger.warning(f"[{container_name}] Error calling/processing list_tools on backend server: {e_list_tools}")

                    # Store the session and signal that initialization is complete
                    self._stdio_client_sessions[instance_uuid] = client_session_stdio
                    initialization_complete_event.set()
                    
                    # Keep running until shutdown is signaled
                    await shutdown_event.wait()
                    logger.info(f"[{container_name}] Shutdown event received.")
                
                finally:
                    # Properly close the client session
                    try:
                        await client_session_stdio.__aexit__(None, None, None)
                    except Exception as e_close:
                        logger.warning(f"[{container_name}] Error during ClientSession close: {e_close}")

            logger.info(f"[{container_name}] stdio_client context exited cleanly.")

        except asyncio.TimeoutError:
            logger.error(f"[{container_name}] Timeout during ClientSession initialization.")
            initialization_complete_event.set() # Signal to unblock provision_instances, even on failure
        except Exception as e:
            logger.error(f"[{container_name}] Error in stdio instance lifecycle: {e}", exc_info=True)
            initialization_complete_event.set() # Ensure provision_instances is unblocked
        finally:
            logger.debug(f"[{container_name}] In _manage_stdio_instance_lifecycle finally block.")
            if client_session_stdio is None:
                logger.info(f"[{container_name}] ClientSession was not created or assigned.")

            # Clean up references
            self._stdio_client_sessions.pop(instance_uuid, None)
            self._stdio_shutdown_events.pop(instance_uuid, None)
            logger.info(f"[{container_name}] Lifecycle task finished.")


    async def shutdown(self) -> None:
        if self.http_client: await self.http_client.aclose()
        
        logger.info(f"Shutting down LocalDockerOrchestrationClient. Cleaning up {len(self._stdio_instance_tasks)} stdio instance tasks.")
        # Signal all active stdio instance tasks to shut down
        for instance_uuid, event in list(self._stdio_shutdown_events.items()): # Iterate on copy
            logger.info(f"Signaling shutdown for stdio instance task {instance_uuid}.")
            event.set()
        
        # Wait for all tasks to complete
        tasks_to_wait_for = list(self._stdio_instance_tasks.values())
        if tasks_to_wait_for:
            await asyncio.gather(*tasks_to_wait_for, return_exceptions=True)
        logger.info("All stdio instance tasks awaited.")
        self._stdio_instance_tasks.clear() # Should be empty if tasks removed themselves

        if self.docker_client:
            for image_tag in list(self._temporary_images):
                try:
                    self.docker_client.images.remove(image=image_tag, force=False) # type: ignore
                    self._temporary_images.discard(image_tag)
                except Exception as e: logger.warning(f"Failed to remove temp image {image_tag}: {e}")
            if hasattr(self.docker_client, 'api') and hasattr(self.docker_client.api, 'close'):
                 self.docker_client.api.close() # type: ignore
            elif hasattr(self.docker_client, 'close'): 
                 self.docker_client.close() # type: ignore
        logger.info("LocalDockerOrchestrationClient shut down.")

    async def _perform_startup_check(self, url: str, check: Dict[str, Any]) -> bool:
        if not self.http_client: return False
        name, args = check.get("tool_name"), check.get("arguments", {})
        if not name: return True
        for attempt in range(5):
            try:
                res = await self.http_client.post(url, json={"tool_name": name, "arguments": args})
                res.raise_for_status(); return True
            except Exception as e:
                logger.warning(f"Startup check fail {attempt+1}/5: {e}")
                if attempt < 4: await asyncio.sleep(2)
        return False

    async def provision_instances(
        self, backend_config: BackendServerConfig, num_instances: int,
        session_id: str, template_details: Optional[Any] = None
    ) -> List[ManagedInstanceInfo]:
        if not self.docker_client: raise RuntimeError("Docker client not initialized.")
        image_to_run_from = backend_config.docker_image
        committed_img_tag: Optional[str] = None
        managed_instances: List[ManagedInstanceInfo] = []

        template_host_path = template_details or backend_config.template_data_path_host
        if backend_config.instance_scoping == "session" and template_host_path:
            # ... (template commit logic remains the same) ...
            if not backend_config.container_template_data_path:
                raise ValueError("container_template_data_path required for stateful session with template.")
            temp_cont_name = f"rk-mcp-template-{session_id}-{backend_config.backend_name_ref}-{uuid.uuid4().hex[:4]}"
            try:
                logger.info(f"Creating template container: {temp_cont_name} from {backend_config.docker_image}")
                temp_c = self.docker_client.containers.run( # type: ignore
                    image=backend_config.docker_image, name=temp_cont_name,
                    volumes={template_host_path: {"bind": backend_config.container_template_data_path, "mode": "rw"}},
                    detach=True
                )
                await asyncio.sleep(5) 
                committed_img_tag = f"rk-mcp-templateimg-{session_id}-{backend_config.backend_name_ref}:{uuid.uuid4().hex[:6]}"
                logger.info(f"Committing {temp_c.id} to {committed_img_tag}") # type: ignore
                temp_c.commit(repository=committed_img_tag.split(':')[0], tag=committed_img_tag.split(':')[1]) # type: ignore
                image_to_run_from = committed_img_tag
                self._temporary_images.add(committed_img_tag)
            finally:
                if 'temp_c' in locals() and temp_c: 
                    try: temp_c.stop(timeout=5); temp_c.remove() # type: ignore
                    except Exception as e: logger.warning(f"Could not cleanup template container: {e}")
        
        for i in range(num_instances):
            instance_uuid = uuid.uuid4().hex[:8]
            container_name = f"rk-mcp-inst-{session_id}-{backend_config.backend_name_ref}-{instance_uuid}"
            mcp_endpoint_url: Optional[str] = None
            host_port: Optional[int] = None
            instance_internal_details: Dict[str, Any] = {"container_name": container_name, "instance_uuid": instance_uuid}
            
            try:
                logger.info(f"Provisioning instance {container_name} (transport: {backend_config.mcp_transport})")
                if backend_config.mcp_transport == "http":
                    # ... (HTTP provisioning logic remains the same) ...
                    if not self.docker_client: raise RuntimeError("Docker client not initialized for HTTP provisioning.")
                    if not backend_config.container_port: raise ValueError("container_port required for http.")
                    port_bindings = {f"{backend_config.container_port}/tcp": 0} 
                    run_kwargs: Dict[str, Any] = {
                        "image": image_to_run_from, "name": container_name, "detach": True,
                        "command": backend_config.container_command, "volumes": backend_config.container_volumes,
                        "labels": {"rewardkit-mcp-session-id": session_id, "rewardkit-mcp-backend-name": backend_config.backend_name_ref, "rewardkit-mcp-instance-id": instance_uuid, "rewardkit-mcp-managed": "true"},
                        "ports": port_bindings, 
                        **(self.app_config.global_docker_options or {}), # type: ignore
                    }
                    container = self.docker_client.containers.run(**run_kwargs) # type: ignore
                    container.reload() # type: ignore
                    bindings = container.attrs.get('NetworkSettings', {}).get('Ports', {}).get(f"{backend_config.container_port}/tcp") # type: ignore
                    if not (bindings and bindings[0].get('HostPort')):
                        # ... (error handling for port binding) ...
                        logs = "N/A"; 
                        try: logs = container.logs(stdout=True, stderr=True).decode(ENCODING, 'replace') # type: ignore
                        except Exception: pass
                        logger.error(f"Failed to get host port for {container_name}. Logs:\n{logs}")
                        try: container.stop(timeout=5); container.remove() # type: ignore
                        except Exception: pass
                        raise RuntimeError(f"Failed to get host port for {container_name}")
                    host_port = int(bindings[0]['HostPort'])
                    self._used_host_ports.add(host_port)
                    mcp_endpoint_url = f"http://localhost:{host_port}/mcp"
                    if backend_config.startup_check_mcp_tool and not await self._perform_startup_check(mcp_endpoint_url, backend_config.startup_check_mcp_tool):
                        # ... (error handling for startup check) ...
                        logs = "N/A"; 
                        try: logs = container.logs(stdout=True, stderr=True).decode(ENCODING, 'replace') # type: ignore
                        except Exception: pass
                        logger.error(f"HTTP Startup check failed for {container_name}. Logs:\n{logs}")
                        try: container.stop(timeout=5); container.remove(); # type: ignore
                        except Exception: pass
                        self._used_host_ports.discard(host_port)
                        raise RuntimeError(f"Startup check failed for {container_name}")
                    logger.info(f"HTTP Instance {container_name} (ID: {container.id}) on port {host_port}") # type: ignore
                    instance_internal_details.update({"container_id": container.id, "host_port": host_port}) # type: ignore

                elif backend_config.mcp_transport == "stdio":
                    docker_run_args = ["run", "--rm", "-i", "--name", container_name]
                    if backend_config.container_volumes:
                        for h_path, c_path_dict in backend_config.container_volumes.items():
                            bind_path, mode = c_path_dict.get("bind"), c_path_dict.get("mode", "rw")
                            if bind_path: docker_run_args.extend(["-v", f"{Path(h_path).resolve()}:{bind_path}:{mode}"])
                    docker_run_args.append(image_to_run_from)
                    if backend_config.container_command: docker_run_args.extend(backend_config.container_command)
                    
                    server_params = StdioServerParameters(command="docker", args=docker_run_args, env=dict(os.environ))
                    logger.info(f"Preparing to launch stdio container {container_name} via dedicated task.")

                    initialization_complete_event = asyncio.Event()
                    shutdown_event = asyncio.Event()
                    self._stdio_shutdown_events[instance_uuid] = shutdown_event
                    
                    lifecycle_task = asyncio.create_task(
                        self._manage_stdio_instance_lifecycle(
                            instance_uuid, container_name, server_params, 
                            initialization_complete_event, shutdown_event
                        )
                    )
                    self._stdio_instance_tasks[instance_uuid] = lifecycle_task
                    
                    # Wait for the lifecycle task to initialize the ClientSession
                    logger.info(f"Waiting for stdio instance {container_name} (task) to complete initialization...")
                    await asyncio.wait_for(initialization_complete_event.wait(), timeout=30.0) # Increased timeout
                    
                    client_session_stdio = self._stdio_client_sessions.get(instance_uuid)
                    if not client_session_stdio:
                        # Task might have failed before setting the session
                        # Check if task is done and had an exception
                        if lifecycle_task.done() and lifecycle_task.exception():
                            raise RuntimeError(f"Stdio instance task for {container_name} failed during initialization.") from lifecycle_task.exception()
                        raise RuntimeError(f"ClientSession not established by lifecycle task for {container_name}.")
                    
                    logger.info(f"Stdio instance {container_name} (task) initialization complete. ClientSession ready.")
                    # No need to store client_session_stdio in instance_internal_details, it's in self._stdio_client_sessions

                    if backend_config.startup_check_mcp_tool:
                         logger.info(f"Performing startup check for stdio instance {container_name}...")
                         startup_tool_name = backend_config.startup_check_mcp_tool.get("tool_name", "ping")
                         startup_tool_args = backend_config.startup_check_mcp_tool.get("arguments", {})
                         # Use the retrieved session for the check (session is already started by lifecycle task)
                         await asyncio.wait_for(client_session_stdio.call_tool(startup_tool_name, startup_tool_args), timeout=10.0)
                         logger.info(f"Stdio startup check for {container_name} successful.")
                else:
                    raise ValueError(f"Unsupported mcp_transport: {backend_config.mcp_transport}")

                managed_instances.append(ManagedInstanceInfo(
                    instance_id=instance_uuid, backend_name_ref=backend_config.backend_name_ref,
                    orchestration_mode="local_docker", mcp_transport=backend_config.mcp_transport,
                    mcp_endpoint_url=mcp_endpoint_url, internal_instance_details=instance_internal_details,
                    committed_image_tag=committed_img_tag
                ))
            except Exception as e:
                logger.error(f"Failed to provision instance {container_name}: {e}", exc_info=True)
                # Ensure cleanup if task was started
                if backend_config.mcp_transport == "stdio":
                    if instance_uuid in self._stdio_shutdown_events:
                        self._stdio_shutdown_events[instance_uuid].set() # Signal task to stop
                    task_to_clean = self._stdio_instance_tasks.pop(instance_uuid, None)
                    if task_to_clean and not task_to_clean.done():
                        try:
                            await asyncio.wait_for(task_to_clean, timeout=5.0)
                        except asyncio.TimeoutError:
                            logger.warning(f"Timeout waiting for stdio task {instance_uuid} to clean up after provisioning error.")
                            task_to_clean.cancel()
                        except Exception as task_e:
                             logger.error(f"Exception during stdio task cleanup for {instance_uuid}: {task_e}")
                raise
        return managed_instances

    async def deprovision_instances(self, instances: List[ManagedInstanceInfo]) -> None:
        if not self.docker_client: logger.warning("Docker client not init for deprovision.")

        for instance in instances:
            if instance.orchestration_mode != "local_docker": continue
            
            details = instance.internal_instance_details
            instance_uuid = details.get("instance_uuid", instance.instance_id) # Use instance_uuid if stored

            if instance.mcp_transport == "http":
                # ... (HTTP deprovisioning remains the same) ...
                container_id = details.get("container_id")
                if not container_id or not self.docker_client: continue
                try:
                    container = self.docker_client.containers.get(container_id) # type: ignore
                    container.stop(timeout=10); container.remove() # type: ignore
                    logger.info(f"HTTP Container {container_id} deprovisioned.")
                    if details.get("host_port"): self._used_host_ports.discard(details["host_port"])
                except Exception as e: logger.error(f"Error deprovisioning HTTP container {container_id}: {e}")
            
            elif instance.mcp_transport == "stdio":
                logger.info(f"Deprovisioning stdio instance {instance_uuid} ({details.get('container_name')})...")
                
                shutdown_event = self._stdio_shutdown_events.pop(instance_uuid, None)
                if shutdown_event:
                    logger.info(f"Signaling shutdown for stdio instance task {instance_uuid}.")
                    shutdown_event.set()
                else:
                    logger.warning(f"No shutdown event found for stdio instance {instance_uuid}.")

                task = self._stdio_instance_tasks.pop(instance_uuid, None)
                if task:
                    logger.info(f"Waiting for stdio instance task {instance_uuid} to complete...")
                    try:
                        await asyncio.wait_for(task, timeout=10.0) # Wait for task to finish
                        logger.info(f"Stdio instance task {instance_uuid} completed.")
                    except asyncio.TimeoutError:
                        logger.error(f"Timeout waiting for stdio instance task {instance_uuid} to complete. Cancelling.")
                        task.cancel()
                        try:
                            await task # Await cancellation
                        except asyncio.CancelledError:
                            logger.info(f"Stdio instance task {instance_uuid} cancelled.")
                        except Exception as e_task_cancel:
                             logger.error(f"Exception during cancellation of stdio task {instance_uuid}: {e_task_cancel}")
                    except Exception as e_task_wait:
                        logger.error(f"Exception waiting for stdio instance task {instance_uuid}: {e_task_wait}")
                else:
                    logger.warning(f"No lifecycle task found for stdio instance {instance_uuid} during deprovision.")
                
                # ClientSession is managed by the task, _stdio_client_sessions should be cleaned by task
                if instance_uuid in self._stdio_client_sessions:
                    logger.warning(f"ClientSession for {instance_uuid} still in _stdio_client_sessions after task handling. Popping.")
                    self._stdio_client_sessions.pop(instance_uuid, None)

                logger.info(f"Stdio instance {instance_uuid} deprovisioning process complete.")

    async def call_tool_on_instance(
        self, instance: ManagedInstanceInfo, tool_name: str, tool_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        if instance.orchestration_mode != "local_docker":
            raise ValueError("Only handles local_docker instances.")

        if instance.mcp_transport == "http":
            # ... (HTTP call logic remains the same) ...
            if not self.http_client: raise RuntimeError("HTTP client not initialized.")
            if not instance.mcp_endpoint_url: raise ValueError(f"mcp_endpoint_url required for HTTP {instance.instance_id}")
            payload = {"tool_name": tool_name, "arguments": tool_args}
            try:
                res = await self.http_client.post(instance.mcp_endpoint_url, json=payload)
                res.raise_for_status(); return res.json()
            except Exception as e: raise RuntimeError(f"MCP HTTP call failed: {e}") from e

        elif instance.mcp_transport == "stdio":
            instance_uuid = instance.internal_instance_details.get("instance_uuid", instance.instance_id)
            cs = self._stdio_client_sessions.get(instance_uuid) # Get from central dict
            
            if not cs or not isinstance(cs, ClientSession):
                raise RuntimeError(f"Valid ClientSession not found for stdio instance {instance_uuid}.")
            
            # Call the tool directly on the session without using async with
            # The session is already managed by the lifecycle task
            try:
                logger.debug(f"Calling tool {tool_name} via stdio ClientSession for {instance_uuid}")
                tool_result = await cs.call_tool(tool_name, tool_args)
                if hasattr(tool_result, 'model_dump'): 
                    dumped = tool_result.model_dump(exclude_none=True)
                    if isinstance(dumped, dict): return dumped
                    return {"error": "Tool result model_dump was not a dict", "details": str(dumped)}
                if isinstance(tool_result, dict): return tool_result
                return {"error": "Tool result unexpected format", "details": str(tool_result)}
            except Exception as e:
                logger.error(f"MCP stdio tool call for {tool_name} on instance {instance_uuid} failed: {e}", exc_info=True)
                raise RuntimeError(f"MCP stdio tool call for {tool_name} failed: {e}") from e
        else:
            raise ValueError(f"Unsupported mcp_transport: {instance.mcp_transport}")

import os
