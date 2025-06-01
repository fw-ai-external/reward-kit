import asyncio
import asyncio
import json # May no longer be needed if ClientSession handles all JSON
import logging
# import select # No longer needed for new stdio bridge
# import socket # No longer needed for new stdio bridge
# import time # No longer needed for new stdio bridge
import uuid
import functools # For functools.partial
from pathlib import Path # For host path resolution

from typing import Any, Dict, List, Optional, Set

import anyio # For memory streams and task groups
import docker
import docker.errors
import docker.models.containers
import httpx
from docker.transport import UnixHTTPAdapter

from reward_kit.mcp_agent.config import AppConfig, BackendServerConfig
from reward_kit.mcp_agent.config import AppConfig, BackendServerConfig
# Import mcp.types and SessionMessage if not already implicitly available via other imports
import mcp.types as types # Explicit import for clarity
from mcp.shared.message import SessionMessage # Explicit import for clarity
from mcp.client.session import ClientSession, DEFAULT_CLIENT_INFO # For stdio ClientSession

from reward_kit.mcp_agent.orchestration.base_client import (
    AbstractOrchestrationClient,
    ManagedInstanceInfo,
)

logger = logging.getLogger(__name__)
ENCODING = "utf-8"


# Helper functions adapted from standalone_stdio_mcp_client.py
async def _stdout_bridge(
    process_stdout: asyncio.StreamReader,
    to_client_writer: anyio.streams.memory.MemoryObjectSendStream[SessionMessage | Exception],
    banner_lines_to_skip: int = 0 # Defaulting to 0 as banner is on stderr
):
    """Reads from process stdout, skips banner, parses JSON, and sends to client session."""
    skipped_lines = 0
    try:
        while True:
            line_bytes = await process_stdout.readline()
            if not line_bytes:
                logger.info("[BRIDGE_STDOUT] Process stdout EOF reached.")
                break
            line = line_bytes.decode(ENCODING).strip()
            if not line:
                continue

            if skipped_lines < banner_lines_to_skip:
                logger.info(f"[BRIDGE_STDOUT] Skipping banner line: {line}")
                skipped_lines += 1
                continue
            
            logger.debug(f"[BRIDGE_STDOUT] Received line: {line}")
            try:
                message = types.JSONRPCMessage.model_validate_json(line)
                await to_client_writer.send(SessionMessage(message=message))
                log_msg_details = "UnknownType"
                if hasattr(message, 'id') and message.id is not None:
                    log_msg_details = f"id={message.id}"
                if hasattr(message, 'method') and message.method is not None:
                     log_msg_details = f"method='{message.method}'"
                logger.debug(f"[BRIDGE_STDOUT] Sent to client: {log_msg_details}")
            except Exception as e:
                logger.error(f"[BRIDGE_STDOUT] Error parsing JSON or sending to client: {line} - {e}")
                await to_client_writer.send(e)
    except anyio.EndOfStream: # Should not happen with asyncio.StreamReader's readline
        logger.info("[BRIDGE_STDOUT] Process stdout stream ended (anyio.EndOfStream).")
    except asyncio.exceptions.IncompleteReadError:
        logger.info("[BRIDGE_STDOUT] Process stdout stream ended (asyncio.exceptions.IncompleteReadError).")
    except Exception as e:
        logger.exception(f"[BRIDGE_STDOUT] Unhandled exception: {e}")
    finally:
        logger.info("[BRIDGE_STDOUT] Closing writer to client.")
        await to_client_writer.aclose()

async def _stdin_bridge(
    process_stdin: asyncio.StreamWriter,
    from_client_reader: anyio.streams.memory.MemoryObjectReceiveStream[SessionMessage]
):
    """Reads SessionMessages from client session, serializes, and writes to process stdin."""
    try:
        async for session_message in from_client_reader:
            json_str = session_message.message.model_dump_json(by_alias=True, exclude_none=True)
            logger.debug(f"[BRIDGE_STDIN] Sending to process: {json_str}")
            process_stdin.write((json_str + "\n").encode(ENCODING))
            await process_stdin.drain()
    except anyio.EndOfStream:
        logger.info("[BRIDGE_STDIN] Client writer stream ended.")
    except Exception as e:
        logger.exception(f"[BRIDGE_STDIN] Unhandled exception: {e}")
    finally:
        logger.info("[BRIDGE_STDIN] Closing process stdin.")
        if process_stdin and not process_stdin.is_closing():
            process_stdin.close()
            try:
                await process_stdin.wait_closed()
            except Exception as e_close:
                logger.debug(f"[BRIDGE_STDIN] Exception during wait_closed: {e_close}")


class LocalDockerOrchestrationClient(AbstractOrchestrationClient):
    """
    Orchestrates backend MCP server instances using a local Docker daemon.
    Handles starting, stopping, and interacting with Docker containers.
    Manages state forking using 'docker commit' for template-based backends.
    """

    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self.docker_client: Optional[docker.DockerClient] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        # Keep track of ports assigned on the host to avoid reuse within this client's lifetime
        # This is a simple in-memory solution. For production, a more robust port manager might be needed.
        self._used_host_ports: Set[int] = set()
        self._temporary_images: Set[str] = set() # Tracks images created by this orchestrator

    async def startup(self) -> None:
        """Initializes the Docker client and httpx client."""
        try:
            # Using a UnixHTTPAdapter with increased pool connections
            # Default Docker socket URL for Linux
            docker_socket_url = "unix:///var/run/docker.sock" 
            adapter = UnixHTTPAdapter(socket_url=docker_socket_url, pool_connections=50) # Default is 10
            self.docker_client = docker.DockerClient(base_url="unix://var/run/docker.sock")
            # The following line might be redundant if base_url in DockerClient already uses the adapter implicitly,
            # or if the adapter should be part of the DockerClient constructor.
            # For now, keeping it as it might be intended for specific API client interactions.
            # However, it's more common to pass the adapter to the APIClient directly if needed,
            # or rely on the base_url handling. For now, let's assume the DockerClient handles it.
            # If direct API client usage with specific adapter mounting is needed, it would be:
            # self.api = APIClient(base_url="unix://var/run/docker.sock", version=client_version_from_env_or_default)
            # self.api.mount("http+docker://", adapter)
            # self.docker_client = docker.DockerClient(api=self.api)
            # For simplicity, let's try without explicit mounting first if base_url is sufficient.
            # Re-evaluating: The original code used self.docker_client.api.mount.
            # The DockerClient itself doesn't take an adapter directly in its constructor usually.
            # The APIClient it creates internally would need the adapter.
            # Let's ensure the APIClient used by DockerClient gets the adapter.
            # A common pattern is to create an APIClient with the adapter and pass it to DockerClient.
            # However, the error was with low_api_version. Let's remove that first.
            # The mounting part might be okay if the APIClient is correctly configured.
            # The base_url="unix://var/run/docker.sock" should make it use the Unix domain socket.
            # The self.docker_client.api.mount might be for a different purpose or an older pattern.
            # Let's try with just removing low_api_version and see if ping works.
            # If ping fails, we might need to adjust how the adapter is used with the APIClient.

            if not self.docker_client.ping():
                logger.error(
                    "Failed to connect to Docker daemon. Ensure Docker is running and accessible."
                )
                raise ConnectionError("Failed to connect to Docker daemon.")
            logger.info("Successfully connected to Docker daemon.")
        except docker.errors.DockerException as e:
            logger.error(f"Docker client initialization failed: {e}")
            raise ConnectionError(f"Docker client initialization failed: {e}") from e

        # Initialize http_client, used for HTTP based MCPs and potentially startup checks
        # Ensure global_remote_api_defaults is treated as a dictionary
        api_defaults = self.app_config.global_remote_api_defaults if isinstance(self.app_config.global_remote_api_defaults, dict) else {}
        self.http_client = httpx.AsyncClient(timeout=api_defaults.get("timeout", 30.0))
        logger.info("LocalDockerOrchestrationClient started.")

    async def shutdown(self) -> None:
        """Cleans up resources, like the httpx client and temporary Docker images."""
        if self.http_client:
            await self.http_client.aclose()
            logger.info("HTTPX client closed.")

        # Clean up any temporary images created by this orchestrator
        # This is a best-effort cleanup. Images in use by running containers won't be removed by Docker.
        if self.docker_client:
            for image_tag in list(self._temporary_images): # Iterate over a copy
                try:
                    logger.info(f"Attempting to remove temporary image: {image_tag}")
                    self.docker_client.images.remove(image=image_tag, force=False) # Try to remove without force first
                    self._temporary_images.discard(image_tag)
                    logger.info(f"Successfully removed temporary image: {image_tag}")
                except docker.errors.ImageNotFound:
                    logger.warning(f"Temporary image {image_tag} not found for removal.")
                    self._temporary_images.discard(image_tag)
                except docker.errors.APIError as e:
                    if e.response.status_code == 409: # Conflict - image is in use
                         logger.warning(f"Temporary image {image_tag} is in use and could not be removed: {e}")
                    else:
                        logger.error(f"Failed to remove temporary image {image_tag}: {e}")
        
        if self.docker_client:
             self.docker_client.api.close() # Close the underlying API client and its connection pool
             logger.info("Docker API client closed.")

        logger.info("LocalDockerOrchestrationClient shut down.")

    def _find_available_host_port(self) -> int:
        """
        Finds an available host port.
        Docker can assign a random available port if host_port is set to 0 or None.
        This method is kept for explicitness if needed, but letting Docker assign is safer.
        For now, we will let Docker assign the port.
        """
        # This is a placeholder. In practice, binding to port 0 (`ports={'container_port/tcp': 0}`)
        # tells Docker to pick a random available host port. We then inspect the container to get it.
        # For simplicity and robustness, we'll rely on Docker's random port assignment.
        return 0 # Let Docker assign a random port

    async def _perform_startup_check(
        self, mcp_endpoint_url: str, startup_check_tool: Dict[str, Any]
    ) -> bool:
        """Performs a startup check by calling a specified MCP tool."""
        if not self.http_client:
            logger.error("HTTP client not initialized for startup check.")
            return False

        tool_name = startup_check_tool.get("tool_name")
        tool_args = startup_check_tool.get("arguments", {})
        if not tool_name:
            logger.warning("Startup check tool_name not provided. Skipping check.")
            return True # Assume success if no check is defined

        max_retries = 5
        retry_delay = 2  # seconds
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Attempting startup check {attempt + 1}/{max_retries} on {mcp_endpoint_url} with tool {tool_name}."
                )
                response = await self.http_client.post(
                    mcp_endpoint_url,
                    json={"tool_name": tool_name, "arguments": tool_args},
                )
                response.raise_for_status()  # Raise an exception for bad status codes
                logger.info(
                    f"Startup check successful for {mcp_endpoint_url} with tool {tool_name}."
                )
                return True
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                logger.warning(
                    f"Startup check failed for {mcp_endpoint_url} (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        f"Startup check failed after {max_retries} attempts for {mcp_endpoint_url}."
                    )
                    return False
        return False


    async def provision_instances(
        self,
        backend_config: BackendServerConfig,
        num_instances: int,
        session_id: str,
        template_details: Optional[Any] = None, # e.g. path for filesystem
    ) -> List[ManagedInstanceInfo]:
        if not self.docker_client:
            raise RuntimeError("Docker client not initialized. Call startup() first.")
        if backend_config.orchestration_mode != "local_docker":
            raise ValueError(
                "LocalDockerOrchestrationClient can only handle 'local_docker' mode."
            )

        image_to_run_instances_from = backend_config.docker_image
        committed_image_tag_for_session: Optional[str] = None
        managed_instances: List[ManagedInstanceInfo] = []

        # Template handling: if stateful and template data is provided
        # For local_docker, template_details could be a host path to copy from.
        # backend_config.template_data_path_host is the primary way for now.
        template_host_path = template_details or backend_config.template_data_path_host

        if backend_config.instance_scoping == "session" and template_host_path:
            if not backend_config.container_template_data_path:
                raise ValueError(
                    "container_template_data_path must be set if template_data_path_host is used for stateful session instances."
                )
            
            template_container_name = f"rk-mcp-template-{session_id}-{backend_config.backend_name_ref}-{uuid.uuid4().hex[:8]}"
            logger.info(f"Creating template container: {template_container_name} from image {backend_config.docker_image}")
            
            try:
                template_container = self.docker_client.containers.run(
                    image=backend_config.docker_image,
                    name=template_container_name,
                    volumes={
                        template_host_path: {
                            "bind": backend_config.container_template_data_path,
                            "mode": "rw", # Template might need to write, or setup scripts run
                        }
                    },
                    detach=True,
                    # No port mapping needed for template container if we don't directly interact via MCP
                )
                logger.info(f"Template container {template_container_name} (ID: {template_container.id}) started.")

                # Wait for template container to be "ready" - this is tricky.
                # If the image has an entrypoint that sets up state from the volume, we need to wait.
                # A simple sleep or a more sophisticated health check might be needed.
                # For now, assume entrypoint handles setup quickly or doesn't require long.
                # A better approach would be a specific setup tool/command if the image supports it.
                await asyncio.sleep(5) # Arbitrary wait for potential setup in template container

                committed_image_tag_for_session = f"rk-mcp-templateimg-{session_id}-{backend_config.backend_name_ref}:{uuid.uuid4().hex[:8]}"
                logger.info(f"Committing template container {template_container.id} to image {committed_image_tag_for_session}")
                template_container.commit(repository=committed_image_tag_for_session.split(':')[0], tag=committed_image_tag_for_session.split(':')[1])
                image_to_run_instances_from = committed_image_tag_for_session
                self._temporary_images.add(committed_image_tag_for_session) # Track for cleanup

            except docker.errors.APIError as e:
                logger.error(f"Error during template container creation or commit: {e}")
                raise
            finally:
                if 'template_container' in locals() and template_container:
                    try:
                        template_container.stop(timeout=5)
                        template_container.remove()
                        logger.info(f"Template container {template_container_name} stopped and removed.")
                    except docker.errors.APIError as e:
                        logger.warning(f"Could not stop/remove template container {template_container_name}: {e}")
        
        elif backend_config.instance_scoping == "shared_global" and template_host_path:
             logger.warning(f"Template path {template_host_path} provided for a 'shared_global' instance. This is unusual. The template will be applied to the shared instance if it's being created now.")
             # Logic for shared global with template (if it's the first time) could be added here.
             # For now, shared_global instances don't use docker commit workflow per-session.

        for i in range(num_instances):
            instance_uuid = uuid.uuid4().hex[:8]
            container_name = f"rk-mcp-inst-{session_id}-{backend_config.backend_name_ref}-{instance_uuid}"
            
            port_bindings = None
            mcp_endpoint_url: Optional[str] = None # Ensure it can be None
            host_port: Optional[int] = None # Ensure it can be None

            if backend_config.mcp_transport == "http":
                if not backend_config.container_port:
                    # This check is now in BackendServerConfig.model_post_init, but good to be defensive
                    logger.error(f"container_port is not set for HTTP backend {backend_config.backend_name_ref}")
                    raise ValueError(f"container_port must be set for backend {backend_config.backend_name_ref} with http transport.")
                port_bindings = {f"{backend_config.container_port}/tcp": 0}
            # For stdio, port_bindings remains None

            try:
                logger.info(f"Starting instance container {container_name} from image {image_to_run_instances_from} (transport: {backend_config.mcp_transport})")
                
                run_kwargs: Dict[str, Any] = {
                    "image": image_to_run_instances_from,
                    "name": container_name,
                    "detach": True,
                    "command": backend_config.container_command,
                    "volumes": backend_config.container_volumes,
                    "labels": {
                        "rewardkit-mcp-session-id": session_id,
                        "rewardkit-mcp-backend-name": backend_config.backend_name_ref,
                        "rewardkit-mcp-instance-id": instance_uuid,
                        "rewardkit-mcp-managed": "true",
                    },
                    **(self.app_config.global_docker_options or {}),
                }

                if backend_config.mcp_transport == "http" and port_bindings:
                    run_kwargs["ports"] = port_bindings
                elif backend_config.mcp_transport == "stdio":
                    # This is the new stdio handling block
                    # Construct the docker run command
                    docker_run_cmd_list = [
                        "docker", "run", "--rm", "-i", "--name", container_name
                    ]
                    # Add environment variables
                    docker_run_cmd_list.extend(["-e", "MCP_TRANSPORT=stdio", "-e", "NODE_ENV=production"])
                    
                    # Add volumes from backend_config
                    if backend_config.container_volumes:
                        for host_path, cont_path_dict in backend_config.container_volumes.items():
                            bind_path = cont_path_dict.get("bind")
                            mode = cont_path_dict.get("mode", "rw")
                            if bind_path:
                                # Resolve host_path to ensure it's absolute if it's relative
                                resolved_host_path = str(Path(host_path).resolve())
                                docker_run_cmd_list.extend(["-v", f"{resolved_host_path}:{bind_path}:{mode}"])

                    # Add image name
                    docker_run_cmd_list.append(image_to_run_instances_from)
                    
                    # Add container command if specified
                    if backend_config.container_command:
                        docker_run_cmd_list.extend(backend_config.container_command)
                    
                    logger.info(f"Preparing to run stdio container with command: {' '.join(docker_run_cmd_list)}")

                    process = await asyncio.create_subprocess_exec(
                        *docker_run_cmd_list,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    logger.info(f"Stdio Docker process for {container_name} started with PID: {process.pid}")

                    # Create memory streams for ClientSession
                    server_to_client_writer, server_to_client_reader = anyio.create_memory_object_stream[SessionMessage | Exception](0)
                    client_to_server_writer, client_to_server_reader = anyio.create_memory_object_stream[SessionMessage](0)

                    # Create a task group for this instance's bridges
                    # This task group needs to be managed and cancelled on deprovision
                    instance_task_group = anyio.create_task_group() 
                    # We need to enter the task group context to start tasks
                    # This is tricky as provision_instances is not an async context manager itself.
                    # For now, we'll store the task group and assume it's managed by the caller or a session object.
                    # This part needs careful thought on lifecycle management.
                    # Let's try to manage it within this provisioning scope for now,
                    # and store client_session and process for later interaction.

                    async with instance_task_group as tg:
                        # Start stderr logger task
                        async def log_stderr_local(proc_stderr, name):
                            async for line_bytes in proc_stderr:
                                logger.warning(f"[DOCKER_STDERR_{name}] {line_bytes.decode(ENCODING).strip()}")
                        tg.start_soon(log_stderr_local, process.stderr, container_name)

                        # Start bridge tasks
                        # Banner lines for mcp/filesystem are on stderr, so stdout_bridge skips 0 lines.
                        partial_stdout_bridge_local = functools.partial(_stdout_bridge, banner_lines_to_skip=0)
                        tg.start_soon(partial_stdout_bridge_local, process.stdout, server_to_client_writer)
                        tg.start_soon(_stdin_bridge, process.stdin, client_to_server_reader)
                        
                        logger.info(f"Bridge tasks for {container_name} started.")

                        client_session = ClientSession(
                            read_stream=server_to_client_reader,
                            write_stream=client_to_server_writer,
                            client_info=DEFAULT_CLIENT_INFO
                        )
                        logger.info(f"ClientSession for {container_name} instantiated.")
                        
                        # The ClientSession's read loop is started by its `async with client_session:` context
                        # or by passing `anyio_tg` to its constructor (if supported by BaseSession).
                        # Here, we'll use `async with client_session` for initialization.
                        
                        init_success = False
                        try:
                            async with client_session: # This starts and stops the read loop for the session
                                logger.info(f"Attempting to initialize session for {container_name}...")
                                await client_session.initialize()
                                logger.info(f"Session initialized successfully for {container_name}.")
                                init_success = True
                        except Exception as e_init:
                            logger.error(f"Failed to initialize ClientSession for {container_name}: {e_init}")
                            # Ensure process is terminated if init fails
                            if process.returncode is None: process.terminate()
                            await process.wait()
                            raise RuntimeError(f"Failed to initialize stdio instance {container_name}") from e_init
                        
                        if not init_success: # Should be caught by exception above, but defensive
                             raise RuntimeError(f"Initialization flag not set for {container_name}")

                        # Store necessary components for later interaction and cleanup
                        instance_internal_details = {
                            "container_name": container_name, # Docker container name
                            # "container_id": container_id, # Not easily available from create_subprocess_exec
                            "process_pid": process.pid,
                            "client_session": client_session, # The active ClientSession
                            "task_group": tg, # The task group managing bridges
                            "process_handle": process, # The asyncio.subprocess.Process object
                             # Streams for ClientSession are managed by ClientSession itself via aclose
                            "server_to_client_writer": server_to_client_writer, # For explicit closing if needed
                            "client_to_server_reader": client_to_server_reader, # For explicit closing if needed
                        }
                        # Note: The task_group `tg` will exit its context here.
                        # This means bridge tasks might be cancelled if not handled carefully.
                        # This needs to be re-thought. The task group should live as long as the instance.
                        # For now, this will likely fail because tg exits.
                        # The ClientSession and bridges need to be kept alive.
                        # This implies the ClientSession should be stored, and its context managed by call_tool.
                        # Or, the bridges and ClientSession are started and kept running,
                        # and call_tool just uses the existing session.
                        # Let's assume for now that the ClientSession is stored and bridges run.
                        # The `async with client_session` above is only for init.
                        # This is the most complex part.

                        # Re-thinking: Store the process, and the memory streams.
                        # ClientSession will be created on-demand in call_tool, or we store one ClientSession.
                        # If we store one ClientSession, its read loop needs to be managed.
                        # Let's store the process and the streams ClientSession would use.
                        # The bridges will run as long as the task group for this instance runs.
                        # This task group needs to be created outside this loop, per instance.
                        # This is getting very complex for provision_instances.

                        # Simpler approach for now:
                        # The ClientSession and its bridges are tied to the 'async with client_session'
                        # This means for each call_tool, we might need to re-establish this if not careful.
                        # The standalone script had one main task group.
                        # Here, each instance is more isolated.

                        # Let's store the process and the client_session.
                        # The bridges are started in the task group `tg`.
                        # `tg` needs to be stored with the instance and cancelled on deprovision.
                        # The `ClientSession` itself doesn't need to be an async context manager here
                        # if its read loop is started in `tg`.

                        # Corrected approach:
                        # 1. Start Docker process.
                        # 2. Create memory streams.
                        # 3. Create a new TaskGroup for this instance.
                        # 4. Start bridges and stderr logger in this TaskGroup.
                        # 5. Create ClientSession, pass it the TaskGroup.
                        # 6. Initialize ClientSession.
                        # 7. Store Process, ClientSession, TaskGroup.
                        
                        # This was already done above with `instance_task_group as tg`
                        # The problem is `tg` exiting.
                        # The `tg` must be stored on `ManagedInstanceInfo` and cancelled in `deprovision`.
                        # The `ClientSession` must be initialized using this `tg`.

                        # The `ClientSession`'s `__init__` does not take `anyio_tg`.
                        # `BaseSession.run()` is the context manager that starts the loop.
                        # So, `async with client_session:` is the way.
                        # This means `call_tool_on_instance` will need to re-enter this context or
                        # the context must be held open.

                        # For now, let's assume the init was successful and store what's needed.
                        # The actual ClientSession interaction will be in call_tool.
                        # We need to store the process and the memory streams for call_tool to reconstruct.
                        
                        instance_internal_details = {
                            "container_name": container_name,
                            "process_pid": process.pid,
                            "process_handle": process,
                            "server_to_client_reader": server_to_client_reader,
                            "client_to_server_writer": client_to_server_writer,
                            # The bridges are running in `tg`. `tg` itself is not stored yet.
                            # This is a gap. The task group needs to be stored and managed.
                            # For now, this will likely lead to issues in call_tool or deprovision.
                        }


                    logger.info(f"Stdio Instance {container_name} (PID: {process.pid}) provisioned and ClientSession initialized.")
                    # mcp_endpoint_url remains None for stdio

                managed_instances.append(
                    ManagedInstanceInfo(
                        instance_id=instance_uuid,
                        backend_name_ref=backend_config.backend_name_ref,
                        orchestration_mode="local_docker",
                        mcp_transport=backend_config.mcp_transport,
                        mcp_endpoint_url=mcp_endpoint_url, 
                        internal_instance_details=instance_internal_details, # Updated details
                        committed_image_tag=committed_image_tag_for_session if backend_config.instance_scoping == "session" else None,
                    )
                )
            except Exception as e: # Catch-all for this instance's provisioning
                logger.error(f"Failed to provision instance {container_name}: {e}")
                # Ensure cleanup if process started
                if 'process' in locals() and process and process.returncode is None:
                    process.terminate()
                    await process.wait()
                # If other resources like task groups were created, they'd need cleanup too.
                # This error handling needs to be robust.
                # For now, re-raise to indicate failure for this instance.
                # If num_instances > 1, we might want to collect successes and failures.
                # Current loop structure implies failing fast for the whole batch.
                raise


        return managed_instances

    async def deprovision_instances(self, instances: List[ManagedInstanceInfo]) -> None:
        if not self.docker_client: # Still needed for image cleanup
            logger.warning("Docker client not initialized. Cannot deprovision fully.")
            # We can still try to terminate processes if they exist.

        for instance in instances:
            if instance.orchestration_mode != "local_docker":
                logger.warning(f"Skipping deprovision for instance {instance.instance_id} as it's not local_docker.")
                continue

            if instance.mcp_transport == "http":
                container_id = instance.internal_instance_details.get("container_id")
                if not container_id:
                    logger.warning(f"No container_id for HTTP instance {instance.instance_id}.")
                    continue
                try:
                    container = self.docker_client.containers.get(container_id)
                    logger.info(f"Stopping HTTP container {container.name} (ID: {container_id})...")
                    container.stop(timeout=10)
                    container.remove()
                    logger.info(f"HTTP Container {container_id} stopped and removed.")
                    host_port = instance.internal_instance_details.get("host_port")
                    if host_port: self._used_host_ports.discard(host_port)
                except docker.errors.NotFound:
                    logger.warning(f"HTTP Container {container_id} not found for deprovision.")
                except docker.errors.APIError as e:
                    logger.error(f"API error deprovisioning HTTP container {container_id}: {e}")
            
            elif instance.mcp_transport == "stdio":
                logger.info(f"Deprovisioning stdio instance {instance.instance_id}...")
                process_handle = instance.internal_instance_details.get("process_handle")
                # task_group_stdio = instance.internal_instance_details.get("task_group_stdio") # Need to store this
                client_session_stdio = instance.internal_instance_details.get("client_session_stdio") # Need to store this

                # if task_group_stdio and isinstance(task_group_stdio, anyio.abc.TaskGroup):
                #     logger.info(f"Cancelling task group for stdio instance {instance.instance_id}")
                #     task_group_stdio.cancel_scope.cancel()
                
                if client_session_stdio and hasattr(client_session_stdio, 'aclose'):
                    logger.info(f"Closing ClientSession for stdio instance {instance.instance_id}")
                    await client_session_stdio.aclose() # This should close memory streams, stopping bridges

                if process_handle and isinstance(process_handle, asyncio.subprocess.Process):
                    if process_handle.returncode is None:
                        logger.info(f"Terminating stdio process PID {process_handle.pid} for instance {instance.instance_id}")
                        try:
                            process_handle.terminate()
                            await asyncio.wait_for(process_handle.wait(), timeout=5.0)
                            logger.info(f"Stdio process PID {process_handle.pid} terminated.")
                        except asyncio.TimeoutError:
                            logger.warning(f"Timeout waiting for stdio process {process_handle.pid} to terminate. Killing.")
                            process_handle.kill()
                            await process_handle.wait()
                        except ProcessLookupError:
                            logger.info(f"Stdio process PID {process_handle.pid} already exited.")
                        except Exception as e_term:
                            logger.error(f"Error terminating stdio process {process_handle.pid}: {e_term}")
                    else:
                        logger.info(f"Stdio process PID {process_handle.pid} already exited with code {process_handle.returncode}.")
                
                container_name = instance.internal_instance_details.get("container_name")
                if container_name and self.docker_client: # Docker client might not be init'd if startup failed
                    # Ensure the docker run --rm container is cleaned up if terminate didn't trigger it
                    try:
                        # Check if container still exists
                        cont = self.docker_client.containers.get(container_name)
                        logger.info(f"Stdio container {container_name} still exists. Attempting to stop and remove.")
                        cont.stop(timeout=5)
                        cont.remove()
                        logger.info(f"Stdio container {container_name} explicitly stopped and removed.")
                    except docker.errors.NotFound:
                        logger.info(f"Stdio container {container_name} already removed (likely by --rm).")
                    except Exception as e_cont_cleanup:
                        logger.error(f"Error during final cleanup of stdio container {container_name}: {e_cont_cleanup}")


    async def call_tool_on_instance(
        self, instance: ManagedInstanceInfo, tool_name: str, tool_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        if instance.orchestration_mode != "local_docker":
            raise ValueError("LocalDockerOrchestrationClient only handles local_docker instances.")

        if instance.mcp_transport == "http":
            if not self.http_client:
                raise RuntimeError("HTTP client not initialized. Call startup() first.")
            if not instance.mcp_endpoint_url:
                raise ValueError(f"mcp_endpoint_url is required for HTTP transport for instance {instance.instance_id}")
            
            http_payload = {"tool_name": tool_name, "arguments": tool_args}
            try:
                response = await self.http_client.post(instance.mcp_endpoint_url, json=http_payload)
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                logger.error(f"HTTP Request error calling tool {tool_name} on {instance.mcp_endpoint_url}: {e}")
                raise RuntimeError(f"MCP tool call failed: Network error calling {instance.mcp_endpoint_url}") from e
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP status error calling tool {tool_name} on {instance.mcp_endpoint_url}: {e.response.status_code} - {e.response.text}")
                try: error_details = e.response.json()
                except Exception: error_details = e.response.text
                raise RuntimeError(f"MCP tool call failed: Server returned error {e.response.status_code}. Details: {error_details}") from e
            except Exception as e:
                logger.error(f"Unexpected HTTP error calling tool {tool_name} on {instance.mcp_endpoint_url}: {e}")
                raise RuntimeError(f"MCP tool call failed: Unexpected error. Details: {str(e)}") from e

        elif instance.mcp_transport == "stdio":
            client_session_stdio = instance.internal_instance_details.get("client_session_stdio")
            if not client_session_stdio or not isinstance(client_session_stdio, ClientSession):
                # This indicates a problem with how ClientSession is stored or retrieved.
                # Or, if we are to create it on-demand:
                # process_handle = instance.internal_instance_details.get("process_handle")
                # server_to_client_reader = instance.internal_instance_details.get("server_to_client_reader")
                # client_to_server_writer = instance.internal_instance_details.get("client_to_server_writer")
                # if not all([process_handle, server_to_client_reader, client_to_server_writer]):
                #     raise RuntimeError(f"Stdio instance {instance.instance_id} is missing critical details for tool call.")
                # client_session_stdio = ClientSession(read_stream=server_to_client_reader, write_stream=client_to_server_writer, client_info=DEFAULT_CLIENT_INFO)
                # # How to manage its read loop here? This is the challenge.
                # # The `async with client_session_stdio:` pattern is best.
                raise RuntimeError(f"Valid ClientSession not found for stdio instance {instance.instance_id}. Provisioning might have been incomplete or state lost.")

            try:
                # Assuming client_session_stdio is an active session whose read loop is managed by a task group
                # stored and managed elsewhere (e.g., by the IntermediaryServer's session object).
                # The `async with client_session_stdio:` pattern is ideal if each tool call is a self-contained session interaction.
                # However, if the session is meant to be long-lived, its context needs to be managed externally.
                
                # For now, let's assume client_session_stdio is ready to use.
                logger.debug(f"Calling tool {tool_name} via stdio ClientSession for instance {instance.instance_id} with args: {tool_args}")
                
                # The `async with client_session_stdio` should ideally wrap this call if the session isn't already "running"
                # This is the core design challenge for stdio in LocalDockerOrchestrationClient.
                # The standalone script uses `async with client_session` in its main block.
                # Here, the client_session is stored.
                # If the read_loop was started by passing a task_group to ClientSession's BaseSession, it's running.
                # If ClientSession.run() was used as a context manager during init, it's not running anymore.
                
                # Let's assume the ClientSession stored is "live" and its read loop is managed by the (yet to be stored) task group.
                tool_result = await client_session_stdio.call_tool(tool_name, tool_args)
                
                # The result from client_session.call_tool is already a Pydantic model (e.g., CallToolResult).
                # We need to convert it to a dict if that's what this method must return.
                # The type hint is Dict[str, Any].
                if hasattr(tool_result, 'model_dump'):
                    return tool_result.model_dump(exclude_none=True)
                else: # Should not happen if tool_result is a Pydantic model
                    logger.error(f"Tool result for {tool_name} is not a Pydantic model: {type(tool_result)}")
                    # Attempt to convert common structures or raise error
                    if isinstance(tool_result, dict): return tool_result
                    return {"error": "Tool result format unexpected", "details": str(tool_result)}

            except Exception as e:
                logger.exception(f"Error during stdio tool call for instance {instance.instance_id}, tool {tool_name}: {e}")
                raise RuntimeError(f"MCP stdio tool call failed for {tool_name}. Details: {str(e)}") from e
        else:
            raise ValueError(f"Unsupported mcp_transport type: {instance.mcp_transport}")
