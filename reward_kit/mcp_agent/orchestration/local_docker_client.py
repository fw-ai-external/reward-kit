import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Set

import docker
import docker.errors
import docker.models.containers
import httpx
from docker.transport import UnixHTTPAdapter

from reward_kit.mcp_agent.config import AppConfig, BackendServerConfig
from reward_kit.mcp_agent.orchestration.base_client import (
    AbstractOrchestrationClient,
    ManagedInstanceInfo,
)

logger = logging.getLogger(__name__)


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
            adapter = UnixHTTPAdapter(pool_connections=50) # Default is 10
            self.docker_client = docker.DockerClient(base_url="unix://var/run/docker.sock", low_api_version="1.39")
            self.docker_client.api.mount(prefix="http+docker://localhost", adapter=adapter)

            if not self.docker_client.ping():
                logger.error(
                    "Failed to connect to Docker daemon. Ensure Docker is running and accessible."
                )
                raise ConnectionError("Failed to connect to Docker daemon.")
            logger.info("Successfully connected to Docker daemon.")
        except docker.errors.DockerException as e:
            logger.error(f"Docker client initialization failed: {e}")
            raise ConnectionError(f"Docker client initialization failed: {e}") from e

        self.http_client = httpx.AsyncClient(timeout=self.app_config.global_remote_api_defaults.get("timeout", 30.0))
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
            
            # Let Docker assign a random available host port
            # The key is the container port, value 0 means assign a random host port.
            port_bindings = {f"{backend_config.container_port}/tcp": 0}

            try:
                logger.info(f"Starting instance container {container_name} from image {image_to_run_instances_from}")
                container = self.docker_client.containers.run(
                    image=image_to_run_instances_from,
                    name=container_name,
                    detach=True,
                    ports=port_bindings,
                    labels={
                        "rewardkit-mcp-session-id": session_id,
                        "rewardkit-mcp-backend-name": backend_config.backend_name_ref,
                        "rewardkit-mcp-instance-id": instance_uuid,
                        "rewardkit-mcp-managed": "true",
                    },
                    **(self.app_config.global_docker_options or {}),
                    **(backend_config.docker_run_args or {}),
                )

                # Retrieve the dynamically assigned host port
                container.reload() # Ensure port information is up-to-date
                host_port = None
                # Docker SDK returns a list of port bindings. Find the one for our container_port.
                # Example: {'23000/tcp': [{'HostIp': '0.0.0.0', 'HostPort': '32769'}, {'HostIp': '::', 'HostPort': '32769'}]}
                # We need the HostPort.
                bindings = container.attrs['NetworkSettings']['Ports'].get(f"{backend_config.container_port}/tcp")
                if bindings and len(bindings) > 0:
                    host_port = int(bindings[0]['HostPort'])
                
                if not host_port:
                    container.stop(timeout=5)
                    container.remove()
                    raise RuntimeError(f"Failed to get assigned host port for container {container_name}")

                self._used_host_ports.add(host_port) # Track port usage

                mcp_endpoint_url = f"http://localhost:{host_port}/mcp" # Assuming /mcp path

                if backend_config.startup_check_mcp_tool:
                    if not await self._perform_startup_check(
                        mcp_endpoint_url, backend_config.startup_check_mcp_tool
                    ):
                        logger.error(f"Startup check failed for container {container_name} at {mcp_endpoint_url}. Stopping and removing.")
                        container.stop(timeout=5)
                        container.remove()
                        self._used_host_ports.discard(host_port)
                        # If one instance fails startup, should we abort all? For now, continue and report failures.
                        # Or raise an exception to halt provisioning for this backend_config.
                        raise RuntimeError(f"Startup check failed for {container_name}")
                
                logger.info(f"Instance container {container_name} (ID: {container.id}) running on host port {host_port}. MCP Endpoint: {mcp_endpoint_url}")

                managed_instances.append(
                    ManagedInstanceInfo(
                        instance_id=instance_uuid,
                        backend_name_ref=backend_config.backend_name_ref,
                        orchestration_mode="local_docker",
                        mcp_endpoint_url=mcp_endpoint_url,
                        internal_instance_details={
                            "container_id": container.id,
                            "container_name": container_name,
                            "host_port": host_port,
                        },
                        committed_image_tag=committed_image_tag_for_session if backend_config.instance_scoping == "session" else None,
                    )
                )
            except docker.errors.APIError as e:
                logger.error(f"Failed to start instance container {container_name}: {e}")
                # Cleanup any successfully started instances for this batch if one fails?
                # For now, we'll let deprovision_instances handle cleanup of what was created.
                raise # Re-raise to signal failure
            except Exception as e:
                logger.error(f"An unexpected error occurred while provisioning {container_name}: {e}")
                raise


        return managed_instances

    async def deprovision_instances(self, instances: List[ManagedInstanceInfo]) -> None:
        if not self.docker_client:
            logger.warning("Docker client not initialized. Cannot deprovision.")
            return

        for instance in instances:
            if instance.orchestration_mode != "local_docker":
                logger.warning(
                    f"Skipping deprovision for instance {instance.instance_id} as it's not local_docker."
                )
                continue

            container_id = instance.internal_instance_details.get("container_id")
            if not container_id:
                logger.warning(
                    f"No container_id found for instance {instance.instance_id}. Cannot deprovision."
                )
                continue

            try:
                container = self.docker_client.containers.get(container_id)
                logger.info(f"Stopping container {container.name} (ID: {container_id})...")
                container.stop(timeout=10) # Graceful stop
                logger.info(f"Removing container {container.name} (ID: {container_id})...")
                container.remove()
                logger.info(f"Container {container_id} stopped and removed.")
                
                host_port = instance.internal_instance_details.get("host_port")
                if host_port:
                    self._used_host_ports.discard(host_port)

            except docker.errors.NotFound:
                logger.warning(
                    f"Container {container_id} for instance {instance.instance_id} not found during deprovision."
                )
            except docker.errors.APIError as e:
                logger.error(
                    f"API error deprovisioning container {container_id} for instance {instance.instance_id}: {e}"
                )
            
            # Committed image tag cleanup is handled in shutdown or by a higher-level session manager
            # to ensure an image isn't removed if other instances from the same session still use it.
            # The self._temporary_images set is cleaned in shutdown().
            # If an image was specific to this instance (not typical for committed templates), it could be removed here.
            # For now, committed_image_tag on ManagedInstanceInfo is mostly for tracking.

    async def call_tool_on_instance(
        self, instance: ManagedInstanceInfo, tool_name: str, tool_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.http_client:
            raise RuntimeError("HTTP client not initialized. Call startup() first.")
        if instance.orchestration_mode != "local_docker":
            raise ValueError("This client only handles local_docker instances.")

        mcp_payload = {"tool_name": tool_name, "arguments": tool_args}
        try:
            logger.debug(
                f"Calling tool {tool_name} on {instance.mcp_endpoint_url} with args: {tool_args}"
            )
            response = await self.http_client.post(
                instance.mcp_endpoint_url, json=mcp_payload
            )
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logger.error(
                f"Request error calling tool {tool_name} on {instance.mcp_endpoint_url}: {e}"
            )
            raise RuntimeError(f"MCP tool call failed: Network error calling {instance.mcp_endpoint_url}") from e
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP status error calling tool {tool_name} on {instance.mcp_endpoint_url}: {e.response.status_code} - {e.response.text}"
            )
            # Attempt to parse error response if JSON
            try:
                error_details = e.response.json()
            except Exception:
                error_details = e.response.text
            raise RuntimeError(f"MCP tool call failed: Server returned error {e.response.status_code}. Details: {error_details}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error calling tool {tool_name} on {instance.mcp_endpoint_url}: {e}"
            )
            raise RuntimeError(f"MCP tool call failed: Unexpected error. Details: {str(e)}") from e
