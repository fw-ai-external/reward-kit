import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Type

from reward_kit.mcp_agent.backends.base import (
    AbstractBackendHandler,
    BackendInitRequest,
    BackendInitResult,
)
from reward_kit.mcp_agent.backends.generic import GenericBackendHandler
# Import specific handlers when they are created, e.g.:
# from reward_kit.mcp_agent.backends.filesystem import FileSystemBackendHandler
# from reward_kit.mcp_agent.backends.duckdb import DuckDBBackendHandler

from reward_kit.mcp_agent.config import AppConfig, BackendServerConfig
from reward_kit.mcp_agent.orchestration.base_client import (
    AbstractOrchestrationClient,
    ManagedInstanceInfo,
)
from reward_kit.mcp_agent.orchestration.local_docker_client import (
    LocalDockerOrchestrationClient,
)
from reward_kit.mcp_agent.orchestration.remote_http_client import (
    RemoteHttpOrchestrationClient,
)
from reward_kit.mcp_agent.session import IntermediarySession # Uses placeholder BaseSession

logger = logging.getLogger(__name__)

# Placeholders for MCP SDK components
# from mcp.server import Server as BaseMcpServer, tool as mcp_tool
# from mcp.server.models import SessionContext # or however session is passed

class SessionContext: # Placeholder
    def __init__(self, session_id: str):
        self.session_id = session_id

def mcp_tool(name: Optional[str] = None): # Placeholder decorator
    def decorator(func):
        func._mcp_tool_name = name or func.__name__
        return func
    return decorator

class BaseMcpServer: # Placeholder
    def __init__(self, config: Optional[Dict[str, Any]] = None, session_class: Type[IntermediarySession] = IntermediarySession):
        self.config = config or {}
        self.sessions: Dict[str, IntermediarySession] = {}
        self.session_class = session_class
        self.app_config: Optional[AppConfig] = None # Will be loaded
        self._local_docker_orchestrator: Optional[LocalDockerOrchestrationClient] = None
        self._remote_http_orchestrators: Dict[str, RemoteHttpOrchestrationClient] = {} # Keyed by remote_api_config_ref
        self._backend_handlers: Dict[str, AbstractBackendHandler] = {} # Keyed by backend_name_ref
        self._shared_global_instances: Dict[str, ManagedInstanceInfo] = {} # Keyed by backend_name_ref

        logger.info("BaseMcpServer (placeholder) initialized.")

    def _get_or_create_session(self, session_id: str) -> IntermediarySession:
        if session_id not in self.sessions:
            self.sessions[session_id] = self.session_class(session_id, app=self)
        return self.sessions[session_id]

    async def startup(self): # Placeholder for server startup
        logger.info("BaseMcpServer (placeholder) starting up...")
        # Actual server would load config, init orchestrators, handlers etc.
        await self._initialize_orchestrators()
        await self._initialize_backend_handlers()
        await self._provision_shared_global_instances()
        logger.info("BaseMcpServer (placeholder) startup complete.")


    async def shutdown(self): # Placeholder for server shutdown
        logger.info("BaseMcpServer (placeholder) shutting down...")
        # Cleanup sessions, orchestrators, shared instances
        
        # Deprovision all session-specific instances first
        for session_id in list(self.sessions.keys()):
            await self.cleanup_session_internal(session_id, self.sessions[session_id]) # Call internal cleanup

        # Then deprovision shared global instances
        shared_instances_to_deprovision = list(self._shared_global_instances.values())
        if shared_instances_to_deprovision:
            logger.info(f"Deprovisioning {len(shared_instances_to_deprovision)} shared global instances.")
            # Need to group by orchestrator
            # This is simplified; a real implementation would group by orchestrator
            if self._local_docker_orchestrator and any(i.orchestration_mode == "local_docker" for i in shared_instances_to_deprovision):
                 await self._local_docker_orchestrator.deprovision_instances([i for i in shared_instances_to_deprovision if i.orchestration_mode == "local_docker"])
            for remote_ref, orch in self._remote_http_orchestrators.items():
                remote_instances = [i for i in shared_instances_to_deprovision if i.orchestration_mode == "remote_http_api" and self.app_config.get_remote_api_config(next(b for b in self.app_config.backends if b.backend_name_ref == i.backend_name_ref)) == orch.app_config.global_remote_apis.get(remote_ref) ] # This condition is complex
                if remote_instances: # Simplified: find correct remote orchestrator
                    await orch.deprovision_instances(remote_instances)


        if self._local_docker_orchestrator:
            await self._local_docker_orchestrator.shutdown()
        for orch in self._remote_http_orchestrators.values():
            await orch.shutdown()
        logger.info("BaseMcpServer (placeholder) shutdown complete.")

    # These would be part of the actual BaseMcpServer or called by it
    async def _initialize_orchestrators(self): pass
    async def _initialize_backend_handlers(self): pass
    async def _provision_shared_global_instances(self): pass
    async def cleanup_session_internal(self, session_id: str, session: IntermediarySession): pass


class RewardKitIntermediaryServer(BaseMcpServer):
    """
    The RewardKit Intermediary MCP Server.
    Manages lifecycles of various backend MCP servers (local Docker or remote)
    and exposes them to clients (e.g., RL rollout workers) through a unified MCP interface.
    """

    def __init__(self, app_config: AppConfig):
        super().__init__(session_class=IntermediarySession)
        self.app_config = app_config
        self._local_docker_orchestrator: Optional[LocalDockerOrchestrationClient] = None
        self._remote_http_orchestrators: Dict[str, RemoteHttpOrchestrationClient] = {} # Keyed by unique remote API base_url or a ref
        self._backend_handlers: Dict[str, AbstractBackendHandler] = {} # Keyed by backend_name_ref
        
        # Cache for shared global instances. Keyed by backend_name_ref.
        self._shared_global_instances: Dict[str, ManagedInstanceInfo] = {}
        self._shared_instance_locks: Dict[str, asyncio.Lock] = {} # For concurrent requests for shared instances

        logger.info("RewardKitIntermediaryServer initialized with AppConfig.")

    async def _initialize_orchestrators(self):
        """Initializes orchestration clients based on AppConfig."""
        logger.info("Initializing orchestration clients...")
        # Check if any backend uses local_docker
        if any(b.orchestration_mode == "local_docker" for b in self.app_config.backends):
            self._local_docker_orchestrator = LocalDockerOrchestrationClient(self.app_config)
            await self._local_docker_orchestrator.startup()
            logger.info("LocalDockerOrchestrationClient initialized and started.")

        # Initialize remote orchestrators for each unique remote API config
        unique_remote_api_refs = set()
        for backend_cfg in self.app_config.backends:
            if backend_cfg.orchestration_mode == "remote_http_api":
                if backend_cfg.remote_api_config_ref:
                    unique_remote_api_refs.add(backend_cfg.remote_api_config_ref)
                elif backend_cfg.remote_api_config_inline:
                    # For inline configs, we might use their base_url or a generated hash as a key
                    # For simplicity, let's assume refs are preferred for shared remote orchestrators
                    logger.warning(f"Inline remote_api_config for {backend_cfg.backend_name_ref}. Consider using global_remote_apis for shared RemoteHttpOrchestrationClient instances.")
                    # Create a dedicated client for this inline config if not already based on base_url
                    key = backend_cfg.remote_api_config_inline.base_url 
                    if key not in self._remote_http_orchestrators:
                        temp_app_config_for_inline = AppConfig(global_remote_apis={key: backend_cfg.remote_api_config_inline}) # Mock AppConfig for this client
                        client = RemoteHttpOrchestrationClient(temp_app_config_for_inline) # This is a bit hacky
                        await client.startup()
                        self._remote_http_orchestrators[key] = client # Store by base_url
                        logger.info(f"RemoteHttpOrchestrationClient for inline config {key} initialized and started.")


        for ref_name in unique_remote_api_refs:
            if ref_name not in self.app_config.global_remote_apis:
                logger.error(f"Remote API reference '{ref_name}' not found in global_remote_apis configuration.")
                continue
            if ref_name not in self._remote_http_orchestrators: # Ensure only one client per referenced config
                # Create a temporary AppConfig that isolates the specific global_remote_api for this client
                # This is to ensure RemoteHttpOrchestrationClient correctly picks up its specific config.
                # A bit of a workaround due to RemoteHttpOrchestrationClient taking the whole AppConfig.
                isolated_app_cfg = AppConfig(global_remote_apis={ref_name: self.app_config.global_remote_apis[ref_name]}, global_remote_api_defaults=self.app_config.global_remote_api_defaults)

                client = RemoteHttpOrchestrationClient(isolated_app_cfg)
                await client.startup()
                self._remote_http_orchestrators[ref_name] = client
                logger.info(f"RemoteHttpOrchestrationClient for '{ref_name}' initialized and started.")
        logger.info("Orchestration clients initialization complete.")


    def _get_orchestration_client(self, backend_cfg: BackendServerConfig) -> AbstractOrchestrationClient:
        if backend_cfg.orchestration_mode == "local_docker":
            if not self._local_docker_orchestrator:
                raise RuntimeError("Local Docker orchestrator not initialized.")
            return self._local_docker_orchestrator
        elif backend_cfg.orchestration_mode == "remote_http_api":
            key = backend_cfg.remote_api_config_ref
            if not key: # Inline config
                if backend_cfg.remote_api_config_inline:
                    key = backend_cfg.remote_api_config_inline.base_url
                else: # Should be caught by pydantic model validation
                     raise ValueError(f"Remote API config missing for {backend_cfg.backend_name_ref}")

            client = self._remote_http_orchestrators.get(key)
            if not client:
                raise RuntimeError(f"Remote HTTP orchestrator for '{key}' not initialized.")
            return client
        else:
            raise ValueError(f"Unsupported orchestration mode: {backend_cfg.orchestration_mode}")

    async def _initialize_backend_handlers(self):
        """Initializes backend handlers based on AppConfig."""
        logger.info("Initializing backend handlers...")
        for backend_cfg in self.app_config.backends:
            handler_class: Type[AbstractBackendHandler]
            # TODO: Add specific handlers as they are implemented
            # if backend_cfg.backend_type == "filesystem":
            #     handler_class = FileSystemBackendHandler
            # elif backend_cfg.backend_type == "duckdb":
            #     handler_class = DuckDBBackendHandler
            # else:
            # For now, use GenericBackendHandler for all
            handler_class = GenericBackendHandler
            
            self._backend_handlers[backend_cfg.backend_name_ref] = handler_class(backend_cfg)
            self._shared_instance_locks[backend_cfg.backend_name_ref] = asyncio.Lock() # Lock per backend_name_ref
            logger.info(f"Initialized {handler_class.__name__} for backend_name_ref '{backend_cfg.backend_name_ref}'.")
        logger.info("Backend handlers initialization complete.")

    async def _get_or_provision_shared_global_instance(self, backend_name_ref: str) -> ManagedInstanceInfo:
        """Gets or provisions a shared global instance for a given backend_name_ref."""
        async with self._shared_instance_locks[backend_name_ref]:
            if backend_name_ref in self._shared_global_instances:
                logger.info(f"Returning existing shared global instance for '{backend_name_ref}'.")
                return self._shared_global_instances[backend_name_ref]

            logger.info(f"Provisioning new shared global instance for '{backend_name_ref}'.")
            backend_cfg = next((b for b in self.app_config.backends if b.backend_name_ref == backend_name_ref), None)
            if not backend_cfg or backend_cfg.instance_scoping != "shared_global":
                raise ValueError(f"Backend '{backend_name_ref}' is not configured for shared_global scoping.")

            orchestration_client = self._get_orchestration_client(backend_cfg)
            handler = self._backend_handlers[backend_name_ref] # Should exist

            # For shared global, num_instances is 1. template_details might come from config.
            # The handler's initialize_session_instances is session-oriented. We need a more direct way.
            # Let's call orchestration_client.provision_instances directly for shared.
            # Session ID for shared instances can be a global/static one.
            shared_session_id = "global_shared_session"
            
            # template_details for shared instances should ideally come from their BackendServerConfig
            # or be None if not applicable.
            # For now, passing template_details from BackendServerConfig if available.
            template_details_for_shared = backend_cfg.template_data_path_host # Example for local_docker

            provisioned_list = await orchestration_client.provision_instances(
                backend_config=backend_cfg,
                num_instances=1,
                session_id=shared_session_id, # Use a special session_id for global
                template_details=template_details_for_shared
            )
            if not provisioned_list:
                raise RuntimeError(f"Failed to provision shared global instance for '{backend_name_ref}'.")
            
            instance_info = provisioned_list[0]
            self._shared_global_instances[backend_name_ref] = instance_info
            logger.info(f"Provisioned and cached shared global instance for '{backend_name_ref}': {instance_info.instance_id}")
            return instance_info

    async def _provision_shared_global_instances(self):
        """Provisions all configured shared_global instances at server startup."""
        logger.info("Pre-provisioning all shared_global instances...")
        for backend_cfg in self.app_config.backends:
            if backend_cfg.instance_scoping == "shared_global":
                try:
                    await self._get_or_provision_shared_global_instance(backend_cfg.backend_name_ref)
                except Exception as e:
                    logger.error(f"Failed to pre-provision shared global instance for '{backend_cfg.backend_name_ref}': {e}", exc_info=True)
                    # Depending on policy, server startup might fail or continue with this backend unavailable.
        logger.info("Shared_global instances pre-provisioning complete.")


    @mcp_tool(name="initialize_session")
    async def initialize_session(
        self, ctx: SessionContext, backends: List[BackendInitRequest]
    ) -> Dict[str, Any]:
        """
        Initializes a new session with the requested backend instances.
        If a session_id is provided in ctx, it tries to reuse/verify it.
        Otherwise, a new session_id is generated.
        """
        session_id = ctx.session_id if ctx and ctx.session_id else f"session-{uuid.uuid4().hex[:12]}"
        session = self._get_or_create_session(session_id) # Ensures session object exists
        
        logger.info(f"Initializing session '{session_id}' with {len(backends)} backend requests.")
        initialized_backends_results: List[BackendInitResult] = []

        for backend_req in backends:
            backend_cfg = next((b for b in self.app_config.backends if b.backend_name_ref == backend_req.backend_name_ref), None)
            if not backend_cfg:
                logger.error(f"Session {session_id}: Backend config with ref_name '{backend_req.backend_name_ref}' not found.")
                # Optionally return partial success or fail the whole request
                # For now, skip this backend and continue. Client should check results.
                # Or, more strictly: raise ValueError(f"Backend config '{backend_req.backend_name_ref}' not found.")
                initialized_backends_results.append(BackendInitResult(backend_name_ref=backend_req.backend_name_ref, instances=[])) # Indicate failure for this one
                continue

            try:
                if backend_cfg.instance_scoping == "shared_global":
                    logger.info(f"Session {session_id}: Request for shared_global backend '{backend_req.backend_name_ref}'. Getting or ensuring shared instance.")
                    # For shared_global, we provide a handle to the single shared instance.
                    # num_instances in request is how many handles client wants.
                    shared_instance_info = await self._get_or_provision_shared_global_instance(backend_req.backend_name_ref)
                    # Client gets `num_instances` copies of the same ManagedInstanceInfo if they asked for more than 1.
                    # This might be confusing. Better to always return 1 for shared_global.
                    # Let's adjust: if shared_global, result always has 1 instance.
                    instances_for_this_backend = [shared_instance_info] * backend_req.num_instances # Or just [shared_instance_info]
                    logger.info(f"Session {session_id}: Provided {len(instances_for_this_backend)} handle(s) to shared instance for '{backend_req.backend_name_ref}'.")

                else: # "session" scoped instances
                    handler = self._backend_handlers.get(backend_req.backend_name_ref)
                    if not handler:
                        raise ValueError(f"No backend handler found for '{backend_req.backend_name_ref}'.")
                    
                    orchestration_client = self._get_orchestration_client(backend_cfg)
                    
                    logger.info(f"Session {session_id}: Delegating to handler for '{backend_req.backend_name_ref}'.")
                    instances_for_this_backend = await handler.initialize_session_instances(
                        session=session,
                        init_request=backend_req,
                        orchestration_client=orchestration_client,
                    )
                    session.add_managed_instances(backend_req.backend_name_ref, instances_for_this_backend)
                
                initialized_backends_results.append(
                    BackendInitResult(
                        backend_name_ref=backend_req.backend_name_ref,
                        instances=instances_for_this_backend,
                    )
                )

            except Exception as e:
                logger.error(f"Session {session_id}: Error initializing backend '{backend_req.backend_name_ref}': {e}", exc_info=True)
                # Add a result indicating failure for this specific backend
                initialized_backends_results.append(BackendInitResult(backend_name_ref=backend_req.backend_name_ref, instances=[]))


        return {
            "session_id": session.session_id,
            "initialized_backends": [res.model_dump() for res in initialized_backends_results],
        }

    @mcp_tool(name="call_backend_tool")
    async def call_backend_tool(
        self,
        ctx: SessionContext, # MCP framework should provide this with session_id
        backend_name_ref: str,
        instance_id: str, # Client specifies which instance within the session
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not ctx or not ctx.session_id:
            raise ValueError("Session context with session_id is required.")
        session = self.sessions.get(ctx.session_id)
        if not session:
            raise ValueError(f"Session '{ctx.session_id}' not found.")

        logger.debug(f"Session {session.session_id}: Call tool '{tool_name}' on backend '{backend_name_ref}', instance '{instance_id}'.")
        
        target_instances = session.get_managed_instances(backend_name_ref, instance_id)
        if not target_instances:
            raise ValueError(f"Instance '{instance_id}' for backend '{backend_name_ref}' not found in session '{session.session_id}'.")
        
        # Assuming instance_id is unique within a backend_name_ref for a session, so target_instances[0] is fine.
        managed_instance_info = target_instances[0]

        backend_cfg = next((b for b in self.app_config.backends if b.backend_name_ref == backend_name_ref), None)
        if not backend_cfg:
             raise ValueError(f"Backend config '{backend_name_ref}' not found.") # Should not happen if instance exists

        orchestration_client = self._get_orchestration_client(backend_cfg)
        
        try:
            result = await orchestration_client.call_tool_on_instance(
                instance=managed_instance_info, tool_name=tool_name, tool_args=tool_args
            )
            logger.debug(f"Session {session.session_id}: Tool '{tool_name}' on instance '{instance_id}' successful.")
            return result
        except Exception as e:
            logger.error(f"Session {session.session_id}: Error calling tool '{tool_name}' on instance '{instance_id}': {e}", exc_info=True)
            # Re-raise to propagate error to client. Error structure might need to be MCP compliant.
            raise # Or return a structured error: e.g., {"error": str(e), "details": ...}


    async def cleanup_session_internal(self, session_id: str, session: IntermediarySession):
        """Internal method to handle actual resource cleanup for a session."""
        logger.info(f"Starting internal cleanup for session '{session_id}'.")
        all_session_instances = session.get_all_managed_instances()
        
        # Group instances by orchestrator to batch deprovisioning calls
        # This is a simplified grouping; a more robust way would be to store orchestrator ref on ManagedInstanceInfo
        # or re-derive it.
        
        local_docker_instances = [inst for inst in all_session_instances if inst.orchestration_mode == "local_docker"]
        if local_docker_instances and self._local_docker_orchestrator:
            logger.info(f"Session {session_id}: Deprovisioning {len(local_docker_instances)} local Docker instances.")
            try:
                await self._local_docker_orchestrator.deprovision_instances(local_docker_instances)
            except Exception as e:
                logger.error(f"Session {session_id}: Error deprovisioning local Docker instances: {e}", exc_info=True)

        # For remote instances, group by the remote_api_config_ref or equivalent key
        remote_instances_by_orchestrator_key: Dict[str, List[ManagedInstanceInfo]] = {}
        for inst in all_session_instances:
            if inst.orchestration_mode == "remote_http_api":
                backend_cfg = next((b for b in self.app_config.backends if b.backend_name_ref == inst.backend_name_ref), None)
                if backend_cfg:
                    key = backend_cfg.remote_api_config_ref or (backend_cfg.remote_api_config_inline.base_url if backend_cfg.remote_api_config_inline else None)
                    if key:
                        if key not in remote_instances_by_orchestrator_key:
                            remote_instances_by_orchestrator_key[key] = []
                        remote_instances_by_orchestrator_key[key].append(inst)
        
        for key, remote_instances_list in remote_instances_by_orchestrator_key.items():
            orchestrator = self._remote_http_orchestrators.get(key)
            if orchestrator and remote_instances_list:
                logger.info(f"Session {session_id}: Deprovisioning {len(remote_instances_list)} remote instances for orchestrator '{key}'.")
                try:
                    await orchestrator.deprovision_instances(remote_instances_list)
                except Exception as e:
                    logger.error(f"Session {session_id}: Error deprovisioning remote instances for '{key}': {e}", exc_info=True)
        
        # Cleanup temporary Docker images associated with the session (if using local Docker)
        if session.temporary_docker_images and self._local_docker_orchestrator:
            logger.info(f"Session {session_id}: Cleaning up {len(session.temporary_docker_images)} temporary Docker images.")
            for image_tag in list(session.temporary_docker_images): # Iterate copy
                try:
                    # The LocalDockerOrchestrationClient's shutdown handles its _temporary_images set.
                    # This ensures images are removed if the session is explicitly cleaned before server shutdown.
                    # We need to ensure the orchestrator knows about these images if it didn't create them directly
                    # (though it should have if it returned committed_image_tag).
                    # For now, let LocalDockerOrchestrationClient.shutdown manage its own list.
                    # If a session ends, its temporary images should be removed if no other session uses them.
                    # This logic is tricky. A reference count on images or more careful management in LocalDockerClient is needed.
                    # For now, we assume LocalDockerClient's _temporary_images set is the source of truth for its cleanup.
                    # If session.temporary_docker_images contains tags, we can try to remove them.
                    logger.info(f"Session {session_id}: Requesting removal of temporary image '{image_tag}'.")
                    self._local_docker_orchestrator.docker_client.images.remove(image=image_tag, force=False) # Best effort
                    session.temporary_docker_images.discard(image_tag)
                    self._local_docker_orchestrator._temporary_images.discard(image_tag) # Also remove from orchestrator's list
                except docker.errors.ImageNotFound:
                     logger.warning(f"Session {session_id}: Temporary image {image_tag} not found for removal.")
                except docker.errors.APIError as e:
                    if e.response.status_code == 409: # Conflict
                        logger.warning(f"Session {session_id}: Temporary image {image_tag} is in use, not removed.")
                    else:
                        logger.error(f"Session {session_id}: Failed to remove temporary image {image_tag}: {e}")
        
        logger.info(f"Internal cleanup for session '{session_id}' complete.")


    @mcp_tool(name="cleanup_session")
    async def cleanup_session(self, ctx: SessionContext) -> Dict[str, str]:
        """
        Cleans up all resources associated with the given session_id.
        This involves deprovisioning all backend instances created for the session.
        """
        if not ctx or not ctx.session_id:
            raise ValueError("Session context with session_id is required.")
        
        session_id_to_clean = ctx.session_id
        session = self.sessions.pop(session_id_to_clean, None)

        if not session:
            logger.warning(f"Cleanup requested for non-existent or already cleaned session '{session_id_to_clean}'.")
            return {"status": "session_not_found_or_already_cleaned", "session_id": session_id_to_clean}

        logger.info(f"Cleaning up resources for session '{session_id_to_clean}'.")
        await self.cleanup_session_internal(session_id_to_clean, session)
        
        # Call the session object's own cleanup hook (from BaseSession)
        await session.cleanup()

        logger.info(f"Session '{session_id_to_clean}' fully cleaned up.")
        return {"status": "cleaned", "session_id": session_id_to_clean}

    # Actual server startup/shutdown would be handled by an ASGI server like Uvicorn
    # For now, adding explicit methods that an external runner could call.
    async def start(self):
        """Complete server startup sequence."""
        logging.basicConfig(level=self.app_config.log_level.upper())
        logger.info("RewardKitIntermediaryServer starting...")
        await self.startup() # Calls the placeholder BaseMcpServer startup
        logger.info("RewardKitIntermediaryServer running.")

    async def stop(self):
        """Complete server shutdown sequence."""
        logger.info("RewardKitIntermediaryServer stopping...")
        await self.shutdown() # Calls placeholder BaseMcpServer shutdown
        logger.info("RewardKitIntermediaryServer stopped.")
