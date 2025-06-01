import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Type, Callable

from reward_kit.mcp_agent.backends.base import (
    AbstractBackendHandler,
    BackendInitRequest,
    BackendInitResult,
)
from reward_kit.mcp_agent.backends.generic import GenericBackendHandler

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
from reward_kit.mcp_agent.session import IntermediarySessionData

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from mcp.server.fastmcp.server import FastMCP
from mcp.server.fastmcp.server import Context as FastMCPContext
from mcp.shared.context import RequestContext 

class RewardKitIntermediaryServer(FastMCP):
    """
    The RewardKit Intermediary MCP Server, now based on FastMCP.
    Manages lifecycles of various backend MCP servers (local Docker or remote)
    and exposes them to clients (e.g., RL rollout workers) through a unified MCP interface.
    """

    def __init__(self, app_config: AppConfig, **kwargs_for_fastmcp):
        super().__init__(
            name="RewardKitIntermediaryMCP",
            instructions="Intermediary Server for managing backend MCP resources for RewardKit RL rollouts.",
            **kwargs_for_fastmcp
        )
        
        self.app_config = app_config
        
        self._local_docker_orchestrator: Optional[LocalDockerOrchestrationClient] = None
        self._remote_http_orchestrators: Dict[str, RemoteHttpOrchestrationClient] = {}
        self._backend_handlers: Dict[str, AbstractBackendHandler] = {}
        
        self._shared_global_instances: Dict[str, ManagedInstanceInfo] = {}
        self._shared_instance_locks: Dict[str, asyncio.Lock] = {}

        # Map to store our custom session data, keyed by the transport session_id
        self.intermediary_session_data: Dict[str, IntermediarySessionData] = {} 
        
        logger.info("RewardKitIntermediaryServer (FastMCP based) initialized. AppConfig loaded.")

        self._internal_tool_handlers: Dict[str, Callable] = {
            "initialize_session": self._initialize_session_actual,
            "call_backend_tool": self._call_backend_tool_actual,
            "cleanup_session": self._cleanup_session_actual,
            "ping": self._ping_actual,
        }

        # Register the single generic proxy tool handler using FastMCP's add_tool method
        self.add_tool(
            self._execute_proxied_tool_impl,
            name="execute_proxied_tool"
        )
        
        logger.info("Registered single proxy tool 'execute_proxied_tool' using self.add_tool().")

    async def _execute_proxied_tool_impl(
        self, 
        mcp_ctx: FastMCPContext, # FastMCP should inject its Context here
        actual_tool_name: str, 
        actual_tool_args: Dict[str, Any]
    ) -> Any:
        """
        Generic handler for all proxied tool calls.
        Invoked by FastMCP's ToolManager.
        `mcp_ctx` is the FastMCP.Context object.
        `actual_tool_name` is the name of the tool the client *actually* wants to call.
        `actual_tool_args` are the arguments for that tool.
        """
        logger.debug(
            f"Proxy handler _execute_proxied_tool_impl called. "
            f"FastMCPContext type: {type(mcp_ctx)}, "
            f"mcp_ctx dir: {dir(mcp_ctx)}, "
            f"Actual tool name: '{actual_tool_name}', "
            f"Actual tool args: {actual_tool_args}"
        )
        if hasattr(mcp_ctx, 'session'):
            logger.debug(f"mcp_ctx.session type: {type(mcp_ctx.session)}, mcp_ctx.session dir: {dir(mcp_ctx.session)}")
            if hasattr(mcp_ctx.session, '_session_id'):
                logger.debug(f"mcp_ctx.session._session_id: {mcp_ctx.session._session_id}")
            else:
                logger.debug("mcp_ctx.session does not have _session_id attribute.")
        else:
            logger.debug("mcp_ctx does not have session attribute.")


        # Get the underlying MCPServer.RequestContext from FastMCPContext
        low_level_ctx: RequestContext = mcp_ctx.request_context 
        if not isinstance(low_level_ctx, RequestContext):
            logger.error(f"CRITICAL: mcp_ctx.request_context is not of type RequestContext. Type: {type(low_level_ctx)}. Value: {low_level_ctx}")
            raise TypeError(f"Expected RequestContext from mcp_ctx.request_context, got {type(low_level_ctx)}")

        if not actual_tool_name:
            logger.error(f"Call to execute_proxied_tool missing 'actual_tool_name'. Args received: actual_tool_name={actual_tool_name}, actual_tool_args={actual_tool_args}")
            raise ValueError("Proxied tool call must contain 'actual_tool_name'.")
        
        if actual_tool_args is None:
            actual_tool_args = {}

        handler_method = self._internal_tool_handlers.get(actual_tool_name)
        if not handler_method:
            logger.error(f"Unknown actual_tool_name '{actual_tool_name}' requested via proxy. Args: {actual_tool_args}")
            raise ValueError(f"Tool '{actual_tool_name}' not found (extracted from proxied call).")

        logger.info(f"Proxying call to internal handler for '{actual_tool_name}' with args: {actual_tool_args}")
        return await handler_method(low_level_ctx, actual_tool_args)


    # _get_or_create_session (custom session logic) is handled within each tool handler method below.

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
        logger.info("Shared_global instances pre-provisioning complete.")

    async def _initialize_session_actual(
        self, ctx: RequestContext, tool_args_dict: Dict[str, Any] # tool_args_dict contains 'backends'
    ) -> Dict[str, Any]:
        """
        Initializes a new session with the requested backend instances.
        Generates a new rk_session_id.
        """
        logger.debug(f"_initialize_session_actual called. ctx type: {type(ctx)}, tool_args_dict: {tool_args_dict}")
        if not isinstance(ctx, RequestContext):
            logger.error(f"CRITICAL: ctx is not RequestContext in _initialize_session_actual. Type: {type(ctx)}.")
            raise TypeError(f"Expected RequestContext for ctx, got {type(ctx)}")

        backends: Optional[List[BackendInitRequest]] = tool_args_dict.get("backends")
        if backends is None: # Or further type check if needed
            logger.error("'_initialize_session_actual' called without 'backends' in tool_args_dict.")
            raise ValueError("'backends' argument missing for initialize_session.")

        rk_session_id = uuid.uuid4().hex
        logger.info(f"Generated new rk_session_id: {rk_session_id}")

        session_data = IntermediarySessionData(session_id=rk_session_id)
        self.intermediary_session_data[rk_session_id] = session_data
        
        logger.info(f"Initializing IntermediarySessionData for rk_session_id '{rk_session_id}' with {len(backends)} backend requests.")
        initialized_backends_results: List[BackendInitResult] = []

        for backend_req_data in backends: # backend_req_data is a dict here
            if isinstance(backend_req_data, dict):
                backend_req = BackendInitRequest(**backend_req_data)
            else:
                backend_req = backend_req_data

            backend_cfg = next((b for b in self.app_config.backends if b.backend_name_ref == backend_req.backend_name_ref), None)
            if not backend_cfg:
                logger.error(f"Session {rk_session_id}: Backend config with ref_name '{backend_req.backend_name_ref}' not found.")
                initialized_backends_results.append(BackendInitResult(backend_name_ref=backend_req.backend_name_ref, instances=[]))
                continue

            try:
                if backend_cfg.instance_scoping == "shared_global":
                    logger.info(f"Session {rk_session_id}: Request for shared_global backend '{backend_req.backend_name_ref}'.")
                    shared_instance_info = await self._get_or_provision_shared_global_instance(backend_req.backend_name_ref)
                    instances_for_this_backend = [shared_instance_info] * backend_req.num_instances
                    logger.info(f"Session {rk_session_id}: Provided {len(instances_for_this_backend)} handle(s) to shared instance for '{backend_req.backend_name_ref}'.")
                else: 
                    handler = self._backend_handlers.get(backend_req.backend_name_ref)
                    if not handler:
                        raise ValueError(f"No backend handler found for '{backend_req.backend_name_ref}'.")
                    
                    orchestration_client = self._get_orchestration_client(backend_cfg)
                    logger.info(f"Session {rk_session_id}: Delegating to handler for '{backend_req.backend_name_ref}'.")
                    instances_for_this_backend = await handler.initialize_session_instances(
                        session_data=session_data, 
                        init_request=backend_req,
                        orchestration_client=orchestration_client,
                    )
                
                # Store the provisioned instances in the session data
                session_data.add_managed_instances(backend_req.backend_name_ref, instances_for_this_backend)

                initialized_backends_results.append(
                    BackendInitResult(
                        backend_name_ref=backend_req.backend_name_ref,
                        instances=instances_for_this_backend,
                    )
                )
            except Exception as e:
                logger.error(f"Session {rk_session_id}: Error initializing backend '{backend_req.backend_name_ref}': {e}", exc_info=True)
                initialized_backends_results.append(BackendInitResult(backend_name_ref=backend_req.backend_name_ref, instances=[], error_message=str(e)))

        return {
            "rk_session_id": rk_session_id, # Return the newly generated rk_session_id
            "initialized_backends": [res.model_dump(exclude_none=True) for res in initialized_backends_results],
        }

    async def _call_backend_tool_actual(
        self,
        ctx: RequestContext,
        tool_args_dict: Dict[str, Any] 
    ) -> Dict[str, Any]:
        logger.debug(f"_call_backend_tool_actual called. ctx type: {type(ctx)}, tool_args_dict: {tool_args_dict}")
        if not isinstance(ctx, RequestContext):
            logger.error(f"CRITICAL: ctx is not RequestContext in _call_backend_tool_actual. Type: {type(ctx)}.")
            raise TypeError(f"Expected RequestContext for ctx, got {type(ctx)}")

        rk_session_id = tool_args_dict.get("rk_session_id")
        backend_name_ref = tool_args_dict.get("backend_name_ref")
        instance_id = tool_args_dict.get("instance_id")
        tool_name = tool_args_dict.get("tool_name")
        tool_args = tool_args_dict.get("tool_args")

        if not all([rk_session_id, backend_name_ref, instance_id, tool_name]): # tool_args can be empty
            missing_args = [k for k,v in {"rk_session_id":rk_session_id, "backend_name_ref":backend_name_ref, "instance_id":instance_id, "tool_name":tool_name}.items() if not v]
            logger.error(f"'_call_backend_tool_actual' missing required arguments: {missing_args}. Received: {tool_args_dict}")
            raise ValueError(f"Missing required arguments for call_backend_tool: {missing_args}")

        session_data = self.intermediary_session_data.get(rk_session_id)
        if not session_data: 
            logger.error(f"IntermediarySessionData for rk_session_id '{rk_session_id}' not found.")
            raise ValueError(f"IntermediarySessionData for rk_session_id '{rk_session_id}' not found.")
        logger.debug(f"IntermediarySessionData for rk_session_id {rk_session_id}: Call tool '{tool_name}' on backend '{backend_name_ref}', instance '{instance_id}'.")
        
        target_instances = session_data.get_managed_instances(backend_name_ref, instance_id)
        if not target_instances:
            raise ValueError(f"Instance '{instance_id}' for backend '{backend_name_ref}' not found in session '{rk_session_id}'.")
        
        managed_instance_info = target_instances[0]

        backend_cfg = next((b for b in self.app_config.backends if b.backend_name_ref == backend_name_ref), None)
        if not backend_cfg:
             raise ValueError(f"Backend config '{backend_name_ref}' not found.")

        orchestration_client = self._get_orchestration_client(backend_cfg)
        
        try:
            result = await orchestration_client.call_tool_on_instance(
                instance=managed_instance_info, tool_name=tool_name, tool_args=tool_args
            )
            logger.debug(f"Session {rk_session_id}: Tool '{tool_name}' on instance '{instance_id}' successful.")
            return result
        except Exception as e:
            logger.error(f"Session {rk_session_id}: Error calling tool '{tool_name}' on instance '{instance_id}': {e}", exc_info=True)
            raise


    async def cleanup_session_internal(self, session_data_to_clean: IntermediarySessionData, rk_session_id: str):
        """Internal method to handle actual resource cleanup for a session, using its rk_session_id and the direct session_data object."""
        logger.info(f"Starting internal cleanup for IntermediarySessionData (rk_session_id: '{rk_session_id}').")
        
        # session_data_to_clean is now passed directly, no need to fetch from self.intermediary_session_data
        # if not session_data_to_clean: # This check is now done by the caller before pop
        #     logger.warning(f"IntermediarySessionData for rk_session_id '{rk_session_id}' not found for internal cleanup. Already cleaned?")
        #     return

        all_session_instances = session_data_to_clean.get_all_managed_instances()
        
        # Group instances by orchestrator
        local_docker_instances = [inst for inst in all_session_instances if inst.orchestration_mode == "local_docker"]
        if local_docker_instances and self._local_docker_orchestrator:
            logger.info(f"Session {rk_session_id}: Deprovisioning {len(local_docker_instances)} local Docker instances.")
            try:
                await self._local_docker_orchestrator.deprovision_instances(local_docker_instances)
            except Exception as e:
                logger.error(f"Session {rk_session_id}: Error deprovisioning local Docker instances: {e}", exc_info=True)

        remote_instances_by_orchestrator_key: Dict[str, List[ManagedInstanceInfo]] = {}
        for inst in all_session_instances:
            if inst.orchestration_mode == "remote_http_api":
                key = self._get_orchestration_client_key_for_instance(inst)
                if key:
                    if key not in remote_instances_by_orchestrator_key:
                        remote_instances_by_orchestrator_key[key] = []
                    remote_instances_by_orchestrator_key[key].append(inst)
        
        for key, remote_instances_list in remote_instances_by_orchestrator_key.items():
            orchestrator = self._remote_http_orchestrators.get(key)
            if orchestrator and remote_instances_list:
                logger.info(f"Session {rk_session_id}: Deprovisioning {len(remote_instances_list)} remote instances for orchestrator '{key}'.")
                try:
                    await orchestrator.deprovision_instances(remote_instances_list)
                except Exception as e:
                    logger.error(f"Session {rk_session_id}: Error deprovisioning remote instances for '{key}': {e}", exc_info=True)
        
        # Cleanup temporary Docker images associated with the session data
        if session_data_to_clean.temporary_docker_images and self._local_docker_orchestrator:
            logger.info(f"Session {rk_session_id}: {len(session_data_to_clean.temporary_docker_images)} temporary Docker images were associated.")
            logger.info(f"Session {rk_session_id}: Image cleanup relies on LocalDockerOrchestrationClient.shutdown() or manual cleanup.")

        logger.info(f"Internal cleanup for session data (rk_session_id: '{rk_session_id}') complete.")

    async def _cleanup_session_actual(self, ctx: RequestContext, tool_args_dict: Dict[str, Any]) -> Dict[str, str]:
        """
        Cleans up all resources associated with the session data identified by rk_session_id.
        Called by the proxy handler. tool_args_dict should contain 'rk_session_id'.
        """
        rk_session_id = tool_args_dict.get("rk_session_id")
        logger.debug(f"_cleanup_session_actual called. ctx type: {type(ctx)}, rk_session_id: {rk_session_id}")

        if not isinstance(ctx, RequestContext):
            logger.error(f"CRITICAL: ctx is not RequestContext in _cleanup_session_actual. Type: {type(ctx)}.")
            raise TypeError(f"Expected RequestContext for ctx, got {type(ctx)}")
        
        if not rk_session_id:
            logger.error("'_cleanup_session_actual' called without 'rk_session_id' in tool_args_dict.")
            raise ValueError("'rk_session_id' argument missing for cleanup_session.")

        logger.info(f"_cleanup_session_actual called for rk_session_id '{rk_session_id}'.")

        session_data_obj = self.intermediary_session_data.pop(rk_session_id, None)

        if not session_data_obj:
            logger.warning(f"IntermediarySessionData for rk_session_id '{rk_session_id}' not found or already cleaned up.")
            return {"status": "custom_session_data_not_found_or_already_cleaned", "rk_session_id": rk_session_id}

        await self.cleanup_session_internal(session_data_obj, rk_session_id)
        
        logger.info(f"IntermediarySessionData for rk_session_id '{rk_session_id}' fully cleaned up.")
        return {"status": "cleaned", "rk_session_id": rk_session_id} # Return rk_session_id

    async def startup(self):
        """Override MCPServer's startup to initialize orchestrators, handlers, and shared instances."""
        
        logger.info("RewardKitIntermediaryServer performing custom startup tasks...")
        try:
            await self._initialize_orchestrators()
            await self._initialize_backend_handlers()
            await self._provision_shared_global_instances()
            logger.info("RewardKitIntermediaryServer custom startup tasks complete.")
        except Exception as e:
            logger.error(f"Error during RewardKitIntermediaryServer custom startup: {e}", exc_info=True)
            # Depending on policy, might want to re-raise to stop server launch
            raise

    async def _ping_actual(self, ctx: RequestContext, transport_session_id: str) -> Dict[str, str]:
        logger.debug(f"_ping_actual called. ctx type: {type(ctx)}, transport_session_id: {transport_session_id}")
        if not isinstance(ctx, RequestContext):
            logger.error(f"CRITICAL: ctx is not RequestContext in _ping_actual. Type: {type(ctx)}.")
            raise TypeError(f"Expected RequestContext for ctx, got {type(ctx)}")
            
        logger.info(f"Ping received by _ping_actual for transport_session_id: {transport_session_id}.")
        return {"reply": "pong", "session_id": transport_session_id}

    async def shutdown(self):
        """Custom shutdown logic for FastMCP based server."""
        logger.info("RewardKitIntermediaryServer (FastMCP based) performing custom shutdown tasks...")
        
        logger.info(f"Cleaning up any remaining IntermediarySessionData ({len(self.intermediary_session_data)} found)...")
        for session_id_key in list(self.intermediary_session_data.keys()): # Iterate over a copy of keys
            logger.info(f"Force cleaning IntermediarySessionData for {session_id_key} during server shutdown.")
            session_data_obj = self.intermediary_session_data.pop(session_id_key, None) # Pop the object
            if session_data_obj:
                await self.cleanup_session_internal(session_data_obj, session_id_key) # Pass object and key
            # No need for another pop as it's already done.
        
        shared_instances_to_deprovision = list(self._shared_global_instances.values())
        if shared_instances_to_deprovision:
            logger.info(f"Deprovisioning {len(shared_instances_to_deprovision)} shared global instances.")
            local_shared = [i for i in shared_instances_to_deprovision if i.orchestration_mode == "local_docker"]
            if local_shared and self._local_docker_orchestrator:
                 await self._local_docker_orchestrator.deprovision_instances(local_shared)
            
            remote_shared_by_key: Dict[str, List[ManagedInstanceInfo]] = {}
            for inst_info in shared_instances_to_deprovision:
                if inst_info.orchestration_mode == "remote_http_api":
                    key = self._get_orchestration_client_key_for_instance(inst_info)
                    if key:
                        remote_shared_by_key.setdefault(key, []).append(inst_info)
            
            for key, instances_list in remote_shared_by_key.items():
                orchestrator = self._remote_http_orchestrators.get(key)
                if orchestrator:
                    await orchestrator.deprovision_instances(instances_list)

        if self._local_docker_orchestrator:
            await self._local_docker_orchestrator.shutdown()
        for orch in self._remote_http_orchestrators.values():
            await orch.shutdown()
        
        logger.info("RewardKitIntermediaryServer custom shutdown tasks complete.")

    def _get_orchestration_client_key_for_instance(self, instance_info: ManagedInstanceInfo) -> Optional[str]:
        """Helper to find the orchestrator key for a remote instance."""
        if instance_info.orchestration_mode == "remote_http_api":
            backend_cfg = next((b for b in self.app_config.backends if b.backend_name_ref == instance_info.backend_name_ref), None)
            if backend_cfg:
                return backend_cfg.remote_api_config_ref or \
                       (backend_cfg.remote_api_config_inline.base_url if backend_cfg.remote_api_config_inline else None)
        return None
