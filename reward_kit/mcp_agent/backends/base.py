import abc
import logging
from typing import Any, List, Optional, TypedDict

from pydantic import BaseModel, Field

from reward_kit.mcp_agent.config import BackendServerConfig
from reward_kit.mcp_agent.orchestration.base_client import (
    AbstractOrchestrationClient,
    ManagedInstanceInfo,
)
from reward_kit.mcp_agent.session import IntermediarySessionData # Changed from IntermediarySession

logger = logging.getLogger(__name__)


class BackendInitRequest(BaseModel):
    """
    Represents a request from the client to initialize one or more instances
    of a specific backend type for a session.
    """
    backend_name_ref: str = Field(
        ...,
        description="The unique reference name of the backend configuration to use (must match one in AppConfig.backends).",
    )
    num_instances: int = Field(
        1, ge=1, description="Number of instances of this backend to provision for the session."
    )
    # orchestration_preference: Optional[Literal["local_docker", "remote_http_api"]] = Field(
    #     None, description="If the backend supports multiple orchestration modes, client can specify a preference."
    # ) # This might be too complex for client to decide, usually server config dictates.
    template_details: Optional[Any] = Field(
        None,
        description="Backend-specific details for initializing stateful instances from a template. E.g., a path to a data file for 'filesystem' or 'duckdb'. This can override or supplement template paths in BackendServerConfig.",
    )

    class Config:
        extra = "forbid"


class BackendInitResult(BaseModel):
    """
    Result for a single backend type initialization within a session.
    """
    backend_name_ref: str
    instances: List[ManagedInstanceInfo] # Provides client with necessary details to interact


class AbstractBackendHandler(abc.ABC):
    """
    Abstract base class for backend handlers.
    Backend handlers encapsulate any logic specific to a particular type of backend
    (e.g., filesystem, duckdb) primarily around interpreting template details
    and validating requests against server configuration.

    The core orchestration (provisioning, deprovisioning, tool calls) is delegated
    to an OrchestrationClient.
    """

    def __init__(self, backend_server_config: BackendServerConfig):
        self.server_config = backend_server_config
        logger.info(f"Initialized backend handler for type '{self.server_config.backend_type}' with ref '{self.server_config.backend_name_ref}'")

    @abc.abstractmethod
    async def initialize_session_instances(
        self,
        session_data: IntermediarySessionData, # Changed from session: IntermediarySession
        init_request: BackendInitRequest,
        orchestration_client: AbstractOrchestrationClient,
    ) -> List[ManagedInstanceInfo]:
        """
        Handles the initialization of instances for this backend type within a given session.
        It validates the request and then delegates to the orchestration_client.

        Args:
            session_data: The IntermediarySessionData object for the current session.
            init_request: The client's request for this backend.
            orchestration_client: The client (Docker or Remote HTTP) responsible for actual provisioning.

        Returns:
            A list of ManagedInstanceInfo objects for the provisioned instances.
        """
        pass

    async def cleanup_session_instances(
        self,
        session_data: IntermediarySessionData, # Changed from session: IntermediarySession
        orchestration_client: AbstractOrchestrationClient,
    ) -> None:
        """
        Handles the cleanup of all instances of this backend type within a given session.
        Delegates to the orchestration_client.

        Args:
            session_data: The IntermediarySessionData object for the session being cleaned up.
            orchestration_client: The client responsible for actual deprovisioning.
        """
        instances_to_cleanup = session_data.get_managed_instances( # Changed from session.get_managed_instances
            backend_name_ref=self.server_config.backend_name_ref
        )
        if not instances_to_cleanup:
            logger.info(
                f"Session {session_data.session_id}: No instances of backend '{self.server_config.backend_name_ref}' to cleanup." # Changed from session.session_id
            )
            return

        logger.info(
            f"Session {session_data.session_id}: Cleaning up {len(instances_to_cleanup)} instances of backend '{self.server_config.backend_name_ref}'." # Changed from session.session_id
        )
        try:
            await orchestration_client.deprovision_instances(instances_to_cleanup)
            logger.info(
                f"Session {session_data.session_id}: Successfully requested deprovisioning for instances of '{self.server_config.backend_name_ref}'." # Changed from session.session_id
            )
        except Exception as e:
            logger.error(
                f"Session {session_data.session_id}: Error during cleanup of instances for backend '{self.server_config.backend_name_ref}': {e}", # Changed from session.session_id
                exc_info=True
            )
            # Decide if to re-raise or just log. For cleanup, usually log and continue.

    def validate_init_request(self, init_request: BackendInitRequest) -> None:
        """
        Validates the BackendInitRequest against the server's configuration for this backend.
        Subclasses can override to add more specific validation.
        """
        if init_request.backend_name_ref != self.server_config.backend_name_ref:
            raise ValueError(
                f"BackendInitRequest.backend_name_ref '{init_request.backend_name_ref}' "
                f"does not match handler's configured ref '{self.server_config.backend_name_ref}'."
            )
        
        if self.server_config.instance_scoping == "shared_global" and init_request.num_instances > 1:
            logger.warning(f"Requested {init_request.num_instances} for a 'shared_global' backend '{self.server_config.backend_name_ref}'. Only one shared instance will be used/created.")
            # The orchestration client for shared_global should handle this, ensuring only one is active.
            # The num_instances for shared_global is more of an indication that the client *wants* to use it.

        # Further validation can be added, e.g., for template_details compatibility.
        logger.debug(f"BackendInitRequest for '{init_request.backend_name_ref}' validated successfully.")
