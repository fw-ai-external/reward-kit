import logging
from typing import List

from reward_kit.mcp_agent.backends.base import (
    AbstractBackendHandler,
    BackendInitRequest,
)
from reward_kit.mcp_agent.orchestration.base_client import (
    AbstractOrchestrationClient,
    ManagedInstanceInfo,
)
from reward_kit.mcp_agent.session import IntermediarySession

logger = logging.getLogger(__name__)


class GenericBackendHandler(AbstractBackendHandler):
    """
    A generic backend handler that can be used for backend types that
    do not require special processing of template_details beyond what the
    OrchestrationClient handles.
    """

    async def initialize_session_instances(
        self,
        session: IntermediarySession,
        init_request: BackendInitRequest,
        orchestration_client: AbstractOrchestrationClient,
    ) -> List[ManagedInstanceInfo]:
        """
        Validates the request and delegates to the orchestration_client to provision instances.
        Passes template_details from the init_request directly to the orchestration_client.
        """
        self.validate_init_request(init_request)

        logger.info(
            f"Session {session.session_id}: Initializing instances for backend '{init_request.backend_name_ref}' "
            f"using generic handler. Num instances: {init_request.num_instances}."
        )

        # Determine effective number of instances, especially for shared_global
        effective_num_instances = init_request.num_instances
        if self.server_config.instance_scoping == "shared_global":
            # For shared_global, we typically only want to ensure one instance exists.
            # The orchestration client should ideally handle idempotency if multiple requests for a shared global come.
            # Here, we might cap num_instances to 1 for shared_global if it's managed per session request,
            # or rely on a global cache in the server for shared instances.
            # For now, let the orchestration client manage the "shared" aspect.
            # If multiple "num_instances" are requested for shared, it might mean multiple handles to the same shared resource.
            # The current design of provision_instances might create multiple if not careful for shared.
            # This needs to be handled by the OrchestrationClient or the server's global shared instance management.
            # For simplicity in the handler, we pass num_instances as requested.
            # The LocalDockerOrchestrationClient/RemoteClient should be smart about shared_global.
            pass


        provisioned_instances = await orchestration_client.provision_instances(
            backend_config=self.server_config,
            num_instances=effective_num_instances,
            session_id=session.session_id,
            template_details=init_request.template_details, # Pass through
        )

        logger.info(
            f"Session {session.session_id}: Provisioned {len(provisioned_instances)} instances "
            f"for backend '{init_request.backend_name_ref}' via generic handler."
        )
        return provisioned_instances
