import logging
from typing import Dict, List, Set

from reward_kit.mcp_agent.orchestration.base_client import ManagedInstanceInfo

logger = logging.getLogger(__name__)

# Placeholder for the actual BaseSession from the MCP Python SDK
# from mcp.server import BaseSession
class BaseSession:
    """
    Placeholder for mcp.server.BaseSession.
    Actual implementation will depend on the MCP Python SDK.
    """
    def __init__(self, session_id: str, app: any = None):
        self.session_id = session_id
        self.app = app # Typically the server application instance
        logger.debug(f"BaseSession (placeholder) initialized with session_id: {session_id}")

    async def cleanup(self):
        """Placeholder for cleanup logic in BaseSession."""
        logger.debug(f"BaseSession (placeholder) cleanup called for session_id: {self.session_id}")
        pass


class IntermediarySession(BaseSession):
    """
    Custom session class for the RewardKit Intermediary MCP Server.
    Manages the state of backend instances associated with a client session.
    """

    def __init__(self, session_id: str, app: any = None):
        super().__init__(session_id, app)
        # Stores ManagedInstanceInfo objects, keyed by backend_name_ref,
        # then by a list of instances for that backend.
        self.managed_backends: Dict[str, List[ManagedInstanceInfo]] = {}
        # Tracks temporary Docker images created specifically for this session
        # to ensure they are cleaned up when the session ends.
        self.temporary_docker_images: Set[str] = set()
        logger.info(f"IntermediarySession created with ID: {self.session_id}")

    def add_managed_instances(
        self, backend_name_ref: str, instances: List[ManagedInstanceInfo]
    ):
        """Adds a list of managed instances for a given backend reference."""
        if backend_name_ref not in self.managed_backends:
            self.managed_backends[backend_name_ref] = []
        self.managed_backends[backend_name_ref].extend(instances)
        logger.info(
            f"Session {self.session_id}: Added {len(instances)} instances for backend '{backend_name_ref}'."
        )
        for instance in instances:
            if instance.committed_image_tag:
                self.temporary_docker_images.add(instance.committed_image_tag)
                logger.debug(f"Session {self.session_id}: Tracking temporary image '{instance.committed_image_tag}'.")


    def get_managed_instances(
        self, backend_name_ref: str, instance_id: Optional[str] = None
    ) -> List[ManagedInstanceInfo]:
        """
        Retrieves managed instances for a backend reference.
        If instance_id is provided, returns a list containing that specific instance (if found).
        Otherwise, returns all instances for the backend_name_ref.
        """
        backend_instances = self.managed_backends.get(backend_name_ref, [])
        if not backend_instances:
            return []

        if instance_id:
            for inst in backend_instances:
                if inst.instance_id == instance_id:
                    return [inst]
            return []  # Specific instance_id not found
        
        return backend_instances # Return all for the backend_name_ref

    def get_all_managed_instances(self) -> List[ManagedInstanceInfo]:
        """Returns a flat list of all managed instances in this session."""
        all_instances = []
        for instances in self.managed_backends.values():
            all_instances.extend(instances)
        return all_instances

    async def cleanup(self):
        """
        Cleanup logic for the IntermediarySession.
        This will be called by the MCP server when the session expires or is explicitly closed.
        It should trigger the deprovisioning of all managed backend instances.
        """
        logger.info(f"Initiating cleanup for IntermediarySession: {self.session_id}")
        
        # The actual deprovisioning logic (calling OrchestrationClient.deprovision_instances)
        # will be handled by the RewardKitIntermediaryServer's cleanup_session tool handler,
        # which will use the information stored in this session object.
        # This method is more of a hook provided by BaseSession.
        
        # For example, if the server app is available and has a method:
        # if hasattr(self.app, 'trigger_session_resource_cleanup'):
        # await self.app.trigger_session_resource_cleanup(self)
        
        # For now, just log. The server's tool handler for `cleanup_session` will iterate
        # through self.get_all_managed_instances() and call the appropriate
        # OrchestrationClient.deprovision_instances().
        # It will also handle removing temporary Docker images listed in self.temporary_docker_images.

        logger.info(f"IntermediarySession {self.session_id}: Cleanup hook executed. Actual deprovisioning is handled by server's cleanup tool.")
        await super().cleanup()
