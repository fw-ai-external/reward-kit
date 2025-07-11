"""
FrozenLake MCP-Gym Implementation

This module implements the north star vision for MCP-Gym environments,
providing a clean, simple implementation of FrozenLake using the McpGym base class.

Key Features:
- Multi-session support with session-based control plane state
- Data plane: Tool responses contain only observations
- Control plane: Server-side state management keyed by session ID
- Rollout system can query control plane state for termination logic

Example usage:
    from frozen_lake_mcp import FrozenLakeMcp

    server = FrozenLakeMcp(seed=42)
    server.run()
"""

from typing import Any, Dict, Optional

from frozen_lake_adapter import FrozenLakeAdapter
from mcp.server.fastmcp import Context

from reward_kit.mcp import McpGym
from reward_kit.mcp.mcpgym import control_plane_endpoint


class FrozenLakeMcp(McpGym):
    """
    FrozenLake MCP-Gym environment implementing the north star vision.

    This demonstrates the clean, simple API for MCP-Gym environments:
    - Inherit from McpGym (which inherits from GymProductionServer)
    - Use proper EnvironmentAdapter pattern
    - Register tools with @self.mcp.tool() decorator
    - Compatible with CondaServerProcessManager
    - Multi-session support with session-based control plane state
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize FrozenLake MCP-Gym environment."""
        adapter = FrozenLakeAdapter()
        super().__init__("FrozenLake-v1", adapter, seed)

        # Multi-session support is now handled by the base class

    # Session management methods are now handled by the base class

    def _register_tools(self):
        """Register domain-specific MCP tools."""

        @self.mcp.tool(
            name="lake_move",
            description="Move on the frozen lake. Actions: LEFT, DOWN, RIGHT, UP. "
            "Returns only observation data; control plane state managed server-side.",
        )
        def lake_move(action: str, ctx: Context) -> Dict[str, Any]:
            """
            Move in the FrozenLake environment.

            Args:
                action: Direction to move (LEFT, DOWN, RIGHT, UP)
                ctx: MCP context (proper FastMCP context)

            Returns:
                Dictionary with observation data ONLY (data plane).
                Control plane state managed server-side per session.
            """
            # Validate action
            if not action or not isinstance(action, str):
                raise ValueError(
                    f"Invalid action parameter: '{action}'. "
                    f"Must be a non-empty string. Valid actions: LEFT, DOWN, RIGHT, UP"
                )

            action = action.strip().upper()

            # Parse action
            try:
                action_int = self.adapter.parse_action(action)
            except ValueError as e:
                raise ValueError(str(e))

            # Get session ID and session data
            session_id = self._get_session_id(ctx)
            session_data = self._get_or_create_session(ctx)

            # Execute environment step using base class method
            observation_data = self._execute_session_environment_step(
                session_id, action_int
            )
            observation_data["action"] = action

            # Log move (no control plane data in logs)
            print(
                f"🎮 Session {session_id[:16]}...: {action} → position {session_data['obs']}"
            )

            return observation_data

        @self.mcp.tool(
            name="get_control_plane_state",
            description="Get current control plane state for this session (for rollout system).",
        )
        def get_control_plane_state(ctx: Context) -> Dict[str, Any]:
            """
            Get control plane state for current session.

            Args:
                ctx: MCP context

            Returns:
                Control plane state dictionary
            """
            session_id = self._get_session_id(ctx)
            control_state = self.get_control_plane_state(session_id)

            if control_state is None:
                # Initialize session if it doesn't exist
                session_data = self._get_or_create_session(ctx)
                control_state = self._get_or_create_session_control_plane(session_id)

            return control_state

    @staticmethod
    def format_observation(obs: int, env: Any) -> Dict[str, Any]:
        """Format observation for MCP response (data plane only)."""
        return {
            "position": int(obs),
            "grid": env.render(),
        }


# Example usage and testing
# if __name__ == "__main__":
#     # Test the FrozenLake MCP-Gym environment
#     print("Creating FrozenLake MCP-Gym server...")
#     server = FrozenLakeMcp(seed=42)
#
#     print("Server created successfully!")
#     print(f"Environment adapter: {server.adapter.__class__.__name__}")
#     print("\n🎛️  Multi-session control plane features:")
#     print("  - Session-based environment isolation")
#     print("  - Server-side control plane state management")
#     print("  - get_control_plane_state tool for rollout system")
#     print("  - Data plane tools return observations only")
#
#     # Run the server
#     print("\nStarting MCP server...")
#     server.run()
