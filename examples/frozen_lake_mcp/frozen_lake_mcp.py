"""
FrozenLake MCP-Gym Implementation

This module implements the north star vision for MCP-Gym environments,
providing a clean, simple implementation of FrozenLake using the McpGym base class.

Key Features:
- Strict data/control plane separation
- Data plane: Tool responses contain only observations
- Control plane: Rewards/termination available via MCP resources (control://reward, control://status)

Example usage:
    from frozen_lake_mcp import FrozenLakeMcp

    server = FrozenLakeMcp(seed=42)
    server.run()
"""

from typing import Any, Dict, Optional

from frozen_lake_adapter import FrozenLakeAdapter
from mcp.server.fastmcp import Context

from reward_kit.mcp import McpGym


class FrozenLakeMcp(McpGym):
    """
    FrozenLake MCP-Gym environment implementing the north star vision.

    This demonstrates the clean, simple API for MCP-Gym environments:
    - Inherit from McpGym (which inherits from GymProductionServer)
    - Use proper EnvironmentAdapter pattern
    - Register tools with @self.mcp.tool() decorator
    - Compatible with CondaServerProcessManager
    - Strict data/control plane separation via MCP resources
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize FrozenLake MCP-Gym environment."""
        print(f"🌱 BACKEND INIT: ===== FrozenLakeMcp.__init__ CALLED =====")
        print(f"🌱 BACKEND INIT: Received seed parameter: {seed}")
        print(f"🌱 BACKEND INIT: Seed type: {type(seed)}")

        adapter = FrozenLakeAdapter()
        print(f"🔧 BACKEND INIT: Created FrozenLakeAdapter: {adapter}")

        print(
            f"🚀 BACKEND INIT: Calling super().__init__('FrozenLake-v1', adapter, seed={seed})"
        )
        super().__init__("FrozenLake-v1", adapter, seed)

        print(f"✅ BACKEND INIT: super().__init__ completed")
        print(f"🎮 BACKEND INIT: Final environment state:")
        print(f"  - Environment object: {self.env}")
        print(f"  - Environment type: {type(self.env)}")
        print(f"  - Initial observation: {self.obs}")
        print(f"  - Environment desc: {getattr(self.env, 'desc', 'N/A')}")

        # Render the initial grid to see what map was actually generated
        if hasattr(self.env, "render"):
            try:
                grid = self.env.render()
                print(f"🗺️  BACKEND INIT: Generated map:\n{grid}")
            except Exception as e:
                print(f"❌ BACKEND INIT: Could not render map: {e}")

        print(f"🌱 BACKEND INIT: ===== FrozenLakeMcp.__init__ COMPLETED =====")

    def _register_tools(self):
        """Register domain-specific MCP tools."""

        @self.mcp.tool(
            name="lake_move",
            description="Move on the frozen lake. Actions: LEFT, DOWN, RIGHT, UP. "
            "Check control://reward and control://status resources for rewards and termination.",
        )
        def lake_move(action: str, ctx: Context) -> Dict[str, Any]:
            """
            Move in the FrozenLake environment.

            Args:
                action: Direction to move (LEFT, DOWN, RIGHT, UP)
                ctx: MCP context (proper FastMCP context)

            Returns:
                Dictionary with observation data ONLY (data plane).
                Rewards and termination info available via control plane resources.
            """
            print(f"🔧 BACKEND: ===== lake_move CALLED =====")
            print(f"🔧 BACKEND: Received lake_move(action='{action}')")
            print(f"🎮 BACKEND: Current state BEFORE move:")
            print(f"  - Position: {self.obs}")
            print(f"  - Step count: {self.control_plane_state['step_count']}")
            print(f"  - Total reward: {self.control_plane_state['total_reward']}")
            print(f"  - Terminated: {self.control_plane_state['terminated']}")

            # Validate action
            if not action or not isinstance(action, str):
                raise ValueError(
                    f"Invalid action parameter: '{action}'. "
                    f"Must be a non-empty string. Valid actions: LEFT, DOWN, RIGHT, UP"
                )

            action = action.strip().upper()
            print(f"🎯 BACKEND: Parsed action: '{action}'")

            # Parse action
            try:
                action_int = self.adapter.parse_action(action)
                print(f"🔢 BACKEND: Action integer: {action_int}")
            except ValueError as e:
                print(f"❌ BACKEND: Action parse error: {e}")
                raise ValueError(str(e))

            # Use base class method to execute step and update control plane
            print(f"🚀 BACKEND: Executing environment step with action {action_int}")
            observation_data = self._execute_environment_step(action_int)

            # Add the action to the response for context
            observation_data["action"] = action

            # Log detailed move information including state progression
            print(
                f"🎮 BACKEND RESULT: {action} → position {self.obs} (step {self.control_plane_state['step_count']})"
            )
            print(f"📤 BACKEND: Returning observation data: {observation_data}")
            print(f"🔧 BACKEND: ===== lake_move COMPLETED =====")

            # Return ONLY data plane information (no rewards/termination)
            return observation_data

    @staticmethod
    def format_observation(obs: int, env: Any) -> Dict[str, Any]:
        """Format observation for MCP response (data plane only)."""
        return {
            "position": int(obs),
            "grid": env.render(),
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the FrozenLake MCP-Gym environment
    print("Creating FrozenLake MCP-Gym server...")
    server = FrozenLakeMcp(seed=42)

    print("Server created successfully!")
    print(f"Initial observation: {server.obs}")
    print(f"Environment adapter: {server.adapter.__class__.__name__}")
    print("\n🎛️  Control plane resources available:")
    print("  - control://reward (current reward and step count)")
    print("  - control://status (termination status and total reward)")
    print("  - control://info (environment info)")

    # Run the server
    print("\nStarting MCP server...")
    server.run()
