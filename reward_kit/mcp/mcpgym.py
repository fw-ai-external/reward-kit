"""
MCP-Gym Framework - North Star Implementation

This module provides the core McpGym base class that implements the north star vision
for universal RL environment integration via MCP protocol.

Key Features:
- Inherits from GymProductionServer for proper MCP integration
- Simple tool registration with @self.mcp.tool() decorator
- Clean separation between data plane (MCP tool calls) and control plane (MCP resources)
- Compatible with CondaServerProcessManager
- Control plane data stored in MCP resources (control://reward, control://status)
"""

import json
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

from mcp.server.fastmcp import Context

from .adapter import EnvironmentAdapter
from .gym_production_server import GymProductionServer


class McpGym(GymProductionServer):
    """
    Base class for MCP-Gym environments implementing the north star vision.

    This class provides the universal adapter pattern for RL environments,
    bridging training infrastructure, production MCP standards, and high-quality
    environments through a clean, standardized interface.

    Key Design Principles:
    - Data Plane: JSON tool calls/responses via MCP (state transitions/actions)
    - Control Plane: Rewards/termination signals via MCP resources
    - Environment Implementation: Single-process MCP server per environment
    - Inherits from GymProductionServer for proper MCP protocol handling
    """

    def __init__(
        self, server_name: str, adapter: EnvironmentAdapter, seed: Optional[int] = None
    ):
        """
        Initialize MCP-Gym environment.

        Args:
            server_name: Name for the MCP server
            adapter: Environment adapter instance
            seed: Optional seed for reproducible environments
        """
        super().__init__(server_name, adapter)

        # Initialize control plane state
        self.control_plane_state = {
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": {},
            "step_count": 0,
            "total_reward": 0.0,
        }

        # Reset with seed if provided
        if seed is not None:
            self.env, self.obs, _info = self._new_env(seed=seed)

        # Register control plane resources
        self._register_control_plane_resources()

    def _register_control_plane_resources(self):
        """Register MCP resources for control plane information."""

        @self.mcp.resource("control://reward")
        def current_reward() -> str:
            """Provide current reward information via MCP resource."""
            return json.dumps(
                {
                    "reward": self.control_plane_state["reward"],
                    "step_count": self.control_plane_state["step_count"],
                }
            )

        @self.mcp.resource("control://status")
        def current_status() -> str:
            """Provide current episode status via MCP resource."""
            return json.dumps(
                {
                    "terminated": self.control_plane_state["terminated"],
                    "truncated": self.control_plane_state["truncated"],
                    "step_count": self.control_plane_state["step_count"],
                    "total_reward": self.control_plane_state["total_reward"],
                }
            )

        @self.mcp.resource("control://info")
        def current_info() -> str:
            """Provide current environment info via MCP resource."""
            return json.dumps(self.control_plane_state["info"])

    def _update_control_plane(
        self, reward: float, terminated: bool, truncated: bool, info: Dict[str, Any]
    ):
        """
        Update control plane state after environment step.

        Args:
            reward: Reward from environment step
            terminated: Whether episode terminated
            truncated: Whether episode truncated
            info: Info dictionary from environment
        """
        self.control_plane_state["reward"] = reward
        self.control_plane_state["terminated"] = terminated
        self.control_plane_state["truncated"] = truncated
        self.control_plane_state["info"] = info
        self.control_plane_state["step_count"] += 1
        self.control_plane_state["total_reward"] += reward

        # Log control plane update (for debugging)
        print(
            f"🎛️  Control plane updated: reward={reward}, terminated={terminated}, step={self.control_plane_state['step_count']}"
        )

    def _execute_environment_step(self, action_int: int) -> Dict[str, Any]:
        """
        Execute environment step and update control plane.

        Args:
            action_int: Parsed action integer

        Returns:
            Data plane response (observation only, no rewards)
        """
        # Execute environment step
        obs, reward, terminated, truncated, info = self.adapter.step_environment(
            self.env, action_int
        )

        # Update global observation state
        self.obs = obs

        # Update control plane (separate from data plane)
        self._update_control_plane(reward, terminated, truncated, info)

        # Return ONLY data plane information (no rewards/termination)
        return self._render(obs)

    @abstractmethod
    def _register_tools(self):
        """
        Register domain-specific MCP tools.

        Subclasses must implement this method to register their specific tools
        using the @self.mcp.tool() decorator pattern.

        IMPORTANT: Tools should only return data plane information (observations).
        Control plane information (rewards, termination) is available via resources.
        """
        pass

    @staticmethod
    @abstractmethod
    def format_observation(obs: Any, env: Any) -> Dict[str, Any]:
        """
        Format observation for MCP response.

        Args:
            obs: Raw observation from environment
            env: Environment instance

        Returns:
            Formatted observation dictionary (DATA PLANE ONLY)
        """
        pass
