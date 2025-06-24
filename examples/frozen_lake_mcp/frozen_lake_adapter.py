"""
Simplified FrozenLake Environment Adapter

Minimal adapter for demonstration purposes.
For production use, see the full framework adapter.
"""

from typing import Any, Dict, Optional, Tuple

from gymnasium.envs.toy_text import FrozenLakeEnv

from reward_kit.mcp import EnvironmentAdapter


class FrozenLakeAdapter(EnvironmentAdapter):
    """Minimal FrozenLake adapter for MCP framework demo."""

    ACTION_NAMES = ["LEFT", "DOWN", "RIGHT", "UP"]

    def create_environment(
        self, config: Optional[Dict[str, Any]] = None
    ) -> FrozenLakeEnv:
        """Create FrozenLake environment."""
        return FrozenLakeEnv(map_name="4x4", is_slippery=False)

    def reset_environment(
        self, env: FrozenLakeEnv, seed: Optional[int] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """Reset environment."""
        return env.reset(seed=seed)

    def step_environment(
        self, env: FrozenLakeEnv, action: int
    ) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """Execute environment step."""
        return env.step(action)

    def close_environment(self, env: FrozenLakeEnv) -> None:
        """Close environment."""
        pass  # FrozenLake doesn't need explicit cleanup

    def parse_action(self, action_str: str) -> int:
        """Parse action string to integer."""
        if action_str not in self.ACTION_NAMES:
            raise ValueError(
                f"Invalid action '{action_str}'. Must be one of {self.ACTION_NAMES}"
            )
        return self.ACTION_NAMES.index(action_str)

    def format_observation(self, observation: int) -> int:
        """Format observation for JSON."""
        return int(observation)

    def get_action_space_description(self) -> Dict[str, Any]:
        """Get action space description."""
        return {
            "type": "discrete",
            "actions": self.ACTION_NAMES,
            "description": "Move actions: LEFT(0), DOWN(1), RIGHT(2), UP(3)",
        }

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for FrozenLake."""
        return {
            "map_name": "4x4",
            "is_slippery": False,
        }
