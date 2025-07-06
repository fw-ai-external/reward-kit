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
        """
        Create FrozenLake environment with random map generation enabled.
        """
        # Determine grid size from config (default to 4x4)
        grid_size = 4
        if config and ("map_name" in config or "grid_type" in config):
            # Support both map_name and grid_type for backward compatibility
            size_key = config.get("map_name") or config.get("grid_type")
            if size_key and "8x8" in size_key:
                grid_size = 8
            elif size_key and "4x4" in size_key:
                grid_size = 4

        # Use random map generation instead of predefined maps
        # This allows different seeds to produce different layouts
        # Determine the map name for FrozenLake
        if grid_size == 8:
            env_map_name = "8x8"
        else:
            env_map_name = "4x4"

        return FrozenLakeEnv(
            desc=None,  # Enable random map generation
            map_name=env_map_name,  # Use determined grid size
            is_slippery=False,
        )

    def create_environment_with_seed(
        self, config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None
    ) -> Tuple[FrozenLakeEnv, int, Dict[str, Any]]:
        """
        Create FrozenLake environment with proper seeding and return initial state.

        Uses FrozenLake's random map generation with proper seeding.

        Returns:
            Tuple of (environment, initial_observation, initial_info)
        """
        # Determine grid size from config (default to 4x4 for easier testing)
        grid_size = 4
        if config and ("map_name" in config or "grid_type" in config):
            # Support both map_name and grid_type for backward compatibility
            size_key = config.get("map_name") or config.get("grid_type")
            if size_key and "8x8" in size_key:
                grid_size = 8
            elif size_key and "4x4" in size_key:
                grid_size = 4

        # Generate random map with seed for reproducible environments
        # This is the key fix - we need to generate the map description using the seed
        from gymnasium.envs.toy_text.frozen_lake import generate_random_map

        # Generate random map with seed
        if seed is not None:
            desc = generate_random_map(size=grid_size, p=0.8, seed=seed)
        else:
            desc = generate_random_map(size=grid_size, p=0.8)

        # Create environment with the generated map description
        env = FrozenLakeEnv(
            desc=desc, is_slippery=False  # Use the randomly generated map
        )

        # Reset with seed to ensure reproducible starting position
        obs, info = env.reset(seed=seed)

        return env, obs, info

    def reset_environment(
        self, env: FrozenLakeEnv, seed: Optional[int] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Reset environment.

        Note: For random map environments, the map is already generated during
        create_environment(). This reset only affects the agent's starting position
        and episode state, not the map layout.
        """
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
            "desc": None,  # Enable random map generation
            "map_name": None,  # Don't use predefined maps
            "is_slippery": False,  # Keep deterministic movement
        }
