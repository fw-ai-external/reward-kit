"""
FrozenLake Environment Adapter

This adapter implements the EnvironmentAdapter interface for FrozenLake environments,
enabling integration with the MCP-Gym framework.
"""

from typing import Any, Dict, Optional, Tuple

from gymnasium.envs.toy_text import FrozenLakeEnv
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from reward_kit.mcp import EnvironmentAdapter


class FrozenLakeAdapter(EnvironmentAdapter):
    """FrozenLake adapter for MCP-Gym framework."""

    ACTION_NAMES = ["LEFT", "DOWN", "RIGHT", "UP"]

    def create_environment(
        self, config: Optional[Dict[str, Any]] = None
    ) -> FrozenLakeEnv:
        """
        Create FrozenLake environment.

        Args:
            config: Configuration dictionary with optional 'map_name' and 'seed'

        Returns:
            FrozenLake environment instance
        """
        config = config or {}

        # Determine grid size from config
        grid_size = 4
        if "map_name" in config:
            if "8x8" in config["map_name"]:
                grid_size = 8

        # Generate random map if seed is provided
        seed = config.get("seed")
        if seed is not None:
            desc = generate_random_map(size=grid_size, p=0.8, seed=seed)
        else:
            desc = generate_random_map(size=grid_size, p=0.8)

        return FrozenLakeEnv(desc=desc, is_slippery=False, render_mode="ansi")

    def create_environment_with_seed(
        self, config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None
    ) -> Tuple[FrozenLakeEnv, int, Dict[str, Any]]:
        """
        Create FrozenLake environment with seed and return initial state.

        Args:
            config: Configuration dictionary
            seed: Seed for reproducible environments

        Returns:
            Tuple of (environment, initial_observation, initial_info)
        """
        config = config or {}

        # Add seed to config for environment creation
        env_config = {**config, "seed": seed}

        env = self.create_environment(env_config)
        obs, info = env.reset(seed=seed)

        return env, obs, info

    def reset_environment(
        self, env: FrozenLakeEnv, seed: Optional[int] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Reset environment.

        Args:
            env: Environment instance
            seed: Optional seed for reset

        Returns:
            Tuple of (observation, info)
        """
        return env.reset(seed=seed)

    def step_environment(
        self, env: FrozenLakeEnv, action: int
    ) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        Execute environment step.

        Args:
            env: Environment instance
            action: Action index

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        return env.step(action)

    def close_environment(self, env: FrozenLakeEnv) -> None:
        """
        Close environment.

        Args:
            env: Environment instance
        """
        # FrozenLake doesn't need explicit cleanup
        pass

    def parse_action(self, action_str: str) -> int:
        """
        Parse action string to integer.

        Args:
            action_str: Action string (LEFT, DOWN, RIGHT, UP)

        Returns:
            Action index

        Raises:
            ValueError: If action is invalid
        """
        action_str = action_str.strip().upper()
        if action_str not in self.ACTION_NAMES:
            raise ValueError(
                f"Invalid action '{action_str}'. Valid actions: {self.ACTION_NAMES}"
            )
        return self.ACTION_NAMES.index(action_str)

    def format_observation(self, observation: int) -> int:
        """
        Format observation for JSON serialization.

        Args:
            observation: Raw observation from environment

        Returns:
            Formatted observation
        """
        return int(observation)

    def get_action_space_description(self) -> Dict[str, Any]:
        """
        Get action space description.

        Returns:
            Action space description dictionary
        """
        return {
            "type": "discrete",
            "actions": self.ACTION_NAMES,
            "description": "Move actions: LEFT(0), DOWN(1), RIGHT(2), UP(3)",
        }

    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "map_name": "4x4",
            "is_slippery": False,
        }
