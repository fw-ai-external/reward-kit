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
        print(f"🏗️  ADAPTER: ===== create_environment CALLED =====")
        print(f"🏗️  ADAPTER: Received config: {config}")

        config = config or {}

        # Determine grid size from config
        grid_size = 4
        if "map_name" in config:
            if "8x8" in config["map_name"]:
                grid_size = 8
        print(f"🏗️  ADAPTER: Grid size: {grid_size}")

        # Generate random map if seed is provided
        seed = config.get("seed")
        print(f"🌱 ADAPTER: Extracted seed from config: {seed}")
        print(f"🌱 ADAPTER: Seed type: {type(seed)}")

        if seed is not None:
            print(f"🎯 ADAPTER: Generating map with seed {seed}")
            desc = generate_random_map(size=grid_size, p=0.8, seed=seed)
            print(f"🗺️  ADAPTER: Generated map desc with seed {seed}:")
            map_str = "\n".join(["".join(row) for row in desc])
            print(f"{map_str}")
        else:
            print(f"⚠️  ADAPTER: No seed provided, generating random map")
            desc = generate_random_map(size=grid_size, p=0.8)
            print(f"🗺️  ADAPTER: Generated map desc without seed:")
            map_str = "\n".join(["".join(row) for row in desc])
            print(f"{map_str}")

        print(
            f"🏗️  ADAPTER: Creating FrozenLakeEnv with desc={len(desc)}x{len(desc[0])}"
        )
        env = FrozenLakeEnv(desc=desc, is_slippery=False, render_mode="ansi")
        print(f"🏗️  ADAPTER: FrozenLakeEnv created: {env}")
        print(f"🏗️  ADAPTER: ===== create_environment COMPLETED =====")

        return env

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
        print(f"🌱 ADAPTER: ===== create_environment_with_seed CALLED =====")
        print(f"🌱 ADAPTER: Received config: {config}")
        print(f"🌱 ADAPTER: Received seed: {seed}")
        print(f"🌱 ADAPTER: Seed type: {type(seed)}")

        config = config or {}

        # Add seed to config for environment creation
        env_config = {**config, "seed": seed}
        print(f"🔧 ADAPTER: Enhanced config with seed: {env_config}")

        print(f"🚀 ADAPTER: Calling create_environment with config: {env_config}")
        env = self.create_environment(env_config)
        print(f"✅ ADAPTER: create_environment returned: {env}")

        print(f"🔄 ADAPTER: Resetting environment with seed: {seed}")
        obs, info = env.reset(seed=seed)
        print(f"🔄 ADAPTER: Reset returned obs={obs}, info={info}")

        print(f"🌱 ADAPTER: ===== create_environment_with_seed COMPLETED =====")
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
        print(f"🔄 ADAPTER: ===== reset_environment CALLED =====")
        print(f"🔄 ADAPTER: Environment: {env}")
        print(f"🔄 ADAPTER: Seed: {seed}")

        result = env.reset(seed=seed)
        print(f"🔄 ADAPTER: Reset result: {result}")
        print(f"🔄 ADAPTER: ===== reset_environment COMPLETED =====")

        return result

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
        print(f"🏃 ADAPTER: ===== step_environment CALLED =====")
        print(f"🏃 ADAPTER: Environment ID: {id(env)}")
        print(f"🏃 ADAPTER: Environment type: {type(env)}")
        print(f"🏃 ADAPTER: Action: {action}")
        print(f"🏃 ADAPTER: Environment state before step:")
        print(f"  - Current position (env.s): {getattr(env, 's', 'N/A')}")
        print(f"  - Last action (env.lastaction): {getattr(env, 'lastaction', 'N/A')}")
        print(f"  - Environment desc: {getattr(env, 'desc', 'N/A')}")
        print(f"  - Environment nrow: {getattr(env, 'nrow', 'N/A')}")
        print(f"  - Environment ncol: {getattr(env, 'ncol', 'N/A')}")

        # Execute the actual step
        print(f"🚀 ADAPTER: Calling env.step({action})")
        step_result = env.step(action)
        print(f"🚀 ADAPTER: env.step returned: {step_result}")

        # Unpack and log the result
        obs, reward, terminated, truncated, info = step_result
        print(f"🎯 ADAPTER: Step result details:")
        print(f"  - Observation: {obs}")
        print(f"  - Reward: {reward}")
        print(f"  - Terminated: {terminated}")
        print(f"  - Truncated: {truncated}")
        print(f"  - Info: {info}")

        print(f"🏃 ADAPTER: Environment state after step:")
        print(f"  - New position (env.s): {getattr(env, 's', 'N/A')}")
        print(f"  - Last action (env.lastaction): {getattr(env, 'lastaction', 'N/A')}")
        print(f"🏃 ADAPTER: ===== step_environment COMPLETED =====")

        return step_result

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
